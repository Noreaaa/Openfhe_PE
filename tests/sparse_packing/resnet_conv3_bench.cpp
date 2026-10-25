#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/cnn/conv2d.hpp"
#include "../../src/cnn/activation.hpp"
#include "../../src/cnn/sparse_activation.hpp"
#include "../../src/encryption/encryption.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/dataloader.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <omp.h>

static double rssMiB() {
    std::ifstream in("/proc/self/statm");
    long total=0,resident=0; in >> total >> resident;
    return resident * static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024*1024);
}
static double peakMiB() {
    rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    return usage.ru_maxrss / 1024.0;
}
static void loadBN(const std::string& root, const std::string& prefix,
    std::vector<double>& gamma, std::vector<double>& beta,
    std::vector<double>& mean, std::vector<double>& var) {
    LoadConv2dBias(root+prefix+"_weight.npy",gamma);
    LoadConv2dBias(root+prefix+"_bias.npy",beta);
    LoadConv2dBias(root+prefix+"_running_mean.npy",mean);
    LoadConv2dBias(root+prefix+"_running_var.npy",var);
}
int main(int argc,char** argv) {
    try {
        const std::string root = argc>1 ? argv[1] : ".";
        const std::string weightsRoot=root+"/python_model/saved_models/ResNet-18/parameters/";
        const std::string image=root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin";
        auto input=load_bin_image_double(image);
        std::vector<double> gamma,beta,mean,var;
        // Derive the third layer input using the exact first two layers in InitResNet18.
        for(int layer=0;layer<2;++layer) {
            auto weights=LoadConv2dWeight(weightsRoot+(layer==0 ? "conv1_weight.npy" : "maxpool_0_weight.npy"));
            loadBN(weightsRoot,layer==0 ? "bn1" : "maxpool_1",gamma,beta,mean,var);
            input=GoldenConv2d(input,weights,std::vector<double>(weights.size(),0),2,layer==0 ? 3 : 1);
            GoldenBN(input,gamma,beta,mean,var,1e-5);
            for(auto& channel:input) for(auto& row:channel) for(auto& value:row) value=std::max(value,0.0);
        }
        auto weights=LoadConv2dWeight(weightsRoot+"layer1_0_conv1_weight.npy");
        loadBN(weightsRoot,"layer1_0_bn1",gamma,beta,mean,var);
        std::vector<double> bias(weights.size(),0);
        if(input.size()!=64 || input[0].size()!=56 || input[0][0].size()!=56)
            throw std::runtime_error("unexpected third-layer input shape");
        const uint32_t ring = argc>2 ? std::stoul(argv[2]) : 16384;
        constexpr uint32_t slots=1024,depth=20;
        const bool optimized = ring == 16384 && (argc<=3 || std::string(argv[3]) != "reference");
        if(ring!=2048 && ring!=16384) throw std::invalid_argument("ring must be 2048 or 16384");
        CCParams<CryptoContextCKKSRNS> params;
        SetParamCKKS(ring,params,slots);
        params.SetMultiplicativeDepth(depth);
        CRYPTOCONTEXT=GenCryptoContext(params);
        for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,SCHEMESWITCH}) CRYPTOCONTEXT->Enable(feature);
        KEYPAIR=CRYPTOCONTEXT->KeyGen();
        CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
        CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);
        SchSwchParams switchParams;
        SetParam_TFHE(ring,switchParams,slots);
        std::cout << "SCHEME_SWITCH num_slots_ckks=" << slots << " num_values=" << slots
                  << " EvalCKKStoFHEW_numCtxts=" << slots << " depth=" << depth << std::endl;
        auto fheKey=CRYPTOCONTEXT->EvalSchemeSwitchingSetup(switchParams);
        CCLWE=CRYPTOCONTEXT->GetBinCCForSchemeSwitch();
        CCLWE->BTKeyGen(fheKey);
        CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR,fheKey);
        CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1u<<25)/(2*CCLWE->GetBeta().ConvertToInt()),1.0);
        RESCALE_REQUIRED=false; USE_COMPACT=false;
        ENCRYPTED_HEIGHT_START=26; ENCRYPTED_HEIGHT_END=30;
        ENCRYPTED_WIDTH_START=26; ENCRYPTED_WIDTH_END=30;
        auto reference=GoldenConv2d(input,weights,bias,1,1);
        GoldenBN(reference,gamma,beta,mean,var,1e-5);
        types::vector2d<Ciphertext<DCRTPoly>> x,y;
        const int per=slots/56;
        x.resize(5);
        for(int h=26;h<=30;++h) for(int startChannel=0;startChannel<64;startChannel+=per) {
            std::vector<double> values(slots,0);
            for(int c=startChannel;c<std::min(64,startChannel+per);++c) for(int w=26;w<=30;++w) {
                values[(c-startChannel)*56+w]=input[c][h][w]; input[c][h][w]=0;
            }
            x[h-26].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.secretKey,
                CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
        }
        std::shared_ptr<Layer> conv, relu;
        std::shared_ptr<sparse_activation> sparseRelu;
        if(ring==2048) {
            conv=std::make_shared<Conv2dBN_P>(CONV_2D,"conv_bn3_legacy",weights,1,1,slots,
                gamma,beta,mean,var,1e-5,bias,CONV_2D_BN);
            relu=std::make_shared<Relu_ss>(RELU_SS_ACTIVATION,"relu3_legacy",fheKey,slots);
        } else {
            conv=std::make_shared<SparseConv2dBN_P>(CONV_2D,"conv_bn3_sparse",ring,slots,weights,1,1,
                gamma,beta,mean,var,1e-5,bias,CONV_2D_BN);
            if (optimized)
                std::dynamic_pointer_cast<SparseConv2dBN_P>(conv)->EnableFastPath(56);
            sparseRelu=std::make_shared<sparse_activation>(RELU_SS_ACTIVATION,"relu3_sparse",fheKey,ring,slots);
            relu=sparseRelu;
        }
        double3d output;
        std::cout << "Benchmark ready: 64x56x56 -> 64x56x56; encrypted input [26,30]x[26,30]; threads="
                  << omp_get_max_threads() << std::endl;
        const double rssBefore=rssMiB(),peakBefore=peakMiB();
        const auto start=std::chrono::steady_clock::now();
        conv->forward(x,input,y,output);
        const auto end=std::chrono::steady_clock::now();
        const double rssAfter=rssMiB(),peakAfter=peakMiB();
        const double latency=std::chrono::duration<double,std::milli>(end-start).count();
        auto verify = [&](const types::vector2d<Ciphertext<DCRTPoly>>& encrypted, const double3d& publicValues, bool activation) {
            double error=0;
            for(int h=0;h<56;++h) for(int block=0;block<4;++block) {
                std::vector<double> values;
                if(h>=25 && h<=31) {
                    Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,encrypted[h-25][block],&decoded);
                    decoded->SetLength(slots); values=decoded->GetRealPackedValue();
                }
                for(int c=block*per;c<std::min(64,(block+1)*per);++c) for(int w=0;w<56;++w) {
                    double actual=(h>=25 && h<=31 && w>=25 && w<=31) ? values[(c%per)*56+w] : publicValues[c][h][w];
                    double expected=activation ? std::max(0.0,reference[c][h][w]) : reference[c][h][w];
                    if(!std::isfinite(actual)) throw std::runtime_error("non-finite benchmark output");
                    error=std::max(error,std::abs(actual-expected));
                }
            }
            return error;
        };
        const double convError=verify(y,output,false);
        if(convError>1e-3) throw std::runtime_error("convolution correctness check failed");
        std::cout << "CONV_COMPLETE latency_ms=" << latency << " peak_mib=" << peakAfter
                  << " max_error=" << convError << std::endl;
        types::vector2d<Ciphertext<DCRTPoly>> activatedCts; double3d activatedPts;
        const double reluRssBefore=rssMiB(),reluPeakBefore=peakMiB();
        const auto reluStart=std::chrono::steady_clock::now();
        relu->forward(y,output,activatedCts,activatedPts);
        const auto reluEnd=std::chrono::steady_clock::now();
        const double reluRssAfter=rssMiB(),reluPeakAfter=peakMiB();
        const double reluMs=std::chrono::duration<double,std::milli>(reluEnd-reluStart).count();
        const double reluError=verify(activatedCts,activatedPts,true);
        // Scheme-switch sign is quantized; report every error and fail if it exceeds 1e-3.
        if(reluError>1e-3) throw std::runtime_error("ReLU correctness check failed");
        std::cout << std::fixed << std::setprecision(3)
                  << "RESULT ring_dim=" << ring << " slots=" << slots << " depth=" << depth << " threads=" << omp_get_max_threads()
                  << " variant=" << (optimized ? "optimized" : "reference")
                  << " conv_ms=" << latency << " rss_before_mib=" << rssBefore
                  << " rss_after_mib=" << rssAfter << " process_peak_before_mib=" << peakBefore
                  << " process_peak_after_mib=" << peakAfter
                  << " relu_ms=" << reluMs << " relu_rss_before_mib=" << reluRssBefore
                  << " relu_rss_after_mib=" << reluRssAfter << " relu_peak_before_mib=" << reluPeakBefore
                  << " relu_peak_after_mib=" << reluPeakAfter
                  << " input_ciphertexts=" << x.size()*x[0].size() << " output_ciphertexts=" << y.size()*y[0].size()
                  << " output_encrypted_h=" << ENCRYPTED_HEIGHT_START << ':' << ENCRYPTED_HEIGHT_END
                  << " output_encrypted_w=" << ENCRYPTED_WIDTH_START << ':' << ENCRYPTED_WIDTH_END << '\n'
                  << std::scientific << "conv_max_error=" << convError << " relu_max_error=" << reluError << std::endl;
        if(sparseRelu) {
            const auto& stats=sparseRelu->last_profile();
            std::cout << std::fixed << "RELU_PROFILE conversion_ms=" << stats.conversion_ms
                      << " sign_ms=" << stats.sign_ms << " mask_ms=" << stats.mask_ms
                      << " converted_values=" << stats.converted_values
                      << " evaluated_signs=" << stats.evaluated_signs << std::endl;
        }
    } catch(const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
