#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/cnn/sparse_activation.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/dataloader.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
constexpr int ring = 16384, slots = 1024, depth = 20;
constexpr double epsilon = 1e-5, tolerance = 1e-3;
struct BN { std::vector<double> gamma, beta, mean, var; };
BN loadBN(const std::string& root, const std::string& prefix) {
    BN bn;
    LoadConv2dBias(root+prefix+"_weight.npy",bn.gamma);
    LoadConv2dBias(root+prefix+"_bias.npy",bn.beta);
    LoadConv2dBias(root+prefix+"_running_mean.npy",bn.mean);
    LoadConv2dBias(root+prefix+"_running_var.npy",bn.var);
    return bn;
}
// Independent direct cross-correlation + inference BN: no GoldenConv2d, packing,
// rotations, folded BN weights, or optimized convolution helpers are used.
double3d reference(const double3d& input, const types::double4d& weights,
                   const BN& bn, int stride, int pad) {
    const int ci=input.size(), hi=input[0].size(), wi=input[0][0].size();
    const int co=weights.size(), kh=weights[0][0].size(), kw=weights[0][0][0].size();
    const int ho=(hi+2*pad-kh)/stride+1, wo=(wi+2*pad-kw)/stride+1;
    if (bn.gamma.size()!=weights.size() || bn.beta.size()!=weights.size() ||
        bn.mean.size()!=weights.size() || bn.var.size()!=weights.size())
        throw std::runtime_error("reference BN shape mismatch");
    double3d output(co,types::double2d(ho,std::vector<double>(wo)));
    #pragma omp parallel for
    for(int o=0;o<co;++o) for(int h=0;h<ho;++h) for(int w=0;w<wo;++w) {
        long double sum=0;
        for(int c=0;c<ci;++c) for(int r=0;r<kh;++r) {
            int ih=h*stride+r-pad;
            if(ih<0 || ih>=hi) continue;
            for(int s=0;s<kw;++s) {
                int iw=w*stride+s-pad;
                if(iw>=0 && iw<wi)
                    sum+=static_cast<long double>(input[c][ih][iw])*weights[o][c][r][s];
            }
        }
        // These exported convolutions have zero bias, as in InitResNet18.
        long double value=(sum-bn.mean[o])*bn.gamma[o]/std::sqrt(
            static_cast<long double>(bn.var[o])+epsilon)+bn.beta[o];
        output[o][h][w]=static_cast<double>(value);
    }
    return output;
}
double3d relu(double3d values) {
    for(auto& channel:values) for(auto& row:channel) for(auto& x:row) x=std::max(0.0,x);
    return values;
}
void save(const std::filesystem::path& path, const double3d& tensor) {
    std::vector<double> flat;
    for(const auto& channel:tensor) for(const auto& row:channel) flat.insert(flat.end(),row.begin(),row.end());
    cnpy::npy_save(path.string(),flat.data(),
        {tensor.size(),tensor[0].size(),tensor[0][0].size()},"w");
}
double3d reconstruct(const types::vector2d<Ciphertext<DCRTPoly>>& cts, double3d publicValues) {
    const int channels=publicValues.size(), width=publicValues[0][0].size(), per=slots/width;
    const size_t blocks=(channels+per-1)/per;
    if(cts.size()!=static_cast<size_t>(ENCRYPTED_HEIGHT_END-ENCRYPTED_HEIGHT_START+1))
        throw std::runtime_error("output ciphertext height mismatch");
    for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h) {
        const auto& row=cts.at(h-ENCRYPTED_HEIGHT_START);
        if(row.size()!=blocks) throw std::runtime_error("output ciphertext blocks mismatch");
        for(size_t block=0;block<blocks;++block) {
            if(row[block]->GetSlots()!=slots) throw std::runtime_error("output slot count is not 1024");
            Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,row[block],&decoded);
            decoded->SetLength(slots); const auto values=decoded->GetRealPackedValue();
            for(int c=block*per;c<std::min(channels,static_cast<int>((block+1)*per));++c)
                for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w)
                    publicValues[c][h][w]=values.at((c%per)*width+w);
        }
    }
    return publicValues;
}
struct Stats {
    size_t count=0, above=0;
    long double absSum=0, squareSum=0, referenceSquareSum=0;
    double maxAbs=-1, expected=0, actual=0;
    int c=0,h=0,w=0;
    void add(double x,double y,int channel,int row,int col) {
        if(!std::isfinite(x)||!std::isfinite(y)) throw std::runtime_error("non-finite reference or output");
        long double error=static_cast<long double>(x)-y;
        double absolute=std::abs(error);
        ++count; above+=absolute>tolerance;
        absSum+=absolute; squareSum+=error*error; referenceSquareSum+=static_cast<long double>(y)*y;
        if(absolute>maxAbs) { maxAbs=absolute;expected=y;actual=x;c=channel;h=row;w=col; }
    }
};
bool report(const std::string& stage,const double3d& actual,const double3d& expected,std::ofstream& csv) {
    Stats all,encrypted,publicRegion;
    for(size_t c=0;c<expected.size();++c) for(size_t h=0;h<expected[c].size();++h)
        for(size_t w=0;w<expected[c][h].size();++w) {
            const double x=actual.at(c).at(h).at(w), y=expected[c][h][w];
            all.add(x,y,c,h,w);
            bool privateValue=static_cast<int>(h)>=ENCRYPTED_HEIGHT_START && static_cast<int>(h)<=ENCRYPTED_HEIGHT_END &&
                              static_cast<int>(w)>=ENCRYPTED_WIDTH_START && static_cast<int>(w)<=ENCRYPTED_WIDTH_END;
            (privateValue ? encrypted : publicRegion).add(x,y,c,h,w);
        }
    for(auto entry:{std::pair<const char*,const Stats*>{"all",&all},{"encrypted",&encrypted},{"public",&publicRegion}}) {
        const Stats& s=*entry.second;
        const double mae=s.absSum/s.count, rmse=std::sqrt(s.squareSum/s.count);
        const double relativeL2=s.referenceSquareSum>0 ? std::sqrt(s.squareSum/s.referenceSquareSum) :
            (s.squareSum==0 ? 0 : std::numeric_limits<double>::infinity());
        csv<<stage<<','<<entry.first<<','<<s.count<<','<<s.maxAbs<<','<<mae<<','<<rmse<<','<<relativeL2<<','
           <<s.above<<','<<s.c<<','<<s.h<<','<<s.w<<','<<s.expected<<','<<s.actual<<'\n';
        std::cout<<stage<<" region="<<entry.first<<" count="<<s.count<<" max_abs="<<s.maxAbs
                 <<" mae="<<mae<<" rmse="<<rmse<<" relative_l2="<<relativeL2
                 <<" above_1e-3="<<s.above<<" worst=("<<s.c<<','<<s.h<<','<<s.w<<')'<<std::endl;
    }
    return all.above==0;
}
} // namespace

int main(int argc,char** argv) {
    try {
        const std::filesystem::path root=argc>1 ? argv[1] : ".";
        const std::filesystem::path outputDir=argc>2 ? argv[2] : "test_results/sparse_real_accuracy";
        std::filesystem::create_directories(outputDir);
        const std::string weightsRoot=(root/"python_model/saved_models/ResNet-18/parameters").string()+"/";
        auto input=load_bin_image_double((root/"datasets/imagenet-mini/ILSVRC2012_val_00000029.bin").string());
        for(int layer=0;layer<2;++layer) {
            auto weights=LoadConv2dWeight(weightsRoot+(layer==0 ? "conv1_weight.npy" : "maxpool_0_weight.npy"));
            auto bn=loadBN(weightsRoot,layer==0 ? "bn1" : "maxpool_1");
            input=relu(reference(input,weights,bn,2,layer==0 ? 3 : 1));
        }
        auto weights=LoadConv2dWeight(weightsRoot+"layer1_0_conv1_weight.npy");
        auto bn=loadBN(weightsRoot,"layer1_0_bn1");
        if(input.size()!=64 || input[0].size()!=56 || input[0][0].size()!=56)
            throw std::runtime_error("expected third-layer input 64x56x56");
        auto expectedConv=reference(input,weights,bn,1,1);
        auto expectedRelu=relu(expectedConv);
        save(outputDir/"input.npy",input);
        save(outputDir/"expected_conv_bn.npy",expectedConv);
        save(outputDir/"expected_relu.npy",expectedRelu);
        CCParams<CryptoContextCKKSRNS> p;
        p.SetSecurityLevel(HEStd_NotSet); p.SetRingDim(ring); p.SetBatchSize(slots);
        p.SetSecretKeyDist(UNIFORM_TERNARY); p.SetNumLargeDigits(3);
        p.SetScalingTechnique(FLEXIBLEAUTO); p.SetScalingModSize(50); p.SetFirstModSize(60);
        p.SetMultiplicativeDepth(depth);
        CRYPTOCONTEXT=GenCryptoContext(p);
        for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,SCHEMESWITCH}) CRYPTOCONTEXT->Enable(f);
        KEYPAIR=CRYPTOCONTEXT->KeyGen();
        CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
        SchSwchParams sw; SetParam_TFHE(ring,sw,slots);
        CRYPTOCONTEXT->SetParamsFromCKKSCryptocontext(sw);
        if(sw.GetNumSlotsCKKS()!=slots || sw.GetNumValues()!=slots) throw std::runtime_error("switch slot count mismatch");
        auto fheKey=CRYPTOCONTEXT->EvalSchemeSwitchingSetup(sw);
        CCLWE=CRYPTOCONTEXT->GetBinCCForSchemeSwitch(); CCLWE->BTKeyGen(fheKey);
        CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR,fheKey);
        CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1u<<25)/(2*CCLWE->GetBeta().ConvertToInt()),1.0);
        RESCALE_REQUIRED=false; USE_COMPACT=false;
        ENCRYPTED_HEIGHT_START=26; ENCRYPTED_HEIGHT_END=30;
        ENCRYPTED_WIDTH_START=26; ENCRYPTED_WIDTH_END=30;
        constexpr int width=56,channels=64,per=slots/width;
        types::vector2d<Ciphertext<DCRTPoly>> encryptedInput(5),convCts,reluCts;
        for(int h=26;h<=30;++h) for(int block=0;block<(channels+per-1)/per;++block) {
            std::vector<double> values(slots,0);
            for(int c=block*per;c<std::min(channels,(block+1)*per);++c) for(int w=26;w<=30;++w) {
                values[(c%per)*width+w]=input[c][h][w]; input[c][h][w]=0;
            }
            encryptedInput[h-26].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
                CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
        }
        SparseConv2dBN_P conv(CONV_2D,"real_accuracy_conv3",ring,slots,weights,1,1,
            bn.gamma,bn.beta,bn.mean,bn.var,epsilon,std::vector<double>(64,0),CONV_2D_BN);
        conv.EnableFastPath(width);
        sparse_activation activation(RELU_SS_ACTIVATION,"real_accuracy_relu3",fheKey,ring,slots);
        std::cout<<"CONFIG ring=16384 slots=1024 depth=20 input_region=[26:30,26:30] tolerance="<<tolerance<<std::endl;
        double3d convPublic,reluPublic;
        conv.forward(encryptedInput,input,convCts,convPublic);
        if(ENCRYPTED_HEIGHT_START!=25 || ENCRYPTED_HEIGHT_END!=31 || ENCRYPTED_WIDTH_START!=25 || ENCRYPTED_WIDTH_END!=31)
            throw std::runtime_error("output encrypted region mismatch");
        auto actualConv=reconstruct(convCts,convPublic);
        save(outputDir/"actual_conv_bn.npy",actualConv);
        std::ofstream csv(outputDir/"errors.csv");
        if(!csv) throw std::runtime_error("cannot open error report");
        csv<<std::setprecision(17)<<"stage,region,count,max_abs,mae,rmse,relative_l2,above_1e-3,worst_c,worst_h,worst_w,expected,actual\n";
        std::cout<<std::scientific<<std::setprecision(8);
        bool passed=report("conv_bn_vs_reference",actualConv,expectedConv,csv);
        activation.forward(convCts,convPublic,reluCts,reluPublic);
        auto actualRelu=reconstruct(reluCts,reluPublic);
        save(outputDir/"actual_relu.npy",actualRelu);
        passed=report("relu_vs_reference",actualRelu,expectedRelu,csv) && passed;
        // Isolate activation error from the error already introduced by CKKS convolution.
        passed=report("relu_vs_decrypted_conv",actualRelu,relu(actualConv),csv) && passed;
        const auto& profile=activation.last_profile();
        if(profile.converted_values!=28u*slots || profile.evaluated_signs!=3136)
            throw std::runtime_error("unexpected switch/sign counts");
        std::cout<<"SWITCH explicit_slots="<<slots<<" converted_values="<<profile.converted_values
                 <<" signs="<<profile.evaluated_signs<<std::endl;
        std::cout<<(passed ? "PASS" : "FAIL")<<" reports="<<outputDir<<std::endl;
        return passed ? 0 : 1;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<std::endl; return 1; }
}
