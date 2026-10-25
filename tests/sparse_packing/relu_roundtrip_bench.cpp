#include "../../src/cnn/sparse_activation.hpp"
#include "baseline/baseline_sparse_activation.hpp"
#include "../../src/encryption/setparam.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>
#include <omp.h>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a) { return std::chrono::duration<double,std::milli>(Clock::now()-a).count(); }
static double peak() { rusage u{}; getrusage(RUSAGE_SELF,&u); return u.ru_maxrss/1024.0; }
static double rss() { std::ifstream f("/proc/self/statm"); long total=0,r=0; f>>total>>r; return r*double(sysconf(_SC_PAGESIZE))/(1024*1024); }
int main(int argc,char** argv) {
 try {
    const bool baseline=argc>1 && std::string(argv[1])=="baseline";
    const uint32_t depth=argc>2 ? std::stoul(argv[2]) : 20;
    constexpr uint32_t ring=16384,slots=1024;
    const uint32_t scaleBits=argc>3 ? std::stoul(argv[3]) : 50;
    std::cout<<std::setprecision(12)<<"CONFIG variant="<<(baseline?"baseline":"encrypted")
             <<" ring="<<ring<<" slots="<<slots<<" depth="<<depth<<" scale_bits="<<scaleBits<<" first_bits="<<scaleBits+10<<" threads="<<omp_get_max_threads()<<std::endl;
    CCParams<CryptoContextCKKSRNS> p; SetParamCKKS(ring,p,slots); p.SetMultiplicativeDepth(depth); p.SetScalingModSize(scaleBits); p.SetFirstModSize(scaleBits+10);
    auto start=Clock::now();
    CRYPTOCONTEXT=GenCryptoContext(p);
    for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,SCHEMESWITCH}) CRYPTOCONTEXT->Enable(f);
    std::cout<<"SETUP context_ms="<<ms(start)<<std::endl;
    start=Clock::now(); KEYPAIR=CRYPTOCONTEXT->KeyGen(); CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
    CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);
    std::cout<<"SETUP ckks_keys_ms="<<ms(start)<<std::endl;
    start=Clock::now(); SchSwchParams sw; SetParam_TFHE(ring,sw,slots);
    auto fhe=CRYPTOCONTEXT->EvalSchemeSwitchingSetup(sw);
    CCLWE=CRYPTOCONTEXT->GetBinCCForSchemeSwitch(); CCLWE->BTKeyGen(fhe);
    CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR,fhe);
    std::cout<<"SETUP switching_keys_ms="<<ms(start)<<std::endl;
    start=Clock::now(); CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1u<<25)/(2*CCLWE->GetBeta().ConvertToInt()),1.0);
    std::cout<<"SETUP precompute_ms="<<ms(start)<<" rss_mib="<<rss()<<" peak_mib="<<peak()<<std::endl;
    RESCALE_REQUIRED=false;
    ENCRYPTED_HEIGHT_START=ENCRYPTED_HEIGHT_END=1;
    ENCRYPTED_WIDTH_START=510; ENCRYPTED_WIDTH_END=511;
    double3d pts(3,types::double2d(2,std::vector<double>(512,-0.25)));
    const std::vector<double> samples={-0.8,0.6,0,1.2,-0.4,0.2};
    types::vector2d<Ciphertext<DCRTPoly>> input(1),out;
    for(int b=0;b<2;++b) {
        std::vector<double> values(slots,0);
        for(int c=2*b;c<std::min(3,2*b+2);++c) for(int w=510;w<=511;++w) {
            values[(c%2)*512+w]=samples[c*2+w-510]; pts[c][1][w]=0;
        }
        input[0].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
            CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
    }
    sparse_activation encrypted(RELU_SS_ACTIVATION,"encrypted",fhe,ring,slots);
    baseline_sparse_activation old(RELU_SS_ACTIVATION,"baseline",fhe,ring,slots);
    double maxError=0;
    for(int trial=0;trial<3;++trial) {
        double3d result;
        start=Clock::now();
        if(baseline) old.forward(input,pts,out,result); else encrypted.forward(input,pts,out,result);
        const double elapsed=ms(start);
        const double measuredPeak=peak(),measuredRss=rss();
        std::cout<<"TRIAL index="<<trial<<" total_ms="<<elapsed;
        if(baseline) {
            const auto& d=old.last_profile();
            std::cout<<" amplify_and_forward_ms="<<d.conversion_ms<<" sign_and_decrypt_ms="<<d.sign_ms
                     <<" encode_and_plain_mult_ms="<<d.mask_ms;
        } else {
            const auto& d=encrypted.last_profile();
            std::cout<<" amplify_ms="<<d.amplify_ms<<" forward_ms="<<d.conversion_ms<<" sign_ms="<<d.sign_ms
                     <<" reverse_ms="<<d.reverse_ms<<" complement_ms="<<d.complement_ms<<" ciphertext_mult_ms="<<d.mask_ms;
        }
        std::cout<<" rss_mib="<<measuredRss<<" peak_mib="<<measuredPeak
                 <<" output_level="<<out[0][0]->GetLevel()<<" towers="<<out[0][0]->GetElements()[0].GetNumOfElements()<<std::endl;
        for(int b=0;b<2;++b) {
            Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,out[0][b],&decoded); decoded->SetLength(slots);
            auto values=decoded->GetRealPackedValue();
            for(uint32_t s=0;s<slots;++s) {
                int c=b*2+s/512,w=s%512;
                double expected=(c<3&&w>=510)?std::max(0.0,samples[c*2+w-510]):0;
                if(!std::isfinite(values[s])) throw std::runtime_error("non-finite output");
                maxError=std::max(maxError,std::abs(values[s]-expected));
                if(c<3 && w>=510) std::cout<<"CHECK trial="<<trial<<" c="<<c<<" w="<<w<<" actual="<<values[s]<<" expected="<<expected<<std::endl;
            }
        }
        for(const auto& c:result) for(const auto& r:c) for(double v:r)
            if(v!=0) throw std::runtime_error("public ReLU mismatch");
        if(maxError>1e-3) throw std::runtime_error("ReLU error exceeds 1e-3");
    }
    // Primitive latencies are measured separately from the pipeline. Each call
    // uses the same fresh level-0 input; report the median of five invocations.
    std::vector<double> ones(slots,1.0);
    auto plain=CRYPTOCONTEXT->MakeCKKSPackedPlaintext(ones,1,0,nullptr,slots);
    auto ct=input[0][0];
    auto product=CRYPTOCONTEXT->EvalMult(ct,ct);
    auto measure=[&](const char* name, auto operation) {
        std::vector<double> timings;
        for(int i=0;i<5;++i) {
            auto t=Clock::now(); auto value=operation(); timings.push_back(ms(t));
            if(!value) throw std::runtime_error("empty primitive result");
        }
        std::sort(timings.begin(),timings.end());
        std::cout<<"PRIMITIVE name="<<name<<" median_ms="<<timings[2]<<" min_ms="<<timings.front()<<" max_ms="<<timings.back()<<std::endl;
    };
    measure("EvalAdd_ct_ct",[&]{return CRYPTOCONTEXT->EvalAdd(ct,ct);});
    measure("EvalRotate_1",[&]{return CRYPTOCONTEXT->EvalRotate(ct,1);});
    measure("EvalMult_scalar",[&]{return CRYPTOCONTEXT->EvalMult(ct,1000.0);});
    measure("EvalMult_ct_pt",[&]{return CRYPTOCONTEXT->EvalMult(ct,plain);});
    measure("EvalMult_ct_ct",[&]{return CRYPTOCONTEXT->EvalMult(ct,ct);});
    measure("EvalSub_scalar_ct",[&]{return CRYPTOCONTEXT->EvalSub(1.0,ct);});
    measure("Rescale",[&]{return CRYPTOCONTEXT->Rescale(product);});
    std::cout<<"PASS max_abs_error="<<maxError<<" final_peak_mib="<<peak()<<std::endl;
    return 0;
 } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
