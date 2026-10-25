#include "../../src/cnn/encrypted_relu.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/globals.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <omp.h>
int main() {
 try {
    using namespace lbcrypto;
    constexpr uint32_t ring=16384,slots=1024;
    CCParams<CryptoContextCKKSRNS> p; SetParamCKKS(ring,p,slots); p.SetMultiplicativeDepth(20);
    CRYPTOCONTEXT=GenCryptoContext(p);
    for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,SCHEMESWITCH}) CRYPTOCONTEXT->Enable(f);
    KEYPAIR=CRYPTOCONTEXT->KeyGen(); CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
    SchSwchParams sw; SetParam_TFHE(ring,sw,slots);
    auto key=CRYPTOCONTEXT->EvalSchemeSwitchingSetup(sw);
    CCLWE=CRYPTOCONTEXT->GetBinCCForSchemeSwitch();
    CRYPTOCONTEXT->EvalSchemeSwitchingKeyGen(KEYPAIR,key);
    CRYPTOCONTEXT->EvalCompareSwitchPrecompute((1u<<25)/(2*CCLWE->GetBeta().ConvertToInt()),1.0);
    std::vector<double> values(slots);
    for(size_t i=0;i<slots;++i) values[i]=(i%2 ? 0.6 : -0.8);
    auto ct=CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots));
    auto lwe=CRYPTOCONTEXT->EvalCKKStoFHEW(CRYPTOCONTEXT->EvalMult(ct,1000.0),slots);
    std::cout<<std::setprecision(12)<<"CONFIG ring=16384 slots=1024 depth=20 scale_bits=50 first_bits=60 threads="<<omp_get_max_threads()<<" reverse_numCtxts=1024 warmup=1 repeats=5 ciphertexts=1"<<std::endl;
    double maxError=0;
    for(int round=0;round<6;++round) {
        std::vector<size_t> counts={6,64,256,1024};
        if(round%2) std::reverse(counts.begin(),counts.end());
        for(auto count:counts) {
            std::vector<size_t> active(count);
            for(size_t i=0;i<count;++i) active[i]=i;
            EncryptedReluProfile detail;
            auto mask=EncryptedReluMask(CRYPTOCONTEXT,CCLWE,lwe,active,slots,&detail);
            Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,mask,&decoded); decoded->SetLength(slots);
            const auto result=decoded->GetRealPackedValue();
            double error=0;
            // Inactive slots contain the filler sign, matching the production
            // helper's semantics. This benchmark checks the mask, not a product.
            for(size_t i=0;i<slots;++i) {
                double expected=i<count ? (values[i]>=0?1.0:0.0) : 0.0;
                if(!std::isfinite(result[i])) throw std::runtime_error("non-finite mask");
                error=std::max(error,std::abs(result[i]-expected));
            }
            maxError=std::max(maxError,error);
            if(error>1e-3) throw std::runtime_error("mask accuracy error exceeds 1e-3");
            rusage u{};getrusage(RUSAGE_SELF,&u);
            std::cout<<(round==0?"WARMUP":"TRIAL")<<" round="<<round<<" active="<<count
                     <<" reverse_inputs=1024 sign_ms="<<detail.sign_ms<<" reverse_ms="<<detail.reverse_ms
                     <<" complement_ms="<<detail.complement_ms<<" error="<<error
                     <<" peak_mib="<<u.ru_maxrss/1024.0<<std::endl;
        }
    }
    std::cout<<"PASS max_abs_error="<<maxError<<std::endl;
    return 0;
 } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
