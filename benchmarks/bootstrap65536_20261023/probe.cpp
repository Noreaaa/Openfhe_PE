#include "openfhe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sys/resource.h>
#include <omp.h>
using namespace lbcrypto;
int main(int argc,char** argv){try{
 const uint32_t scale=argc>1?std::stoul(argv[1]):30;
 const uint32_t depth=argc>2?std::stoul(argv[2]):30;
 const uint32_t first=argc>3?std::stoul(argv[3]):scale+10;
 const uint32_t correction=argc>4?std::stoul(argv[4]):10;
 omp_set_num_threads(96);omp_set_dynamic(0);omp_set_max_active_levels(1);
 CCParams<CryptoContextCKKSRNS> p;
 p.SetSecurityLevel(HEStd_128_classic);p.SetSecretKeyDist(UNIFORM_TERNARY);
 p.SetRingDim(65536);p.SetBatchSize(1024);p.SetScalingModSize(scale);p.SetFirstModSize(first);
 p.SetScalingTechnique(FLEXIBLEAUTO);p.SetKeySwitchTechnique(HYBRID);p.SetNumLargeDigits(3);p.SetMultiplicativeDepth(depth);
 auto cc=GenCryptoContext(p);for(auto f:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,FHE})cc->Enable(f);
 std::cout<<"CONTEXT ring="<<cc->GetRingDimension()<<" security=HEStd_128_classic depth="<<depth<<" scale="<<scale<<" first="<<first<<" correction="<<correction<<" reserve="<<FHECKKSRNS::GetBootstrapDepth({3,3},UNIFORM_TERNARY)<<std::endl;
 cc->EvalBootstrapSetup({3,3},{0,0},1024,correction);auto keys=cc->KeyGen();cc->EvalMultKeyGen(keys.secretKey);cc->EvalBootstrapKeyGen(keys.secretKey,1024);
 std::vector<double> values(1024);for(size_t i=0;i<values.size();++i)values[i]=(int(i%9)-4)*0.05;
 auto ct=cc->Encrypt(keys.publicKey,cc->MakeCKKSPackedPlaintext(values,1,depth-1,nullptr,1024));
 std::cout<<"READY input_level="<<ct->GetLevel()<<" input_towers="<<ct->GetElements()[0].GetNumOfElements()<<std::endl;
 auto t=std::chrono::steady_clock::now();auto output=cc->EvalBootstrap(ct);auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
 Plaintext decoded;cc->Decrypt(keys.secretKey,output,&decoded);decoded->SetLength(1024);auto actual=decoded->GetRealPackedValue();double error=0;
 for(size_t i=0;i<values.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error("nonfinite output");error=std::max(error,std::abs(actual[i]-values[i]));}
 rusage u{};getrusage(RUSAGE_SELF,&u);
 std::cout<<"SUCCESS ring="<<output->GetElements()[0].GetRingDimension()<<" slots="<<output->GetSlots()<<" output_level="<<output->GetLevel()<<" towers="<<output->GetElements()[0].GetNumOfElements()<<" noise_scale_degree="<<output->GetNoiseScaleDeg()<<" bootstrap_ms="<<ms<<" max_abs_error="<<error<<" peak_mib="<<u.ru_maxrss/1024.0<<std::endl;
 return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
