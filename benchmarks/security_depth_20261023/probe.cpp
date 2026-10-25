#include "openfhe.h"
#include <iostream>
#include <cmath>
using namespace lbcrypto;
int main(){
 for(uint32_t ring:{16384u,32768u,65536u}) for(uint32_t scale:{30u,40u,50u}) {
  for(uint32_t depth=1;depth<=50;++depth){
   CCParams<CryptoContextCKKSRNS> p;
   p.SetSecurityLevel(HEStd_128_classic);p.SetSecretKeyDist(UNIFORM_TERNARY);
   p.SetRingDim(ring);p.SetBatchSize(1024);p.SetScalingTechnique(FLEXIBLEAUTO);
   p.SetScalingModSize(scale);p.SetFirstModSize(scale+10);p.SetNumLargeDigits(3);
   p.SetKeySwitchTechnique(HYBRID);p.SetMultiplicativeDepth(depth);
   try{
    auto cc=GenCryptoContext(p);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    double q=0,pbits=0;
    for(const auto& m:cp->GetElementParams()->GetParams())q+=std::log2(m->GetModulus().ConvertToDouble());
    if(cp->GetParamsP())for(const auto& m:cp->GetParamsP()->GetParams())pbits+=std::log2(m->GetModulus().ConvertToDouble());
    std::cout<<"PASS ring="<<ring<<" scale="<<scale<<" first="<<scale+10<<" depth="<<depth<<" logQ="<<q<<" logP="<<pbits<<" logQP="<<q+pbits<<std::endl;
    CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
   }catch(const std::exception& e){std::cout<<"FAIL ring="<<ring<<" scale="<<scale<<" first="<<scale+10<<" depth="<<depth<<" reason="<<e.what()<<std::endl;if(std::string(e.what()).find("does not comply")!=std::string::npos)break;}
  }
 }
}
