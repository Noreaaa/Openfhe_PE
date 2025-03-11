#pragma once 
#define PROFILE

#include "openfhe.h"


using namespace lbcrypto;

void TestOperation(uint32_t RingDim);

int main(int argc, char* argv[]){

    uint32_t RingDim = 1 << 11;
    for (int i = 0; i < 10; i++){
        RingDim = RingDim << 1;
        std::cout << "Ring dimension: " << RingDim << std::endl;
        std::vector<double> x_vec;
        for (int j = 0; j < RingDim/2; j++){
            x_vec.push_back(rand() % 100);
        }
        TestOperation(RingDim, x_vec);
        x_vec.clear();
    }
    
}

void TestOperation(uint32_t RingDim, std::vector<double>& x_vec){

    CCParams<CryptoContextCKKSRNS> parameters;

    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(RingDim);
    parameters.SetNumLargeDigits(3);
    parameters.SetKeySwitchTechnique(HYBRID);


    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 59;
    usint firstMod               = 60;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    

}