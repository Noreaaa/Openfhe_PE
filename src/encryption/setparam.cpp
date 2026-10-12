#include "setparam.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lbcrypto;

void SetParamCKKS(uint32_t RingDim, CCParams<CryptoContextCKKSRNS>& parameters,
                  uint32_t num_slots){
    if (RingDim < 2 || (RingDim & (RingDim - 1)) != 0) {
        throw std::invalid_argument("SetParamCKKS: RingDim must be a power of two >= 2");
    }
    if (num_slots == 0) {
        num_slots = RingDim / 2;
    }
    if (num_slots > RingDim / 2 || (num_slots & (num_slots - 1)) != 0) {
        throw std::invalid_argument("SetParamCKKS: num_slots must be a power of two <= RingDim / 2");
    }
    // Step 1: Set CryptoContext

    // A. Specify main parameters
    /*  A1) Secret key distribution
    * The secret key distribution for CKKS should either be SPARSE_TERNARY or UNIFORM_TERNARY.
    * The SPARSE_TERNARY distribution was used in the original CKKS paper,
    * but in this example, we use UNIFORM_TERNARY because this is included in the homomorphic
    * encryption standard.
    */
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    parameters.SetBatchSize(num_slots);

    /*  A2) Desired security level based on FHE standards.
    * In this example, we use the "NotSet" option, so the example can run more quickly with
    * a smaller ring dimension. Note that this should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices. In production-like environments, we recommend using
    * HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic for 128-bit, 192-bit,
    * or 256-bit security, respectively. If you choose one of these as your security level,
    * you do not need to set the ring dimension.
    */
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(RingDim);

    /*  A3) Key switching parameters.
    * By default, we use HYBRID key switching with a digit size of 3.
    * Choosing a larger digit size can reduce complexity, but the size of keys will increase.
    * Note that you can leave these lines of code out completely, since these are the default values.
    */
    parameters.SetNumLargeDigits(3);
    //parameters.SetKeySwitchTechnique(HYBRID);

    /*  A4) Scaling parameters.
    * By default, we set the modulus sizes and rescaling technique to the following values
    * to obtain a good precision and performance tradeoff. We recommend keeping the parameters
    * below unless you are an FHE expert.
    */
#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    // Currently, only FIXEDMANUAL and FIXEDAUTO modes are supported for 128-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint scaleModSize           = 78;
    usint firstMod               = 89;
#else
    // All modes are supported for 64-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint scaleModSize               = 30;
    usint firstMod               = 40;
#endif

    parameters.SetScalingModSize(scaleModSize);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    /*  A4) Bootstrapping parameters.
    * We set a budget for the number of levels we can consume in bootstrapping for encoding and decoding, respectively.
    * Using larger numbers of levels reduces the complexity and number of rotation keys,
    * but increases the depth required for bootstrapping.
	* We must choose values smaller than ceil(log2(slots)). A level budget of {4, 4} is good for higher ring
    * dimensions (65536 and higher).
    */
    std::vector<uint32_t> levelBudget = {3, 3};

    /* We give the user the option of configuring values for an optimization algorithm in bootstrapping.
    * Here, we specify the giant step for the baby-step-giant-step algorithm in linear transforms
    * for encoding and decoding, respectively. Either choose this to be a power of 2
    * or an exact divisor of the number of slots. Setting it to have the default value of {0, 0} allows OpenFHE to choose
    * the values automatically.
    */
    std::vector<uint32_t> bsgsDim = {0, 0};

     /*  A5) Multiplicative depth.
    * The goal of bootstrapping is to increase the number of available levels we have, or in other words,
    * to dynamically increase the multiplicative depth. However, the bootstrapping procedure itself
    * needs to consume a few levels to run. We compute the number of bootstrapping levels required
    * using GetBootstrapDepth, and add it to levelsAvailableAfterBootstrap to set our initial multiplicative
    * depth.
    */
    uint32_t levelsAvailableAfterBootstrap = 3;
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    std::cout << "depth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);
}

void SetParam_TFHE(uint32_t RingDim, SchSwchParams& params, uint32_t num_slots){
    if (num_slots == 0) num_slots = RingDim / 2;

    uint32_t logQ_ccLWE = 25; 
    params.SetSecurityLevelCKKS(HEStd_NotSet);
    params.SetSecurityLevelFHEW(TOY);
    params.SetNumSlotsCKKS(num_slots);
    params.SetNumValues(num_slots);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    
}

