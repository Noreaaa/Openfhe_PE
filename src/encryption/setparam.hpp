#pragma once

#include <cstdint>
#include "openfhe.h"

// num_slots == 0 selects RingDim / 2 (full packing).
void SetParamCKKS(uint32_t RingDim,
                  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>& parameters,
                  uint32_t num_slots = 0);
void SetParam_TFHE(uint32_t RingDim, lbcrypto::SchSwchParams& params, uint32_t num_slots = 0);
