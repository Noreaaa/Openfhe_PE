#pragma once 

#include "openfhe.h"
#include "../utils/types.hpp"
#include "../utils/globals.hpp"
#include "../utils/helper.hpp"
using namespace lbcrypto;

void Partial_Encrypt_Sparse(types::double3d &image_3d, uint32_t numSlots, 
uint32_t depth,CryptoContext<DCRTPoly> cryptocontext, 
KeyPair<lbcrypto::DCRTPoly> Keypair, int channel, int max_channel, int height_start, int height_end,
int width_start, int width_end, std::vector<Ciphertext<DCRTPoly>> &x_ctxt);

void Partial_Encrypt(types::double3d &image_3d, uint32_t numSlots, 
uint32_t depth, CryptoContext<DCRTPoly> cryptocontext, 
KeyPair<lbcrypto::DCRTPoly> Keypair, int channel, 
int enc_height, int enc_width,
int height_start, int height_end, 
int width_start, int width_end,
Ciphertext<DCRTPoly> &x_ctxt);

void Encrypt_MCSR(types::double3d image3d, uint32_t numSlots, int depth,
    int max_channel,  CryptoContext<DCRTPoly> cryptocontext, 
    KeyPair<lbcrypto::DCRTPoly> Keypair, std::vector<Ciphertext<DCRTPoly>> &x_ctxt);

void Encrypt_MCSR_P(types::double3d& image3d, uint32_t numSlots, int depth,
    CryptoContext<DCRTPoly> cryptocontext, int enc_height_start, int enc_height_end, int enc_width_start, int enc_width_end,
    KeyPair<lbcrypto::DCRTPoly> Keypair, std::vector<Ciphertext<DCRTPoly>> &x_ctxt);

