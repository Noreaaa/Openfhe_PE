#pragma once 

#include "openfhe.h"
#include <random>
#include "../utils/types.hpp"
#include "../utils/globals.hpp"
#include "../utils/helper.hpp"
using namespace lbcrypto;
using types::double3d;
using types::double2d;
using std::vector;
using types::double4d;

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

void Encrypt_MCSR_P(types::double3d& image3d, uint32_t numSlots, 
    CryptoContext<DCRTPoly> cryptocontext, int enc_height_start, int enc_height_end, int enc_width_start, int enc_width_end,
    KeyPair<lbcrypto::DCRTPoly> Keypair, types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt);

void Encrypt_MCSR_P_COMPACT(types::double3d& image3d, uint32_t numSlots, 
CryptoContext<DCRTPoly> cryptocontext, int enc_height_start, int enc_height_end, int enc_width_start, int enc_width_end,
KeyPair<lbcrypto::DCRTPoly> Keypair, types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt);

void Gen_random_cts2d(uint32_t numSlots, uint32_t valid_size, int dim1, int dim2,
         CryptoContext<DCRTPoly> cryptocontext, KeyPair<lbcrypto::DCRTPoly> Keypair, 
         types::vector2d<Ciphertext<DCRTPoly>> &x_ctxt);

std::vector<Ciphertext<DCRTPoly>> Encrypt_baseline(double3d& image3d, uint32_t numSlots, int image_size, int channels, CryptoContext<DCRTPoly> cryptocontext, KeyPair<DCRTPoly> Keypair);
