#pragma once
#include "../cnn/model.hpp"
#include <cstdint>
#include <string>

void InitBaselineTest(Network& model, const std::string& resnet18_path, const std::string& imagenet_image_path, uint32_t numSlots, int test_nums, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair, int height_start, int height_end, int width_start, int width_end);
