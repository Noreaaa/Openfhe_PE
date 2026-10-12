#pragma once
#include "../cnn/model.hpp"
#include <cstdint>
#include <string>

void InitResNet18(Network& model, const std::string& resnet18_path, const std::string& imagenet_image_path, uint32_t numSlots, double eps, int act_type, lbcrypto::LWEPrivateKey privateKeyFHEW, lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair, int height_start, int height_end, int width_start, int width_end);
