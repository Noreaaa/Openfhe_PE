#pragma once
#include "../cnn/model.hpp"
#include <cstdint>
#include <string>

void InitHCNN(Network& model, const std::string& hcnn_path, uint32_t numSlots, double eps);
