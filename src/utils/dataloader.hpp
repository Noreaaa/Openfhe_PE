#pragma once 

#include "cnpy.h"
#include "openfhe.h"
#include "globals.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include "types.hpp"

types::double4d LoadConv2dWeight(const std::string& filename);

void LoadLinearWeight(const std::string& filename, types::vector2d<double>& weight);

void LoadConv2dBias(const std::string& filename, std::vector<double>& bias);

void LoadImageCifar(const std::string& filename, types::double3d& image_3d, int& label, int index);

void NormalizeImage(types::double3d& image_3d);

void load_resnet18_8(types::double4d& w, const std::string& key, const std::string& npz_path);

void load_resnet18_8(types::double3d& w, const std::string& key, const std::string& npz_path);

void load_resnet18_8(types::double2d& w, const std::string& key, const std::string& npz_path);

void load_resnet18_8(std::vector<double>& w, const std::string& key, const std::string& npz_path);

types::double3d load_layer3_0_feat(const std::string& npy_path);

types::double3d load_bin_image_double(const std::string& filename);