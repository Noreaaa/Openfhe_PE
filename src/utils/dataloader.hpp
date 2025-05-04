#pragma once 

#include "cnpy.h"
#include "openfhe.h"
#include "globals.hpp"
#include <vector>
#include <iostream>
#include "types.hpp"

types::double4d LoadConv2dWeight(const std::string& filename);

void LoadLinearWeight(const std::string& filename, types::vector2d<double>& weight);

void LoadConv2dBias(const std::string& filename, std::vector<double>& bias);
