#pragma once 

#include "cnpy.h"
#include "openfhe.h"
#include "globals.hpp"
#include <vector>
#include <iostream>
#include "types.hpp"

void LoadConv2dWeight(const std::string& filename, types::double3d& weight);