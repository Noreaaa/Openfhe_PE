#pragma once
#include "globals.h"

#include <vector>

using namespace lbcrypto;

namespace types {

template <typename T>
using vector2d = std::vector<std::vector<T>>;

template <typename T>
using vector3d = std::vector<std::vector<std::vector<T>>>;

template <typename T>
using vector4d = std::vector<std::vector<std::vector<std::vector<T>>>>;

template <typename T>
using vector5d =
    std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>;

template <typename T>
using vector6d = std::vector<
    std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>>;

using ciphertext1d = std::vector<Ciphertext<DCRTPoly>>;


using float2d = vector2d<float>;
using float3d = vector3d<float>;
using float4d = vector4d<float>;
using float5d = vector5d<float>;
using float6d = vector6d<float>;

using double2d = vector2d<double>;
using double3d = vector3d<double>;
using double4d = vector4d<double>;
using double5d = vector5d<double>;
using double6d = vector6d<double>;


}  // namespace types
