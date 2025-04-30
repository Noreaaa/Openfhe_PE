#include "dataloader.hpp"


void LoadConv2dWeight(const std::string& filename, types::double3d& weight)
{
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // 检查是不是float32
    if (arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected float32 data type." << std::endl;
        return;
    }
    if (arr.fortran_order) {
        std::cerr << "Error: Fortran order not supported." << std::endl;
        return;
    }

    // 先拿到 float* 指针
    const float* float_data = arr.data<float>();

    // 计算总元素数
    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];
    size_t dim2 = arr.shape[2];

    // 重新组织成 3D vector
    weight.resize(dim0);
    for (size_t i = 0; i < dim0; ++i) {
        weight[i].resize(dim1);
        for (size_t j = 0; j < dim1; ++j) {
            weight[i][j].resize(dim2);
            for (size_t k = 0; k < dim2; ++k) {
                size_t idx = i * dim1 * dim2 + j * dim2 + k;
                weight[i][j][k] = static_cast<double>(float_data[idx]);
            }
        }
    }

    std::cout << "Loaded Conv2d weights from " << filename << std::endl;
}

void LoadConv2dBias(const std::string& filename, std::vector<double>& bias)
{
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // 检查是不是 float32
    if (arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected float32 data type." << std::endl;
        return;
    }
    if (arr.fortran_order) {
        std::cerr << "Error: Fortran order not supported." << std::endl;
        return;
    }

    // 先拿到 float* 指针
    const float* float_data = arr.data<float>();

    // bias应该是一维数组
    if (arr.shape.size() != 1) {
        std::cerr << "Error: Expected 1D array for bias." << std::endl;
        return;
    }

    size_t dim0 = arr.shape[0];

    // 重新组织成 1D vector
    bias.resize(dim0);
    for (size_t i = 0; i < dim0; ++i) {
        bias[i] = static_cast<double>(float_data[i]);
    }

    std::cout << "Loaded Bias from " << filename << std::endl;
}