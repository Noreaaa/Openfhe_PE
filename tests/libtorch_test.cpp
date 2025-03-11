#include <torch/torch.h>
#include <iostream>

int main() {
    // 创建一个 2x3 的随机张量
    torch::Tensor tensor = torch::rand({2, 3});

    // 打印张量
    std::cout << "Random Tensor: \n" << tensor << std::endl;

    // 检查是否有CUDA支持
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        tensor = tensor.to(torch::kCUDA); // 将张量移动到GPU
        std::cout << "Tensor on GPU: \n" << tensor << std::endl;
    } else {
        std::cout << "CUDA is not available. Running on CPU." << std::endl;
    }

    return 0;
}