#include "dataloader.hpp"


types::double4d LoadConv2dWeight(const std::string& filename)
{
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];
    size_t dim2 = arr.shape[2];
    size_t dim3 = arr.shape[3];
    types::double4d weight(dim0, types::double3d(dim1, types::double2d(dim2, std::vector<double>(dim3, 0))));

    if (arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected float32 data type." << std::endl;
        return weight;
    }
    if (arr.fortran_order) {
        std::cerr << "Error: Fortran order not supported." << std::endl;
        return weight;
    }


    const float* float_data = arr.data<float>();


    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            for (size_t k = 0; k < dim2; ++k) {
                for (size_t l = 0; l < dim3; ++l) {
                    size_t idx = i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l;
                    weight[i][j][k][l] = static_cast<double>(float_data[idx]);
                }
            }
        }
    }

    std::cout << "Loaded Conv2d weights from " << filename << std::endl;

    std::cout << "Weight size: " << weight.size() << "x" << weight[0].size() << "x" << weight[0][0].size() << "x" << weight[0][0][0].size() << std::endl;

    return weight;
}

void LoadConv2dBias(const std::string& filename, std::vector<double>& bias)
{
    cnpy::NpyArray arr = cnpy::npy_load(filename);


    if (arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected float32 data type." << std::endl;
        return;
    }
    if (arr.fortran_order) {
        std::cerr << "Error: Fortran order not supported." << std::endl;
        return;
    }


    const float* float_data = arr.data<float>();


    if (arr.shape.size() != 1) {
        std::cerr << "Error: Expected 1D array for bias." << std::endl;
        return;
    }

    size_t dim0 = arr.shape[0];

    bias.resize(dim0);
    for (size_t i = 0; i < dim0; ++i) {
        bias[i] = static_cast<double>(float_data[i]);
    }

    std::cout << "Loaded Bias from " << filename << std::endl;
}

void LoadLinearWeight(const std::string& filename, types::vector2d<double>& weight)
{
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    if (arr.word_size != sizeof(float)) {
        std::cerr << "Error: Expected float32 data type." << std::endl;
        return;
    }
    if (arr.fortran_order) {
        std::cerr << "Error: Fortran order not supported." << std::endl;
        return;
    }

    const float* float_data = arr.data<float>();

    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];
    weight.resize(dim0);
    for (size_t i = 0; i < dim0; ++i) {
        weight[i].resize(dim1);
        for (size_t j = 0; j < dim1; ++j) {
            size_t idx = i * dim1 + j;
            weight[i][j] = static_cast<double>(float_data[idx]);
        }
    }

    std::cout << "Loaded Linear weights from " << filename << std::endl;

    std::cout << "Weight size: " << weight.size() << "x" << weight[0].size() << std::endl;

}

const int IMAG_SIZE = 32;
const int CHANNELS = 3;
const int BYTES_PER_IMAGE = IMAG_SIZE * IMAG_SIZE * CHANNELS + 1; // 1 byte for label

void LoadImageCifar(const std::string& filename, types::double3d& image_3d, int& label, int index){
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file");
    image_3d.clear();
    file.seekg(index * BYTES_PER_IMAGE, std::ios::beg);

    unsigned char label_byte;
    file.read(reinterpret_cast<char*>(&label_byte), 1);
    label = static_cast<int>(label_byte);
    image_3d.resize(CHANNELS);
    for (int ch = 0; ch < CHANNELS; ++ch) {
        image_3d[ch].resize(IMAG_SIZE);
        for (int row = 0; row < IMAG_SIZE; ++row) {
            image_3d[ch][row].resize(IMAG_SIZE);
            for (int col = 0; col < IMAG_SIZE; ++col) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image_3d[ch][row][col] = static_cast<double>(pixel) / 255.0;
            }
        }
    }

}

void NormalizeImage(types::double3d& image_3d) {
    for (int ch = 0; ch < CHANNELS; ++ch) {
        for (int row = 0; row < IMAG_SIZE; ++row) {
            for (int col = 0; col < IMAG_SIZE; ++col) {
                image_3d[ch][row][col] = (image_3d[ch][row][col] - 0.5) * 2.0; // Normalize to [-1, 1]
            }
        }
    }
}

void load_resnet18_8(types::double4d& w, const std::string& key,
                                const std::string& npz_path)
{
    static cnpy::npz_t npz = cnpy::npz_load(npz_path);

    auto it = npz.find(key);
    if (it == npz.end())
        throw std::runtime_error("Key \"" + key + "\" not found in npz");

    const cnpy::NpyArray& arr = it->second;
    if (arr.shape.size() != 4)
        throw std::runtime_error("Tensor \"" + key + "\" is not 4D");

    const size_t OC = arr.shape[0];
    const size_t IC = arr.shape[1];
    const size_t KH = arr.shape[2];
    const size_t KW = arr.shape[3];

    const float* src = arr.data<float>();


    size_t idx = 0;
    w.resize(OC);
    for (size_t o = 0; o < OC; ++o){
        w[o].resize(IC);
        for (size_t i = 0; i < IC; ++i){
            w[o][i].resize(KH);
            for (size_t h = 0; h < KH; ++h){
                w[o][i][h].resize(KW);
                for (size_t k = 0; k < KW; ++k){
                    w[o][i][h][k] = static_cast<double>(src[idx++]);
                }
            }
        }
    }

    // print sizes 
    std::cout << "Loaded ResNet-18 weights from " << key << std::endl;
    std::cout << "Weight size: " << w.size() << "x" << w[0].size() << "x" << w[0][0].size() << "x" << w[0][0][0].size() << std::endl;

}

void load_resnet18_8(types::double3d& w, const std::string& key,
                                const std::string& npz_path)
{
    static cnpy::npz_t npz = cnpy::npz_load(npz_path);

    auto it = npz.find(key);
    if (it == npz.end())
        throw std::runtime_error("Key \"" + key + "\" not found in npz");

    const cnpy::NpyArray& arr = it->second;
    if (arr.shape.size() != 3)
        throw std::runtime_error("Tensor \"" + key + "\" is not 2D");

    const size_t dim1 = arr.shape[0];
    const size_t dim2 = arr.shape[1];
    const size_t dim3 = arr.shape[2];


    const float* src = arr.data<float>();



    size_t idx = 0;
    w.resize(dim1);
    for (size_t i = 0; i < dim1; ++i){
        w[i].resize(dim2);
        for (size_t j = 0; j < dim2; ++j){
            w[i][j].resize(dim3);
            for (size_t k = 0; k < dim3; ++k)
                w[i][j][k] = static_cast<double>(src[idx++]);
        }
    }


}

void load_resnet18_8(types::double2d& w, const std::string& key,
                                const std::string& npz_path)
{
    static cnpy::npz_t npz = cnpy::npz_load(npz_path);

    auto it = npz.find(key);
    if (it == npz.end())
        throw std::runtime_error("Key \"" + key + "\" not found in npz");

    const cnpy::NpyArray& arr = it->second;
    if (arr.shape.size() != 2)
        throw std::runtime_error("Tensor \"" + key + "\" is not 2D");

    const size_t dim1 = arr.shape[0];
    const size_t dim2 = arr.shape[1];


    const float* src = arr.data<float>();



    size_t idx = 0;
    w.resize(dim1);
    for (size_t i = 0; i < dim1; ++i){
        w[i].resize(dim2);
        for (size_t j = 0; j < dim2; ++j)
                w[i][j] = static_cast<double>(src[idx++]);
    }


}


void load_resnet18_8(std::vector<double>& w, const std::string& key,
                                const std::string& npz_path)
{
    static cnpy::npz_t npz = cnpy::npz_load(npz_path);

    auto it = npz.find(key);
    if (it == npz.end())
        throw std::runtime_error("Key \"" + key + "\" not found in npz");

    const cnpy::NpyArray& arr = it->second;

    const size_t dim1 = arr.shape[0];
    const float* src = arr.data<float>();
    size_t idx = 0;
    w.resize(dim1);
    for (size_t i = 0; i < dim1; ++i){
        w[i] = static_cast<double>(src[idx++]);
    }


}

types::double3d load_layer3_0_feat(const std::string& npy_path)
{
    cnpy::NpyArray arr = cnpy::npy_load(npy_path);

    if (arr.shape.size() != 3)
        throw std::runtime_error("Expected 3-D tensor, got "
                                 + std::to_string(arr.shape.size()) + "-D");

    size_t C  = arr.shape[0];
    size_t H  = arr.shape[1];
    size_t W  = arr.shape[2];

    const double* src = arr.data<double>();      // 因为 Python 存 float64
    types::double3d feat(C, std::vector<std::vector<double>>(H, std::vector<double>(W)));

    size_t idx = 0;
    for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w)
                feat[c][h][w] = src[idx++];


    return feat;
}


types::double3d load_bin_image_double(const std::string& filename) {
    const int C = 3;
    const int H = 224;
    const int W = 224;
    const size_t total_size = C * H * W;

    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("fail to open file: " + filename);
    }

    std::vector<float> buffer_f(total_size);
    file.read(reinterpret_cast<char*>(buffer_f.data()), total_size * sizeof(float));
    if (!file) {
        throw std::runtime_error("fail to read file: " + filename);
    }
    file.close();

    types::double3d image(C, types::double2d(H, std::vector<double>(W)));
    size_t idx = 0;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                image[c][h][w] = static_cast<double>(buffer_f[idx++]);
            }
        }
    }

    return image;
}