#pragma once
#include "layer.hpp"

// Independent sparse-packing layers. Only SparseConv2dBN_P::forward is implemented.
class SparseConv2d : public Layer {
public:
    SparseConv2d(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double3d filters, std::vector<double> biases, int stride, int padding);
    ~SparseConv2d() override = default;
    using Layer::forward;
    using Layer::forward_C;
    void forward(std::vector<Ciphertext<DCRTPoly>>&, std::vector<Ciphertext<DCRTPoly>>&) override;
private:
    uint32_t ring_dim_;
    uint32_t packing_slots_;
    types::double3d filters_;
    std::vector<double> biases_;
    int stride_;
    int padding_;
};

class SparseConv2d_C : public Layer {
public:
    SparseConv2d_C(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double4d filters, int stride, int input_height, int input_width);
    ~SparseConv2d_C() override = default;
    using Layer::forward;
    using Layer::forward_C;
    void forward(std::vector<Ciphertext<DCRTPoly>>&, std::vector<Ciphertext<DCRTPoly>>&) override;
private:
    uint32_t ring_dim_;
    uint32_t packing_slots_;
    types::double4d filters_;
    int stride_;
    int input_height_;
    int input_width_;
};

class SparseConv2d_P : public Layer {
public:
    SparseConv2d_P(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double3d filters, std::vector<double> biases, int stride, int padding, PLayerType next_pool_layer_type);
    ~SparseConv2d_P() override = default;
    using Layer::forward;
    using Layer::forward_C;
    void forward(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) override;
private:
    uint32_t ring_dim_;
    uint32_t packing_slots_;
    types::double3d filters_;
    std::vector<double> biases_;
    int stride_;
    int padding_;
    PLayerType next_pool_layer_type_;
};

class SparseConv2dBN_P : public Layer {
public:
    SparseConv2dBN_P(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double4d filters, int stride, int padding, std::vector<double> gamma, std::vector<double> beta, std::vector<double> mean, std::vector<double> var, double epsilon, std::vector<double> bias, PLayerType next_pool_layer_type);
    ~SparseConv2dBN_P() override = default;
    using Layer::forward;
    using Layer::forward_C;
    void forward(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) override;
    void forward_C(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) override;
    // Opt in after key setup. The fast path supports automatic rescaling and stride=1/2 with standard same padding.
    void EnableFastPath(uint32_t input_width, bool generate_keys = true);
    void forward_reference(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&,
                           types::vector2d<Ciphertext<DCRTPoly>>&, double3d&);
private:
    void forward_fast(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&,
                      types::vector2d<Ciphertext<DCRTPoly>>&, double3d&);
    uint32_t fast_width_ = 0;
    Plaintext Encode(const std::vector<double>& values) const;
    void ValidateInput(const types::vector2d<Ciphertext<DCRTPoly>>& cts, const double3d& pts) const;
    Plaintext GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx, int channels_per_cts);
    uint32_t ring_dim_;
    uint32_t packing_slots_;
    types::double4d filters_;
    int stride_;
    int padding_;
    std::vector<double> gamma_;
    std::vector<double> beta_;
    std::vector<double> mean_;
    std::vector<double> var_;
    double epsilon_;
    std::vector<double> bias_;
    PLayerType next_pool_layer_type_;
};
