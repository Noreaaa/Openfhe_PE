// deprecated file
#pragma once

#include "layer.hpp"
#include "conv2d.hpp"
#include <utility>
#include <cassert>
#include <cstdlib>



class Conv2dBN_P_L : public Layer {
    public:
        Conv2dBN_P_L(
            PLayerType layer_type,
            std::string layer_name,
            types::double4d& filters,
            int stride,
            int padding,
            uint32_t batch_size,
            std::vector<double> & gamma,
            std::vector<double> & beta,
            std::vector<double> & mean,
            std::vector<double> & var,
            double epsilon,
            PLayerType next_pool_layer_type
        );
        ~Conv2dBN_P_L();

        Plaintext GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx);

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

    private:
        types::double4d filters_;
        int stride_;
        int padding_;
        uint32_t batch_size_;
        std::vector<double> gamma_;
        std::vector<double> beta_;
        std::vector<double> mean_;
        std::vector<double> var_;
        double epsilon_;
        PLayerType next_pool_layer_type_;
};
