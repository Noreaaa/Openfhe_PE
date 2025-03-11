#pragma once
#include "layer.hpp"


class SumPooling : public Layer {
    public:
        SumPooling(
            PLayerType layer_type,
            std::string layer_name,
            int kernel_size,
            int stride,
            int padding,
            uint32_t batch_size
        );
        ~SumPooling();

        void forward(vector<Ciphertext<DCRTPoly>>& x_cts, vector<Ciphertext<DCRTPoly>>& y_cts);

        private:
            int kernel_size_;
            int stride_;
            int padding_;
            uint32_t batch_size_;
}