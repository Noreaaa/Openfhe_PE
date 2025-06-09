#pragma once
#include "layer.hpp"
#include "conv2d.hpp"

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
};

class SumPooling_P : public Layer {
    public:
        SumPooling_P(
            PLayerType layer_type,
            std::string layer_name,
            int kernel_size,
            int stride,
            int padding,
            uint32_t batch_size
        );
    ~SumPooling_P();

    void forward(vector<Ciphertext<DCRTPoly>>& x_cts, vector<Ciphertext<DCRTPoly>>& y_cts, double3d& x_pts, double3d& y_pts);


    private:
        int kernel_size_;
        int stride_;
        int padding_;
        uint32_t batch_size_;
};



class AvgPooling_P : public Layer {
    public:
        AvgPooling_P(
            PLayerType layer_type,
            std::string layer_name,
            int kernel_size,
            int stride,
            int padding,
            uint32_t batch_size
        );
        ~AvgPooling_P();

    void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
        types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

    private:
        int kernel_size_;
        int stride_;
        int padding_;
        uint32_t batch_size_;
};

bool isOutputEncryptedFromPooling(int val, int kernel_size, int enc_start, int enc_end);

void golden_AvgPooling(types::double3d& x_pts, int kernel_size, int stride);

std::vector<double> sumAdjacentPairs(std::vector<double>& input);