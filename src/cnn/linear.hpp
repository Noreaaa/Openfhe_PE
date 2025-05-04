#pragma once 

#include "layer.hpp"


class Linear_P : public Layer {
    public:
        Linear_P(
            PLayerType layer_type,
            std::string layer_name,
            types::vector2d<double>& weights,
            vector<double>& bias,
            int batch_size
        );

        ~Linear_P();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            vector<double>& y_pts) override;

        void forward(vector<double>& input, vector<double>& output) override;

    private:
        types::vector2d<double> weights_;
        vector<double> bias_;
        int batch_size_;
        

};

void GoldenLinear_3d_input(double3d& input, types::double2d& weights, vector<double>& bias, vector<double>& output);

void GoldenLinear(vector<double>& input, types::double2d& weights, vector<double>& bias);