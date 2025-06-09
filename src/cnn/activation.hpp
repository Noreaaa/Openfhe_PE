#pragma once 

#include "layer.hpp"


class Square : public Layer {
    public:
        Square(
            PLayerType layer_type,
            std::string layer_name
        );
        ~Square();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

};


void golden_Square(
    types::double3d& x_pts
);

class Relu_ss : public Layer {
    public:
        Relu_ss(
            PLayerType layer_type,
            std::string layer_name,
            lbcrypto::LWEPrivateKey FHEWKey
        );
        ~Relu_ss();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

    private:
        lbcrypto::LWEPrivateKey FHEWKey_;
};

void golden_Relu(
    types::double3d& x_pts
);