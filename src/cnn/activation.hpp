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
            lbcrypto::LWEPrivateKey FHEWKey,
            uint32_t batch_size
        );
        ~Relu_ss();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

        void forward_C(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

        void forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
            std::vector<Ciphertext<DCRTPoly>>& y_cts) override;

    private:
        lbcrypto::LWEPrivateKey FHEWKey_;
        uint32_t batch_size_;
};


class Relu_appx : public Layer {
    public:
        Relu_appx(
            PLayerType layer_type,
            std::string layer_name
        );
        ~Relu_appx();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;

        void forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
            std::vector<Ciphertext<DCRTPoly>>& y_cts) override;


};

void golden_Relu(
    types::double3d& x_pts
);


lbcrypto::Ciphertext<DCRTPoly> EvalF1(lbcrypto::Ciphertext<DCRTPoly>& x);

lbcrypto::Ciphertext<DCRTPoly> EvalF2(lbcrypto::Ciphertext<DCRTPoly>& x);

lbcrypto::Ciphertext<DCRTPoly> EvalF3(lbcrypto::Ciphertext<DCRTPoly>& x);

void test_relu_appx(lbcrypto::Ciphertext<DCRTPoly>& x);