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