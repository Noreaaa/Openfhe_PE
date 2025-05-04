#pragma once

#include "layer.hpp"

class Bootstrap_P : public Layer {
    public:
        Bootstrap_P(
            PLayerType layer_type,
            std::string layer_name
        );
        ~Bootstrap_P();

        void forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) override;
    
    private:
        
};

