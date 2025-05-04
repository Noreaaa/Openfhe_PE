#include "bootstrap.hpp"


Bootstrap_P::Bootstrap_P(
    PLayerType layer_type,
    std::string layer_name
) : Layer(layer_type, layer_name) {
    CONSUMED_LEVEL++;
}
Bootstrap_P::~Bootstrap_P() {
}

void Bootstrap_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    y_pts = x_pts;
    y_cts.resize(x_cts.size());
    for (size_t i = 0; i < x_cts.size(); ++i) {
        y_cts[i].resize(x_cts[i].size());
    }

    size_t inner_dim = y_cts[0].size();

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (size_t i = 0; i < x_cts.size(); i++){
        for (size_t j = 0; j < inner_dim; j++){
            y_cts[i][j] = CRYPTOCONTEXT->EvalBootstrap(x_cts[i][j]);
        }
    }
    #define DEBUG
    #ifdef DEBUG
    std::cout << "do bootstrapping" << std::endl;
    #endif
}