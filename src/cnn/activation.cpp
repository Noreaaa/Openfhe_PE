#include "activation.hpp"
using std::cout;
using std::endl;

Square::Square(
    PLayerType layer_type,
    std::string layer_name
) : Layer(layer_type, layer_name) {
    CONSUMED_LEVEL++;
}

Square::~Square() {
}
void Square::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    // Check if the input and output dimensions match
    
    y_cts.resize(x_cts.size());
    y_pts.resize(x_pts.size());

    for (size_t i = 0; i < x_pts.size(); i++){
        y_pts[i].resize(x_pts[i].size());
        for (size_t j = 0; j < x_pts[i].size(); j++){
            y_pts[i][j].resize(x_pts[i][j].size());
            for (size_t k = 0; k < x_pts[i][j].size(); k++){
                y_pts[i][j][k] = x_pts[i][j][k] * x_pts[i][j][k];
            }
        }
    }
    //#define DEBUG
    #ifdef DEBUG
    cout << "check the unencrypted parts" << endl;
    print_3d(y_pts);
    #endif
    // Perform the square operation on each element

    for (size_t i = 0; i < x_cts.size(); ++i) {
        y_cts[i].resize(x_cts[i].size());
    }
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < x_cts.size(); ++i) {
        for (size_t j = 0; j < x_cts[i].size(); ++j) {
            y_cts[i][j] = CRYPTOCONTEXT->EvalSquare(x_cts[i][j]);
        }
    }

    //#define Y_CTS_CHECK
    #ifdef Y_CTS_CHECK
    cout << "value check for encrypted parts" << endl;
    for(int i = 0; i < static_cast<int>(y_cts.size()); i++){
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            Plaintext res;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &res);
            cout << "y_cts[" << i << "][" << j << "]: " << res << endl;
        }
    }
    cout << "check the avaliable level" << endl;
    for(int i = 0; i < static_cast<int>(y_cts.size()); i++){
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            cout << "y_cts[" << i << "][" << j << "]: " << y_cts[i][j]->GetLevel() << endl;
        }
    }
    #endif

}