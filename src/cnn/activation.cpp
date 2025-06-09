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
    cout << layer_name_ << " forward" << endl;

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


void golden_Square(
    types::double3d& x_pts
) {
    int channels = x_pts.size();
    int height = x_pts[0].size();
    int width = x_pts[0][0].size();

    for (int c = 0; c < channels; ++c)
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w)
                x_pts[c][h][w] *= x_pts[c][h][w];
}


Relu_ss::Relu_ss(
    PLayerType layer_type,
    std::string layer_name,
    lbcrypto::LWEPrivateKey FHEWKey
) : Layer(layer_type, layer_name),
 FHEWKey_(FHEWKey) {
    CONSUMED_LEVEL++;
}

Relu_ss::~Relu_ss() {
}

void Relu_ss::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    cout << layer_name_ << " forward" << endl;

    y_cts.resize(x_cts.size());
    for (size_t i = 0; i < x_cts.size(); i++){
        y_cts[i].resize(x_cts[i].size());
    }
    
    
    for (size_t i = 0; i < x_pts.size(); i++){
        y_pts[i].resize(x_pts[i].size());
        for (size_t j = 0; j < x_pts[i].size(); j++){
            y_pts[i][j].resize(x_pts[i][j].size());
            for (size_t k = 0; k < x_pts[i][j].size(); k++){
                y_pts[i][j][k] = y_pts[i][j][k] >= 0 ? y_pts[i][j][k] : 0; // relu operation
            }
        }
    }
    // evaluate with sign evaluation in FHEW 1 if negative, 0 if positive
    for (size_t i = 0; i < x_cts.size(); i++){
        for (size_t j = 0; j < x_cts[i].size(); j++){
            auto LWECiphertexts = CRYPTOCONTEXT -> EvalCKKStoFHEW(x_cts[i][j], 0);
            std::vector<double> relu_signs;
            for (size_t k = 0; k < LWECiphertexts.size(); k++){
                LWEPlaintext plainLWE;
                LWECiphertext LWESign = CCLWE->EvalSign(LWECiphertexts[k]);
                CCLWE->Decrypt(FHEWKey_, LWESign, &plainLWE, 2);
                //plaintext is int64_t, 1 if negative 0 if positive
                if (plainLWE == 1)
                    relu_signs.push_back(0);
                else 
                    relu_signs.push_back(1);
            }
            y_cts[i][j] = CRYPTOCONTEXT->EvalMult(
                x_cts[i][j], 
                CRYPTOCONTEXT->MakeCKKSPackedPlaintext(relu_signs)
            );
        }
    }
}






void golden_Relu(types::double3d& x_pts){
    int channels = x_pts.size();
    int height = x_pts[0].size();
    int width = x_pts[0][0].size();

    for (int c = 0; c < channels; ++c)
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w)
                if (x_pts[c][h][w] < 0) x_pts[c][h][w] = 0; 
}