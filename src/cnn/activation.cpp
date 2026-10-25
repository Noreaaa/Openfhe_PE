#include "activation.hpp"
#include "encrypted_relu.hpp"
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
    lbcrypto::LWEPrivateKey /*FHEWKey*/,
    uint32_t batch_size
) : Layer(layer_type, layer_name),
 batch_size_(batch_size) {
    CONSUMED_LEVEL++;
}

Relu_ss::~Relu_ss() {
}

void Relu_ss::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {

    int input_w = x_pts[0][0].size();
    cout << layer_name_ << " forward" << endl;

    y_cts.resize(x_cts.size());

    for (size_t i = 0; i < x_cts.size(); i++){
        y_cts[i].resize(x_cts[i].size());
    }

    size_t dim_2 = x_cts[0].size();
    y_pts.resize(x_pts.size());
    for (size_t i = 0; i < x_pts.size(); i++){
        y_pts[i].resize(x_pts[i].size());
        for (size_t j = 0; j < x_pts[i].size(); j++){
            y_pts[i][j].resize(x_pts[i][j].size());
            for (size_t k = 0; k < x_pts[i][j].size(); k++){
                y_pts[i][j][k] = x_pts[i][j][k] >= 0 ? x_pts[i][j][k] : 0;
            }
        }
    }

    std::cout << "finish unencrypted relu_ss" << std::endl;
    for (size_t i = 0; i < x_cts.size(); ++i) for (size_t j = 0; j < dim_2; ++j) {
        auto amplified = CRYPTOCONTEXT->EvalMult(x_cts[i][j], 1000.0);
        auto values = CRYPTOCONTEXT->EvalCKKStoFHEW(amplified, batch_size_);
        std::vector<size_t> active;
        for (size_t k = 0; k + input_w <= batch_size_; k += input_w)
            for (int w = ENCRYPTED_WIDTH_START; w <= ENCRYPTED_WIDTH_END; ++w)
                active.push_back(k+w);
        auto mask = EncryptedReluMask(CRYPTOCONTEXT, CCLWE, values, active, batch_size_, nullptr, true);
        y_cts[i][j] = CRYPTOCONTEXT->EvalMult(x_cts[i][j], mask);
    }
}

void Relu_ss::forward_C(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    y_pts = x_pts;
    for (auto& channel : y_pts) for (auto& row : channel) for (auto& v : row) v = std::max(0.0,v);
    y_cts.resize(x_cts.size());
    const size_t width = ENCRYPTED_WIDTH_END-ENCRYPTED_WIDTH_START+1;
    for (size_t i=0;i<x_cts.size();++i) {
        y_cts[i].resize(x_cts[i].size());
        for (size_t j=0;j<x_cts[i].size();++j) {
            const size_t valid = j+1 == x_cts[i].size() ? REMAINING_SLOTS : (batch_size_/width)*width;
            auto amplified = CRYPTOCONTEXT->EvalMult(x_cts[i][j],1000.0);
            auto values = CRYPTOCONTEXT->EvalCKKStoFHEW(amplified,batch_size_);
            std::vector<size_t> active;
            for(size_t k=0;k<valid;++k) active.push_back(k);
            auto mask = EncryptedReluMask(CRYPTOCONTEXT,CCLWE,values,active,batch_size_,nullptr,true);
            y_cts[i][j] = CRYPTOCONTEXT->EvalMult(x_cts[i][j],mask);
        }
    }
}

void Relu_ss::forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
    std::vector<Ciphertext<DCRTPoly>>& y_cts) {
    y_cts.resize(x_cts.size());
    std::vector<size_t> active;
    for(const auto& row : VALID_INDEX_MAP) for(int slot : row) active.push_back(slot);
    for(size_t i=0;i<x_cts.size();++i) {
        auto amplified = CRYPTOCONTEXT->EvalMult(x_cts[i],1000.0);
        auto values = CRYPTOCONTEXT->EvalCKKStoFHEW(amplified,batch_size_);
        auto mask = EncryptedReluMask(CRYPTOCONTEXT,CCLWE,values,active,batch_size_,nullptr,true);
        y_cts[i] = CRYPTOCONTEXT->EvalMult(x_cts[i],mask);
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

Relu_appx::Relu_appx(
    PLayerType layer_type,
    std::string layer_name
) : Layer(layer_type, layer_name) {
    CONSUMED_LEVEL++;
}

Relu_appx::~Relu_appx() {
}

// f1 = 10.8541842577442 x - 62.2833925211098 x^3 + 114.36920213692 x^5 - 62.8023982549446 x^7
lbcrypto::Ciphertext<DCRTPoly> EvalF1(lbcrypto::Ciphertext<DCRTPoly>& x) {
    auto x2 = CRYPTOCONTEXT->EvalMult(x, x);  // x^2
    if (RESCALE_REQUIRED)
        x2 = CRYPTOCONTEXT->Rescale(x2);
    auto term = CRYPTOCONTEXT->EvalMult(x2, -62.2834);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 114.3692);
    term = CRYPTOCONTEXT->EvalMult(term, x2);  // x^4
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, -62.8023);
    term = CRYPTOCONTEXT->EvalMult(term, x2);  // x^6
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 10.8542);
    auto result = CRYPTOCONTEXT->EvalMult(x, term);  // x *
    if (RESCALE_REQUIRED)
        result = CRYPTOCONTEXT->Rescale(result);
    return result;
}

lbcrypto::Ciphertext<DCRTPoly> EvalF2(lbcrypto::Ciphertext<DCRTPoly>& x) {
    auto x2 = CRYPTOCONTEXT->EvalMult(x, x);  // x^2
    if (RESCALE_REQUIRED)
        x2 = CRYPTOCONTEXT->Rescale(x2);
    auto term = CRYPTOCONTEXT->EvalMult(x2, -5.8499);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 2.9438);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, -0.4545);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 4.1398);
    auto result = CRYPTOCONTEXT->EvalMult(x, term);
    if (RESCALE_REQUIRED)
        result = CRYPTOCONTEXT->Rescale(result);
    return result;
}


lbcrypto::Ciphertext<DCRTPoly> EvalF3(lbcrypto::Ciphertext<DCRTPoly>& x) {
    auto x2 = CRYPTOCONTEXT->EvalMult(x, x);  // x^2
    if (RESCALE_REQUIRED)
        x2 = CRYPTOCONTEXT->Rescale(x2);
    auto term = CRYPTOCONTEXT->EvalMult(x2, 0.2464);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, -2.0430);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 6.9417);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, -12.4917);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 12.8908);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, -7.8423);
    term = CRYPTOCONTEXT->EvalMult(term, x2);
    if (RESCALE_REQUIRED)
        term = CRYPTOCONTEXT->Rescale(term);
    term = CRYPTOCONTEXT->EvalAdd(term, 3.2996);
    auto result = CRYPTOCONTEXT->EvalMult(x, term);
    if (RESCALE_REQUIRED)
        result = CRYPTOCONTEXT->Rescale(result);
    return result;
}

//dergree 7 polynomial approx
void Relu_appx::forward(std::vector<Ciphertext<DCRTPoly>>& x_cts,
    std::vector<Ciphertext<DCRTPoly>>& y_cts) {
    cout << layer_name_ << " forward" << endl;
    y_cts.resize(x_cts.size());
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < x_cts.size(); i++){
        auto x2 = CRYPTOCONTEXT->EvalMult(x_cts[i], x_cts[i]);
        auto term = CRYPTOCONTEXT->EvalMult(x2, 1.1551725);
        term = CRYPTOCONTEXT->EvalAdd(term, -2.089185);
        term = CRYPTOCONTEXT->EvalMult(term, x2);
        term = CRYPTOCONTEXT->EvalAdd(term, 1.4340124);
        term = CRYPTOCONTEXT->EvalMult(term, x2);
        auto halfx = CRYPTOCONTEXT->EvalMult(x_cts[i], 0.5);
        term = CRYPTOCONTEXT->EvalAdd(term, halfx);
        term = CRYPTOCONTEXT->EvalBootstrap(term, 1, 0);
        y_cts[i] = CRYPTOCONTEXT->EvalAdd(term, 0.0229645);
    }
}

//degree 7 polynomial approx
void Relu_appx::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
            types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts){
    cout << layer_name_ << " forward" << endl;
    y_cts.resize(x_cts.size());
    y_pts.resize(x_pts.size());
    for (size_t i = 0; i < x_cts.size(); i++){
        y_cts[i].resize(x_cts[i].size());
        for (size_t j = 0; j < x_cts[i].size(); j++){
            y_pts[i].resize(x_pts[i].size());
        }
    }
    size_t dim_2 = x_cts[0].size();

    for (size_t i = 0; i < x_pts.size(); i++){
        y_pts[i].resize(x_pts[i].size());
        for (size_t j = 0; j < x_pts[i].size(); j++){
            y_pts[i][j].resize(x_pts[i][j].size());
            for (size_t k = 0; k < x_pts[i][j].size(); k++){
                y_pts[i][j][k] = x_pts[i][j][k] >= 0 ? x_pts[i][j][k] : 0; // relu operation
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (size_t i = 0; i < x_cts.size(); i++){
        for (size_t j = 0; j < dim_2; j++){
            auto x2 = CRYPTOCONTEXT->EvalMult(x_cts[i][j], x_cts[i][j]);
            auto term = CRYPTOCONTEXT->EvalMult(x2, 1.1551725);
            term = CRYPTOCONTEXT->EvalAdd(term, -2.089185);
            term = CRYPTOCONTEXT->EvalMult(term, x2);
            term = CRYPTOCONTEXT->EvalAdd(term, 1.4340124);
            term = CRYPTOCONTEXT->EvalMult(term, x2);
            auto halfx = CRYPTOCONTEXT->EvalMult(x_cts[i][j], 0.5);
            term = CRYPTOCONTEXT->EvalAdd(term, halfx);
            term = CRYPTOCONTEXT->EvalBootstrap(term, 1, 0);
            y_cts[i][j] = CRYPTOCONTEXT->EvalAdd(term, 0.0229645);

            //auto f1x = EvalF1(x_cts[i][j]);
            //f1x = CRYPTOCONTEXT->EvalBootstrap(f1x, 1, 50);
            //auto f2x = EvalF2(f1x);
            //f2x = CRYPTOCONTEXT->EvalBootstrap(f2x, 1, 50);
            //auto f3x = EvalF3(f2x);
            //f3x = CRYPTOCONTEXT->EvalBootstrap(f3x, 1, 50);
            //auto x_sign = CRYPTOCONTEXT->EvalMult(x_cts[i][j],f3x);
            //if (RESCALE_REQUIRED)
            //    x_sign = CRYPTOCONTEXT->Rescale(x_sign);
            //auto relu = CRYPTOCONTEXT->EvalAdd(x_cts[i][j], x_sign);
            //y_cts[i][j] = CRYPTOCONTEXT->EvalMult(relu, 0.5);
            ////y_cts[i][j] = CRYPTOCONTEXT->EvalBootstrap(relu, 1, 0);
            //if (RESCALE_REQUIRED)
            //    y_cts[i][j] = CRYPTOCONTEXT->Rescale(y_cts[i][j]);
        }
    }
}

void test_relu_appx(lbcrypto::Ciphertext<DCRTPoly>& x) {
    auto f1x = EvalF1(x);
    f1x = CRYPTOCONTEXT->EvalBootstrap(f1x, 1, 0);
    //check intermediate result
    Plaintext res1;
    CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, f1x, &res1);
    cout << "f1x result: " << res1 << endl;
    auto f2x = EvalF2(f1x);
    f2x = CRYPTOCONTEXT->EvalBootstrap(f2x, 1, 0);
    //check intermediate result
    Plaintext res2;
    CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, f2x, &res2);
    cout << "f2x result: " << res2 << endl;
    auto f3x = EvalF3(f2x);
    auto x_sign = CRYPTOCONTEXT->EvalMult(x, f3x);
    //check intermediate result
    Plaintext res3;
    CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, x_sign, &res3);
    cout << "x_sign result: " << res3 << endl;
    if (RESCALE_REQUIRED)
        x_sign = CRYPTOCONTEXT->Rescale(x_sign);
    auto relu = CRYPTOCONTEXT->EvalAdd(x, x_sign);
    auto result = CRYPTOCONTEXT->EvalMult(relu, 0.5);
    if (RESCALE_REQUIRED)
        result = CRYPTOCONTEXT->Rescale(result);

    Plaintext res;
    CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, result, &res);
    cout << "relu_appx result: " << res << endl;
    cout << "relu_appx level: " << result->GetLevel() << endl;

}