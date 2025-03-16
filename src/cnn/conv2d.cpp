
#include "conv2d.hpp"
#include <utility>
#include <cassert>
#include <cstdlib>

using std::vector;
using std::pair;
using types::double4d;
using types::double3d;
using types::double2d;
using std::vector;
using std::cout;
using std::endl;


    #define IS_IN_RANGE(x, min_val, max_val) ((x) >= (min_val) && (x) <= (max_val))
	#define POSITIVE(x) ((x) < 0 ? 0 : (x))

Conv2d::Conv2d(
    PLayerType layer_type,
    std::string layer_name,
    double3d& filters,
    vector<double>& biases,
    int stride,
    int padding,
    uint32_t batch_size)
    : Layer(PLayerType::CONV_2D, layer_name),
    filters_(filters),
    biases_(biases),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size) {
        CONSUMED_LEVEL++;
}

Conv2d::~Conv2d() {}
// copy from conv test
#define DEBUG
void Conv2d::forward(vector<Ciphertext<DCRTPoly>>& x_cts,
    vector<Ciphertext<DCRTPoly>>& y_cts) {
    // need to know the input row size 
    // if it is the first layer padding is not available
    
    #ifdef DEBUG
    cout << "Conv2d forward" << endl;
    #endif
    int f_n = filters_.size();
    int f_h = filters_[0].size();
    int f_w = filters_[0][0].size();
    int output_h = (CURRENT_HEIGHT + 2 * padding_ - f_h) / stride_ + 1;
    int output_w = (CURRENT_WIDTH + 2 * padding_ - f_w) / stride_ + 1;
    std::vector<std::vector<std::vector<double>>> f_rows;
    f_rows.resize(f_n);
    y_cts.clear();
    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << f_n << endl;
    cout << "f_h: " << f_h << endl;
    cout << "f_w: " << f_w << endl;
    #endif
    // create filter vector to multiply with x_cts
    for (int n = 0; n < f_n; n++){
        for (int h = 0; h < f_h; h++){
            std::vector<double> filter_row_vec;
            for (int i = 0; i < CURRENT_CHANNEL * CURRENT_WIDTH; i++){
                if (i%CURRENT_WIDTH < f_w){
                    filter_row_vec.push_back(filters_[n][h][i%CURRENT_WIDTH]);
                }
                else{
                    filter_row_vec.push_back(0);
                }
            }
            #ifdef DEBUG
            cout << "filter_row_vec: " << endl;
            for(int i = 0; i < static_cast<int>(filter_row_vec.size()); i++){
                cout << filter_row_vec[i] << " ";
            }
            cout << endl;
            #endif
            f_rows[n].push_back(filter_row_vec);
        }
    }
    #ifdef DEBUG
    cout << "f_plaintexts[0][0] value check: " << endl;
    for (int i = 0; i < static_cast<int>(f_rows[0][0].size()); i++){
        cout << f_rows[0][0][i] << " ";
    }
    cout << endl;
    #endif

    
    for (int oh = 0; oh < output_h; oh++){
        Ciphertext<DCRTPoly> y_ct;
        for (int n = 0; n < f_n; n++){
            // if next layer requires padding; to do 
            for(int ow = 0; ow < output_w; ow++){
                // for each output value 

                // generate mask vector to multiply with the result
                std::vector<double> mask; 
                for (int i = 0; i < f_n * output_w; i++){
                    if(i == n * output_w + ow){
                        mask.push_back(1);
                    }
                    else{
                        mask.push_back(0);
                    }
                }
                auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);


                Ciphertext<DCRTPoly> temp_res;
                Plaintext filter; 

                for(int i = 0; i < f_h; i++){
                    //rotate the fiter row 
                    std::vector<double> filter_vec = rotateVector(f_rows[n][i], ow * stride_);
                    filter = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_vec);
                    // for i = 0 initialize the temp_res
                    if (i == 0){
                        temp_res = CRYPTOCONTEXT->EvalMult(x_cts[i+oh], filter);
                        continue;
                    }
                    // for each filter row, we need to calculate the dot product 
                    auto temp_res_row = CRYPTOCONTEXT->EvalMult(x_cts[i+oh], filter);
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_row);
                }

                if (ow == 0 && n == 0){
                    y_ct = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    y_ct = CRYPTOCONTEXT->EvalMult(y_ct, mask_plain);
                }
                else{
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    y_ct = CRYPTOCONTEXT->EvalAdd(y_ct, temp_res);
                }

            }
        }
        y_cts.push_back(y_ct);
    }

    CURRENT_HEIGHT = output_h;
    CURRENT_WIDTH = output_w;
    CURRENT_CHANNEL = f_n;
    #ifdef DEBUG
        for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
            Plaintext res;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i], &res);
            cout << "y_cts[" << i << "]: " << res << endl;
        }
    return;
    #endif
}



Conv2d_P::Conv2d_P(
    PLayerType layer_type,
    std::string layer_name,
    double3d& filters,
    vector<double>& biases,
    int stride,
    int padding,
    uint32_t batch_size)
    : Layer(PLayerType::CONV_2D, layer_name),
    filters_(filters),
    biases_(biases),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size) {
        CONSUMED_LEVEL++;
}

Conv2d_P::~Conv2d_P() {}



void Conv2d_P::forward(vector<Ciphertext<DCRTPoly>>& x_cts,
    vector<Ciphertext<DCRTPoly>>& y_cts, double3d& x_pts, double3d& y_pts) {
    #ifdef DEBUG
    cout << "Conv2d_Partial forward" << endl;
    #endif


    int filter_n = filters_.size();
    int filter_h = filters_[0].size();
    int filter_w = filters_[0][0].size();
    int output_h = (CURRENT_HEIGHT + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (CURRENT_WIDTH + 2 * padding_ - filter_w) / stride_ + 1;
    std::vector<std::vector<std::vector<double>>> f_rows;
    f_rows.resize(filter_n);
    y_cts.clear();


    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << filter_n << endl;
    cout << "f_h: " << filter_h << endl;
    cout << "f_w: " << filter_w << endl;
    #endif

    // create filter vector to multiply with x_cts
    for (int n = 0; n < filter_n; n++){
        for (int h = 0; h < filter_h; h++){
            std::vector<double> filter_row_vec;
            for (int i = 0; i < CURRENT_CHANNEL * CURRENT_WIDTH; i++){
                if (i%CURRENT_WIDTH < filter_w){
                    filter_row_vec.push_back(filters_[n][h][i%CURRENT_WIDTH]);
                }
                else{
                    filter_row_vec.push_back(0);
                }
            }
            #ifdef DEBUG
            cout << "filter_row_vec: " << endl;
            for(int i = 0; i < static_cast<int>(filter_row_vec.size()); i++){
                cout << filter_row_vec[i] << " ";
            }
            cout << endl;
            #endif
            f_rows[n].push_back(filter_row_vec);
        }
    }
    #ifdef DEBUG
    cout << "f_plaintexts[0][0] value check: " << endl;
    for (int i = 0; i < static_cast<int>(f_rows[0][0].size()); i++){
        cout << f_rows[0][0][i] << " ";
    }
    cout << endl;
    #endif

    // first we do the unencrypted convolution
    y_pts.resize(filter_n);
    for (int i = 0; i < filter_n; i++){
        y_pts[i].resize(output_h);
        for (int j = 0; j < output_h; j++){
            y_pts[i][j].resize(output_w);
        }
    }
    for (int fn = 0; fn < filter_n; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < CURRENT_CHANNEL; c++){
                                sum += x_pts[c][oh*stride_ + fh][ow*stride_ + fw] * 
                                filters_[fn][fh][fw];
                            }
                        }
                    }
                    y_pts[fn][oh][ow] = sum;
                }
            }
        }
    }
    #ifdef DEBUG
    cout << "value check for unencrypted parts" << endl;
    print_3d(y_pts);
    #endif

    // then we do re-encryption and addition
    // regarding encrypted rows
    int re_encrypt_start = 0;
    int re_encrypt_end = 0;
    for(int w = 0; w < output_w; w+=stride_){
        if (intervalsOverlap(w, w + filter_w, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            re_encrypt_start = w;
            break;
        }
    }
    for(int w = re_encrypt_start; w < output_w; w+=stride_){
        if (!intervalsOverlap(w, w + filter_w, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            re_encrypt_end = w;
            break;
        }
    }
    std::cout << "re_encrypt_start: " << re_encrypt_start << std::endl;
    std::cout << "re_encrypt_end: " << re_encrypt_end << std::endl;

    for(int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for(int c = 0; c < CURRENT_CHANNEL; c++){
            for(int w = 0; w < CURRENT_WIDTH; w++){
                if (isInRange(w, re_encrypt_start, re_encrypt_end)){
                    to_add.push_back(x_pts[c][h][w]);
                }
                else{
                    to_add.push_back(0);
                }
            }
        }
        auto to_add_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(to_add);
        x_cts[h - ENCRYPTED_HEIGHT_START] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START], to_add_plain);
    }

    //#define ADD_CTS_CHECK
    #ifdef ADD_CTS_CHECK
    std::cout << "check added cts:" << std::endl;
    for (int i = 0; i < static_cast<int>(x_cts.size()); i++){
        Plaintext plain;
        CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, x_cts[i], &plain);
        std::cout << plain << std::endl;
    }
    return;
    #endif

    for (int oh = 0; oh < output_h; oh++) {
        if (!isEncrypted_h(oh * stride_, filter_h)) {
            continue;
        }
    
        std::vector<Ciphertext<DCRTPoly>> y_vec_ct;
    
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            // 每个线程维护自己的临时 vector
            std::vector<Ciphertext<DCRTPoly>> local_y_vec_ct;
    
            #ifdef _OPENMP
            #pragma omp for collapse(2)
            #endif
            for (int fn = 0; fn < filter_n; fn++) {
                for (int ow = 0; ow < output_w; ow++) {
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w)) {
                        continue;
                    }
    
                    std::vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    auto temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.secretKey, temp_plain);
                    double sum = 0;
    
                    std::vector<double> mask(filter_n * output_w, 0);
                    mask[fn * output_w + ow] = 1;
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
    
                    for (int fh = 0; fh < filter_h; fh++) {
                        if (!isInRange(oh * stride_ + fh, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
                            for (int fw = 0; fw < filter_w; fw++) {
                                for (int c = 0; c < CURRENT_CHANNEL; c++) {
                                    sum += x_pts[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][fh][fw];
                                }
                            }
                        } else {
                            std::vector<double> filter_vec = rotateVector(f_rows[fn][fh], ow * stride_);
                            auto filter_row_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_vec);
                            auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[oh + fh - ENCRYPTED_HEIGHT_START], filter_row_plain);
    
                            temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);
                        }
                    }
    
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    std::vector<double> sum_vec(filter_n * output_w, 0);
                    sum_vec[fn * output_w + ow] = sum;
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);
    
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
    
                    // 线程私有变量，不会造成竞争
                    local_y_vec_ct.push_back(temp_res);
                }
            }
    
            // 合并线程结果到 y_vec_ct
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            y_vec_ct.insert(y_vec_ct.end(), local_y_vec_ct.begin(), local_y_vec_ct.end());
        }
    
        y_cts.push_back(CRYPTOCONTEXT->EvalAddMany(y_vec_ct));
    }

    #define Y_CTS_CHECK
    #ifdef Y_CTS_CHECK
    cout << "value check for encrypted parts" << endl;
    for(int i = 0; i < static_cast<int>(y_cts.size()); i++){
        Plaintext res;
        CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i], &res);
        cout << "y_cts[" << i << "]: " << res << endl;
    }
    #endif
    
}

bool isEncrypted_h(int val, int filter_size){
    int end_val = val + filter_size;

    bool overlap = (val <= ENCRYPTED_HEIGHT_END && end_val > ENCRYPTED_HEIGHT_START);
    return overlap;
}

// check if the output value will involve encrypted values
bool isEncrypted(int oh, int ow, int fh, int fw) {
    int end_h = oh + fh;
    int end_w = ow + fw;

    // 判断区域是否与加密区域有交集
    bool height_overlap = (oh <= ENCRYPTED_HEIGHT_END && end_h > ENCRYPTED_HEIGHT_START);
    bool width_overlap = (ow <= ENCRYPTED_WIDTH_END && end_w > ENCRYPTED_WIDTH_START);

    return height_overlap && width_overlap;
}










void GoldenConv2d(double3d& input, double3d& filters, int stride, int padding){
    // do the convolution
    int input_n = input.size();
    int filter_h = filters[0].size();
    int filter_w = filters[0][0].size();
    int output_n = filters.size();
    int output_h = (input[0].size() + 2 * padding - filters[0].size()) / stride + 1;
    int output_w = (input[0][0].size() + 2 * padding - filters[0][0].size()) / stride + 1;

    for (int n = 0; n < output_n; n++){
        for (int h = 0; h < output_h; h++){
            for (int w = 0; w < output_w; w++){
                double sum = 0;
                for (int i = 0; i < filter_h; i++){
                    for (int j = 0; j < filter_w; j++){
                        for (int c = 0; c < input_n; c++){
                            sum += input[c][h*stride + i][w*stride + j] * filters[n][i][j];
                        }
                    }
                }
                cout << sum << " ";
            }
            cout << endl;
        }
    }


}