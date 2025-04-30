
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
//#define DEBUG
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
    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int input_w = x_pts[0][0].size();
    int filter_n = filters_.size();
    int filter_h = filters_[0].size();
    int filter_w = filters_[0][0].size();
    int output_h = (input_h + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - filter_w) / stride_ + 1;
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
            for (int i = 0; i < input_n * input_w; i++){
                if (i%input_w < filter_w){
                    filter_row_vec.push_back(filters_[n][h][i%input_w]);
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

    double3d padded_input(input_n, double2d(input_h + 2 * padding_, vector<double>(input_w + 2 * padding_, 0)));


    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                padded_input[i][j + padding_][k + padding_] = x_pts[i][j][k];
            }
        }
    }

    for (int fn = 0; fn < filter_n; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < CURRENT_CHANNEL; c++){
                                sum += padded_input[c][oh*stride_ + fh][ow*stride_ + fw] * 
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
    // modify below based on padding 
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
        if (!intervalsOverlap(w, w + filter_w, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END) || w == output_w - 1){
            re_encrypt_end = w;
            break;
        }
    }
    std::cout << "re_encrypt_start: " << re_encrypt_start << std::endl;
    std::cout << "re_encrypt_end: " << re_encrypt_end << std::endl;

    for(int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for(int c = 0; c < input_n; c++){
            for(int w = 0; w < input_w; w++){
                if (isInRange(w, re_encrypt_start, re_encrypt_end)){
                    to_add.push_back(padded_input[c][h][w]);
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
        if (!isEncrypted_h(oh * stride_, filter_h, padding_)) {
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
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)) {
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
                    //auto start = std::chrono::high_resolution_clock::now();
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    //auto end = std::chrono::high_resolution_clock::now();
                    //std::cout << "EvalMult time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
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

    //#define Y_CTS_CHECK
    #ifdef Y_CTS_CHECK
    cout << "value check for encrypted parts" << endl;
    for(int i = 0; i < static_cast<int>(y_cts.size()); i++){
        Plaintext res;
        CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i], &res);
        cout << "y_cts[" << i << "]: " << res << endl;
    }
    #endif
    
}


void Conv2d_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    #ifdef DEBUG
    cout << "Conv2d_Partial forward" << endl;
    #endif

    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int padded_input_h = input_h + 2 * padding_;
    int input_w = x_pts[0][0].size();
    int padded_input_w = input_w + 2 * padding_;


    int filter_n = filters_.size();
    int filter_h = filters_[0].size();
    int filter_w = filters_[0][0].size();
    int output_h = (input_h + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - filter_w) / stride_ + 1;
    double3d f_rows;
    f_rows.resize(filter_n);
    y_cts.clear();


    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << filter_n << endl;
    cout << "f_h: " << filter_h << endl;
    cout << "f_w: " << filter_w << endl;
    #endif

    // first we do the unencrypted convolution
    y_pts.resize(filter_n);
    for (int i = 0; i < filter_n; i++){
        y_pts[i].resize(output_h);
        for (int j = 0; j < output_h; j++){
            y_pts[i][j].resize(output_w);
        }
    }
    double3d padded_input(input_n, double2d(padded_input_h, vector<double>(padded_input_w, 0)));
    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                padded_input[i][j + padding_][k + padding_] = x_pts[i][j][k];
            }
        }
    }

    for (int fn = 0; fn < filter_n; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < CURRENT_CHANNEL; c++){
                                sum += padded_input[c][oh*stride_ + fh][ow*stride_ + fw] * 
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

    // create filter vector to multiply with x_cts

    for (int n = 0; n < filter_n; n++){
        for (int h = 0; h < filter_h; h++){
            std::vector<double> filter_row_vec;
            for (int i = 0; i < padded_input_w; i++){
                if (i < filter_w){
                    filter_row_vec.push_back(filters_[n][h][i]);
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


    // then we do re-encryption and addition
    // regarding encrypted rows
    for(int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for(int c = 0; c < input_n; c++){
            for(int w = 0; w < input_w; w++){
                to_add.push_back(x_pts[c][h][w]);
            }
        }
        for (size_t i = 0; i < to_add.size(); i+= batch_size_){
            size_t end = std::min(i + batch_size_, to_add.size());
            std::vector<double> one_batch(to_add.begin() + i, to_add.begin() + end);
            auto to_add_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(one_batch);
            x_cts[h - ENCRYPTED_HEIGHT_START][i/batch_size_] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START][i/batch_size_], to_add_plain);
        }
    }
    
    //#define ADD_CTS_CHECK
    #ifdef ADD_CTS_CHECK
    std::cout << "check added cts:" << std::endl;
    for (int i = 0; i < static_cast<int>(x_cts.size()); i++){
        cout << "encrypted row: " << i << endl;
        for (int j = 0; j < static_cast<int>(x_cts[i].size()); j++){
            Plaintext plain;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, x_cts[i][j], &plain);
            std::cout << plain << std::endl;
        }
    }
    return;
    #endif

    int count = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (isEncrypted_h(oh * stride_, filter_h, padding_)){
            count++;
        }
    }
    y_cts.resize(count);

    int y_cts_idx = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (!isEncrypted_h(oh * stride_, filter_h, padding_)) {
            continue;
        }
        int output_2nd_dim = (filter_n * output_w + batch_size_ - 1)/batch_size_;
        types::vector2d<Ciphertext<DCRTPoly>> y_vec_ct;
        y_vec_ct.resize(output_2nd_dim);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            // 每个线程维护自己的临时 vector
            types::vector2d<Ciphertext<DCRTPoly>> local_y_vec_ct;
            local_y_vec_ct.resize(output_2nd_dim);
    
            #ifdef _OPENMP
            #pragma omp for collapse(2)
            #endif
            // each output channel
            for (int fn = 0; fn < filter_n; fn++) {
                // each output row 
                for (int ow = 0; ow < output_w; ow++) {
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)) {
                        continue;
                    }
    
                    std::vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    auto temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.secretKey, temp_plain);
                    double sum = 0;
    
                    std::vector<double> mask(batch_size_, 0);
                    mask[(fn * output_w + ow) % batch_size_] = 1;
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
    
                    for (int fh = 0; fh < filter_h; fh++) {
                        if (!isInRange(oh * stride_ + fh, ENCRYPTED_HEIGHT_START + padding_, ENCRYPTED_HEIGHT_END + padding_)) {
                            for (int fw = 0; fw < filter_w; fw++) {
                                for (int c = 0; c < input_n; c++) {
                                    sum += padded_input[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][fh][fw];
                                }
                            }
                        } else {
                            // consider more than one ciphertext for each row 
                            for (int cts_idx = 0; cts_idx < static_cast<int>(x_cts[oh + fh - ENCRYPTED_HEIGHT_START - padding_].size()); cts_idx++){
                                std::vector<double> filter_vec;
                                std::vector<double> filter_row = rotateVector(f_rows[fn][fh], ow * stride_);
                                std::vector<double> temp(filter_row.begin() + padding_, filter_row.end()- padding_);
                                for (int i = 0; i < input_n; i++){
                                    filter_vec.insert(filter_vec.end(), temp.begin(), temp.end());
                                }
                                filter_vec.assign(filter_vec.begin(), filter_vec.begin() + batch_size_);
                                auto filter_row_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_vec);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[oh + fh - ENCRYPTED_HEIGHT_START - padding_][cts_idx], filter_row_plain);
                                temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);

                                //std::cout << "oh: " << oh << ", ow:" << ow << ", fh:" << fh << endl; 
                                //std::cout << "filter_row_plain: " << filter_row_plain << endl;
                            }
                        }
                    }
    
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    std::vector<double> sum_vec(batch_size_, 0);
                    sum_vec[(fn * output_w+ ow) % batch_size_] = sum;
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);
    
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
    
                    // 线程私有变量，不会造成竞争
                    // get a single output value in cts
                    local_y_vec_ct[(fn * output_w + ow)/batch_size_].push_back(temp_res);
                }

            }
    
            // 合并线程结果到 y_vec_ct
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            for (size_t i = 0; i < y_vec_ct.size(); i++){
                y_vec_ct[i].insert(y_vec_ct[i].end(), local_y_vec_ct[i].begin(), local_y_vec_ct[i].end());
            }
            

        }
        //y_cts.push_back(CRYPTOCONTEXT->EvalAddMany(y_vec_ct));
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            //std::cout << "y_vec_ct size: " << y_vec_ct.size() << std::endl;
            size_t length = y_vec_ct.size();
            for (size_t i = 0; i < length; i ++){
                y_cts[y_cts_idx].push_back(CRYPTOCONTEXT->EvalAddMany(y_vec_ct[i]));
            }
            y_cts_idx++;
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
    #endif
    
}


Conv2dBN_P::Conv2dBN_P(
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
    batch_size_(batch_size),
    
    {
        CONSUMED_LEVEL++;
}

Conv2dBN_P::~Conv2dBN_P() {}

void Conv2dBN_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    #ifdef DEBUG
    cout << "Conv2d_Partial forward" << endl;
    #endif

    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int padded_input_h = input_h + 2 * padding_;
    int input_w = x_pts[0][0].size();
    int padded_input_w = input_w + 2 * padding_;


    int filter_n = filters_.size();
    int filter_h = filters_[0].size();
    int filter_w = filters_[0][0].size();
    int output_h = (input_h + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - filter_w) / stride_ + 1;
    double3d f_rows;
    f_rows.resize(filter_n);
    y_cts.clear();

    vector<double> bn_a(input_n, 0);
    vector<double> bn_b(input_n, 0);

    for (int i = 0; i < input_n; i++){
        bn_a[i] = gamma_[i] / sqrt(var_[i] + eps_);
        bn_b[i] = beta_[i] - gamma_[i] * mean_[i] / sqrt(var_[i] + eps_);
    }


    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << filter_n << endl;
    cout << "f_h: " << filter_h << endl;
    cout << "f_w: " << filter_w << endl;
    #endif

    // first we do the unencrypted convolution
    y_pts.resize(filter_n);
    for (int i = 0; i < filter_n; i++){
        y_pts[i].resize(output_h);
        for (int j = 0; j < output_h; j++){
            y_pts[i][j].resize(output_w);
        }
    }
    double3d padded_input(input_n, double2d(padded_input_h, vector<double>(padded_input_w, 0)));
    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                padded_input[i][j + padding_][k + padding_] = x_pts[i][j][k];
            }
        }
    }

    for (int fn = 0; fn < filter_n; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < CURRENT_CHANNEL; c++){
                                sum += padded_input[c][oh*stride_ + fh][ow*stride_ + fw] * 
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

    // create filter vector to multiply with x_cts

    for (int n = 0; n < filter_n; n++){
        for (int h = 0; h < filter_h; h++){
            std::vector<double> filter_row_vec;
            for (int i = 0; i < padded_input_w; i++){
                if (i < filter_w){
                    filter_row_vec.push_back(filters_[n][h][i]);
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


    // then we do re-encryption and addition
    // regarding encrypted rows
    for(int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for(int c = 0; c < input_n; c++){
            for(int w = 0; w < input_w; w++){
                to_add.push_back(x_pts[c][h][w]);
            }
        }
        for (size_t i = 0; i < to_add.size(); i+= batch_size_){
            size_t end = std::min(i + batch_size_, to_add.size());
            std::vector<double> one_batch(to_add.begin() + i, to_add.begin() + end);
            auto to_add_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(one_batch);
            x_cts[h - ENCRYPTED_HEIGHT_START][i/batch_size_] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START][i/batch_size_], to_add_plain);
        }
    }
    
    //#define ADD_CTS_CHECK
    #ifdef ADD_CTS_CHECK
    std::cout << "check added cts:" << std::endl;
    for (int i = 0; i < static_cast<int>(x_cts.size()); i++){
        cout << "encrypted row: " << i << endl;
        for (int j = 0; j < static_cast<int>(x_cts[i].size()); j++){
            Plaintext plain;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, x_cts[i][j], &plain);
            std::cout << plain << std::endl;
        }
    }
    return;
    #endif

    int count = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (isEncrypted_h(oh * stride_, filter_h, padding_)){
            count++;
        }
    }
    y_cts.resize(count);

    int y_cts_idx = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (!isEncrypted_h(oh * stride_, filter_h, padding_)) {
            continue;
        }
        int output_2nd_dim = (filter_n * output_w + batch_size_ - 1)/batch_size_;
        types::vector2d<Ciphertext<DCRTPoly>> y_vec_ct;
        y_vec_ct.resize(output_2nd_dim);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            // 每个线程维护自己的临时 vector
            types::vector2d<Ciphertext<DCRTPoly>> local_y_vec_ct;
            local_y_vec_ct.resize(output_2nd_dim);
    
            #ifdef _OPENMP
            #pragma omp for collapse(2)
            #endif
            // each output channel
            for (int fn = 0; fn < filter_n; fn++) {
                // each output row 
                for (int ow = 0; ow < output_w; ow++) {
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)) {
                        continue;
                    }
    
                    std::vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    auto temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.secretKey, temp_plain);
                    double sum = 0;
    
                    std::vector<double> mask(batch_size_, 0);
                    mask[(fn * output_w + ow) % batch_size_] = bn_a[fn];
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
    
                    for (int fh = 0; fh < filter_h; fh++) {
                        if (!isInRange(oh * stride_ + fh, ENCRYPTED_HEIGHT_START + padding_, ENCRYPTED_HEIGHT_END + padding_)) {
                            for (int fw = 0; fw < filter_w; fw++) {
                                for (int c = 0; c < input_n; c++) {
                                    sum += padded_input[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][fh][fw];
                                }
                            }
                        } else {
                            // consider more than one ciphertext for each row 
                            for (int cts_idx = 0; cts_idx < static_cast<int>(x_cts[oh + fh - ENCRYPTED_HEIGHT_START - padding_].size()); cts_idx++){
                                std::vector<double> filter_vec;
                                std::vector<double> filter_row = rotateVector(f_rows[fn][fh], ow * stride_);
                                std::vector<double> temp(filter_row.begin() + padding_, filter_row.end()- padding_);
                                for (int i = 0; i < input_n; i++){
                                    filter_vec.insert(filter_vec.end(), temp.begin(), temp.end());
                                }
                                filter_vec.assign(filter_vec.begin(), filter_vec.begin() + batch_size_);
                                auto filter_row_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_vec);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[oh + fh - ENCRYPTED_HEIGHT_START - padding_][cts_idx], filter_row_plain);
                                temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);

                                //std::cout << "oh: " << oh << ", ow:" << ow << ", fh:" << fh << endl; 
                                //std::cout << "filter_row_plain: " << filter_row_plain << endl;
                            }
                        }
                    }
    
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    std::vector<double> sum_vec(batch_size_, 0);
                    sum_vec[(fn * output_w+ ow) % batch_size_] = sum;
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);
    
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, bn_b[fn]);
    
                    // 线程私有变量，不会造成竞争
                    // get a single output value in cts
                    local_y_vec_ct[(fn * output_w + ow)/batch_size_].push_back(temp_res);
                }

            }
    
            // 合并线程结果到 y_vec_ct
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            for (size_t i = 0; i < y_vec_ct.size(); i++){
                y_vec_ct[i].insert(y_vec_ct[i].end(), local_y_vec_ct[i].begin(), local_y_vec_ct[i].end());
            }
            

        }
        //y_cts.push_back(CRYPTOCONTEXT->EvalAddMany(y_vec_ct));
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            //std::cout << "y_vec_ct size: " << y_vec_ct.size() << std::endl;
            size_t length = y_vec_ct.size();
            for (size_t i = 0; i < length; i ++){
                y_cts[y_cts_idx].push_back(CRYPTOCONTEXT->EvalAddMany(y_vec_ct[i]));
            }
            y_cts_idx++;
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
    #endif
    
}


bool isEncrypted_h(int val, int filter_size, int padding){
    int end_val = val + filter_size;

    bool overlap = (val <= ENCRYPTED_HEIGHT_END + padding && end_val > ENCRYPTED_HEIGHT_START + padding);
    return overlap;
}

// check if the output value will involve encrypted values
bool isEncrypted(int oh, int ow, int fh, int fw, int padding) {
    int end_h = oh + fh;
    int end_w = ow + fw;

    // 判断区域是否与加密区域有交集
    bool height_overlap = (oh <= ENCRYPTED_HEIGHT_END + padding && end_h > ENCRYPTED_HEIGHT_START + padding);
    bool width_overlap = (ow <= ENCRYPTED_WIDTH_END + padding && end_w > ENCRYPTED_WIDTH_START + padding);

    return height_overlap && width_overlap;
}










void GoldenConv2d(double3d& input, double3d& filters, int stride, int padding){
    // do the convolution
    int input_n = input.size();
    int input_h = input[0].size();
    int input_w = input[0][0].size();
    int filter_h = filters[0].size();
    int filter_w = filters[0][0].size();
    int output_n = filters.size();
    int output_h = (input[0].size() + 2 * padding - filters[0].size()) / stride + 1;
    int output_w = (input[0][0].size() + 2 * padding - filters[0][0].size()) / stride + 1;

    double3d padded_input(input_n, double2d(input_h + 2 * padding, vector<double>(input_w + 2 * padding, 0)));
    for (int i = 0; i < input_n; i++){
        for (int j = 0; j < input_h; j++){
            for (int k = 0; k < input_w; k++){
                padded_input[i][j + padding][k + padding] = input[i][j][k];
            }
        }
    }

    for (int n = 0; n < output_n; n++){
        for (int h = 0; h < output_h; h++){
            for (int w = 0; w < output_w; w++){
                double sum = 0;
                for (int i = 0; i < filter_h; i++){
                    for (int j = 0; j < filter_w; j++){
                        for (int c = 0; c < input_n; c++){
                            sum += padded_input[c][h*stride + i][w*stride + j] * filters[n][i][j];
                        }
                    }
                }
                cout << sum << " ";
            }
            cout << endl;
        }
        cout << endl;
    }


}