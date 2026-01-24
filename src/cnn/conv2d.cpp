
#include "conv2d.hpp"
#include <utility>
#include <cassert>
#include <cstdlib>
#include <algorithm>


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

Conv2d_C::Conv2d_C(
    PLayerType layer_type,
    std::string layer_name,
    types::double4d& filters,
    int stride,
    int input_height,
    int input_width,
    uint32_t batch_size
): Layer(PLayerType::CONV_2D_C, layer_name),
  filters_(filters),
  stride_(stride),
  input_height_(input_height),
  input_width_(input_width),
  batch_size_(batch_size) {
    CONSUMED_LEVEL++;
}

Conv2d_C::~Conv2d_C() {}



/**
 * need to handle strided convolution 
 * shape of cts 
 * row major order 
 * does not support padding 
 */
void Conv2d_C::forward(std::vector<Ciphertext<DCRTPoly>>& x_cts, 
std::vector<Ciphertext<DCRTPoly>>& y_cts){
    cout << layer_name_ << " forward" << endl;
    // initialize parameters
    int out_channels = filters_.size();
    int in_channels = filters_[0].size();
    int kernel_size = filters_[0][0].size();
    int input_height = input_height_;
    int input_width = input_width_;
    int output_height = (input_height - kernel_size) / stride_ + 1;
    int output_width = (input_width - kernel_size) / stride_ + 1;

    y_cts.clear();
    y_cts.resize(out_channels);

    // calculate anchor map
    std::vector<int> ROTATE_IDX;
    std::vector<std::pair<int,int>> anchor_map = generate_anchor(input_height, input_width, stride_, kernel_size);
    #pragma omp parallel for
    for (int oc = 0; oc < out_channels; oc++) {
        Ciphertext<DCRTPoly> local_sum; // to accumulate across ic
        bool first_ic = true;
    
        for (int ic = 0; ic < in_channels; ic++) {
            // construct filters
            types::vector2d<Plaintext> filter_vecs(kernel_size, std::vector<Plaintext>(kernel_size));
        
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    std::vector<double> filter_vec(input_height * input_width, 0);
                    for (size_t i = 0; i < anchor_map.size(); i++) {
                        int pos_h = anchor_map[i].first + kh;
                        int pos_w = anchor_map[i].second + kw;
                        int index = VALID_INDEX_MAP[pos_h][pos_w];
                        filter_vec[index] = filters_[oc][ic][kh][kw];
                    }
                    filter_vecs[kh][kw] = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_vec);
                }
            }
        
            // multiply with input feature map
            Ciphertext<DCRTPoly> conv_result;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    auto temp_result = CRYPTOCONTEXT->EvalMult(x_cts[ic], filter_vecs[kh][kw]);
                    if (kh == 0 && kw == 0) {
                        conv_result = temp_result;
                    } else {
                        int offset = VALID_INDEX_MAP[kh][kw] - VALID_INDEX_MAP[0][0];
                        std::cout << "offset: " << offset << std::endl;
                        if(offset != 0){
                            temp_result = CRYPTOCONTEXT->EvalRotate(temp_result, offset);
                        }
                        conv_result = CRYPTOCONTEXT->EvalAdd(conv_result, temp_result);
                    }
                }
            }
        
            // accumulate into local_sum
            if (first_ic) {
                local_sum = conv_result;
                first_ic = false;
            } else {
                local_sum = CRYPTOCONTEXT->EvalAdd(local_sum, conv_result);
            }
        }
    
        // write result for this oc
        y_cts[oc] = local_sum;
    }


    update_index_maps(output_height, output_width, stride_, kernel_size, false);



}
/**
 * update the valid index map 
 * find the valid value positions in the ciphertext
 * VALID_INDEX_MAP[0][0] = 1 the valid value of feature map [0][0] is at index 1 on cts
 * h w ---> index on cts
 */
void update_index_maps(int height, int width, int stride, int kernel_size, bool initial) {
    // Initialize index mapping
    if (initial) {
        int initial_height = height;
        int initial_width = width;
        VALID_INDEX_MAP.clear();
        VALID_INDEX_MAP.resize(initial_height);
        for (int h = 0; h < initial_height; h++) {
            VALID_INDEX_MAP[h].resize(initial_width);
        }
        for(int h = 0; h < initial_height; h++){
            for(int w = 0; w < initial_width; w++){
                VALID_INDEX_MAP[h][w] = {h * initial_width + w};
            }
        }
    }
    else {
        // based on stride update INDEX_MAP
        int output_height = height;
        int output_width = width;
        types::vector2d<int> old_index_map = VALID_INDEX_MAP;
        VALID_INDEX_MAP.clear();
        VALID_INDEX_MAP.resize(output_height);
        for (int h = 0; h < output_height; h++) {
            VALID_INDEX_MAP[h].resize(output_width);
        }
        for (int h = 0; h < output_height; h++) {
            for (int  w = 0; w < output_width; w++) {
                VALID_INDEX_MAP[h][w] = old_index_map[h * stride][w * stride];
            }
        }
        // Debug: Print the updated index map
        std::cout << "Updated VALID_INDEX_MAP:" << std::endl;
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                std::cout <<"height: " << h << " width: " << w << " value: " << VALID_INDEX_MAP[h][w] << " ";
            }
            std::cout << std::endl;
        }
    }

}

/**
 * generate anchor mapping table
 * mapping anchors [0] to input positions [index] on the valid index map
 * for example anchor_map[0] = 0,0 means the first anchor is stay on the index VALID_INDEX_MAP[0][0] 
 */
std::vector<std::pair<int, int>> generate_anchor(int height, int width, int stride, int kernel_size){
    
    std::vector<std::pair<int, int>> anchor_map;
    int output_width = (width - kernel_size) / stride + 1;
    int output_height = (height - kernel_size) / stride + 1;
    anchor_map.resize(output_height * output_width);
    
    for (int oh = 0; oh < output_height; oh++){
        for (int ow = 0; ow < output_width; ow++){
            anchor_map[oh * output_width + ow] = std::make_pair(oh * stride, ow * stride);
        }
    }

    return anchor_map;
}

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
    cout << layer_name_ << " forward" << endl;
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





Conv2dBN_P::Conv2dBN_P(
    PLayerType layer_type,
    std::string layer_name,
    types::double4d& filters,
    int stride,
    int padding,
    uint32_t batch_size,
    std::vector<double>& gamma,
    std::vector<double>& beta,
    std::vector<double>& mean,
    std::vector<double>& var,
    double epsilon,
    std::vector<double>& bias,
    PLayerType next_pool_layer_type)
    : Layer(PLayerType::CONV_2D, layer_name),
    filters_(filters),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size),
    gamma_(gamma),
    beta_(beta),
    mean_(mean),
    var_(var),
    epsilon_(epsilon),
    bias_(bias),
    next_pool_layer_type_(next_pool_layer_type)
    {
        CONSUMED_LEVEL++;
    }

Conv2dBN_P::~Conv2dBN_P() {}

void Conv2dBN_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    cout << layer_name_ << " forward" << endl;
    #ifdef DEBUG
    cout << "Conv2d_Partial forward" << endl;
    #endif

    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int padded_input_h = input_h + 2 * padding_;
    int input_w = x_pts[0][0].size();
    int padded_input_w = input_w + 2 * padding_;
    int out_channels = filters_.size();
    int in_channels = filters_[0].size();
    int filter_h = filters_[0][0].size();
    int filter_w = filters_[0][0][0].size();
    int output_h = (input_h + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - filter_w) / stride_ + 1;


    double3d f_rows;
    f_rows.resize(out_channels);
    y_cts.clear();
    vector<double> bn_a(out_channels, 0);
    vector<double> bn_b(out_channels, 0);
    for (int i = 0; i < out_channels; i++){
        bn_a[i] = gamma_[i] / sqrt(var_[i] + epsilon_);
        bn_b[i] = beta_[i] - gamma_[i] * mean_[i] / sqrt(var_[i] + epsilon_);
    }

    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << out_channels << endl;
    cout << "f_h: " << filter_h << endl;
    cout << "f_w: " << filter_w << endl;
    #endif
    y_pts.resize(out_channels);
    for (int i = 0; i < out_channels; i++){
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
    
    for (int fn = 0; fn < out_channels; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < in_channels; c++){
                                sum += padded_input[c][oh*stride_ + fh][ow*stride_ + fw] * 
                                filters_[fn][c][fh][fw];
                            }
                        }
                    }
                    y_pts[fn][oh][ow] = (sum + bias_[fn]) * bn_a[fn] + bn_b[fn];                                                                                           
                }
            }
        }
    }
    //#define CHECK_UNENCRYPTED
    #ifdef CHECK_UNENCRYPTED
    cout << "value check for unencrypted parts" << endl;
    print_3d(y_pts);
    #endif

    // then we do re-encryption and addition
    // regarding encrypted rows
    int in_channels_per_cts = batch_size_ / input_w;
    int values_per_cts = in_channels_per_cts * input_w;
    for(int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for(int c = 0; c < input_n; c++){
            for(int w = 0; w < input_w; w++){
                to_add.push_back(x_pts[c][h][w]);
            }
        }
        for (size_t i = 0; i < to_add.size(); i+= values_per_cts){
            size_t end = std::min(i + values_per_cts, to_add.size());
            std::vector<double> one_batch(to_add.begin() + i, to_add.begin() + end);
            auto to_add_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(one_batch);
            x_cts[h - ENCRYPTED_HEIGHT_START][i / values_per_cts] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START][i / values_per_cts], to_add_plain);
        }
    }
    
    //#define ADD_CTS_CHECK
    std::cout << "check point added cts" << std::endl; 
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
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            count++;
        }
    }
    y_cts.resize(count);
    bool interleave = false; 
    int out_channels_per_cts =  batch_size_ / output_w; 
    int output_2nd_dim = (out_channels + out_channels_per_cts - 1) / out_channels_per_cts;
    if (next_pool_layer_type_ == PLayerType::AVG_POOLING){
        int even_count = 0;
        for (int i = 0; i < output_w * out_channels; i++){
            if (i % 2 == 0){
                ++even_count;
            }
        }
        int even_batches = (even_count + batch_size_ - 1) / batch_size_;
        output_2nd_dim = even_batches * 2;
        interleave = true;
    }

    
    int y_cts_idx = 0;
    for (int oh = 0; oh < output_h; oh++) {
        // skip unencrypted rows 
        if (!isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            continue;
        }
        types::vector2d<Ciphertext<DCRTPoly>> y_vec_ct;
        y_vec_ct.resize(output_2nd_dim);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            types::vector2d<Ciphertext<DCRTPoly>> local_y_vec_ct;
            local_y_vec_ct.resize(output_2nd_dim);
    
            #ifdef _OPENMP
            #pragma omp for collapse(2)
            #endif
            // each output channel
            for (int fn = 0; fn < out_channels; fn++) {
                // each output row 
                for (int ow = 0; ow < output_w; ow++) {
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)) {
                        continue;
                    }
                    double sum = 0;
                    vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey, temp_plain);    
                    for (int fh = 0; fh < filter_h; fh++) {
                        if (!isInRange(oh * stride_ + fh, ENCRYPTED_HEIGHT_START + padding_, ENCRYPTED_HEIGHT_END + padding_)) {
                            for (int fw = 0; fw < filter_w; fw++) {
                                for (int c = 0; c < in_channels; c++) {
                                    sum += padded_input[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][c][fh][fw];
                                }
                            }
                        } else {
                            // consider more than one ciphertext for each row 
                            int encrypted_row_idx = oh * stride_ + fh - ENCRYPTED_HEIGHT_START - padding_;
                            if (encrypted_row_idx < 0 || encrypted_row_idx >= static_cast<int>(x_cts.size())) {
                                continue;
                            }
                            for (int cts_idx = 0; cts_idx < static_cast<int>(x_cts[encrypted_row_idx].size()); cts_idx++){
                                auto filter_row_plain = Conv2dBN_P::GenPlainFilter(fn, fh, ow, input_w, cts_idx, in_channels_per_cts);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[encrypted_row_idx][cts_idx], filter_row_plain);
                                if(RESCALE_REQUIRED == true){
                                    temp_res_per_row = CRYPTOCONTEXT->Rescale(temp_res_per_row);
                                }
                                temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);
                            }
                        }
                    }
                    std::vector<double> mask(batch_size_, 0);
                    std::vector<double> BN_b(batch_size_, 0);
                    std::vector<double> sum_vec(batch_size_, 0);
                    int channel_idx = fn % out_channels_per_cts;

                    // temporarily unchanged 
                    if (interleave){
                        // interleave the output for pooling layer
                        mask[((fn * output_w + ow)/2) % batch_size_] = bn_a[fn];
                        BN_b[((fn * output_w + ow)/2) % batch_size_] = bn_b[fn];
                        sum_vec[((fn * output_w + ow)/2) % batch_size_] = sum + bias_[fn];
                    }else{
                        mask[(channel_idx * output_w + ow) % batch_size_] = bn_a[fn];
                        BN_b[(channel_idx * output_w + ow) % batch_size_] = bn_b[fn];
                        sum_vec[(channel_idx * output_w + ow) % batch_size_] = sum + bias_[fn];
                    }
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
                    auto bn_b_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(BN_b);
    
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);
    
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    if (RESCALE_REQUIRED == true){
                        temp_res = CRYPTOCONTEXT->Rescale(temp_res);
                    }
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, bn_b_plain);
    
                    
                    // get a single output value in cts
                    if (interleave){
                        int idx = fn * output_w + ow; 
                        int batch_idx = idx / (2 * batch_size_);
                        int target_idx = (idx % 2 == 0) ? batch_idx * 2 : batch_idx * 2 + 1;
                        local_y_vec_ct[target_idx].push_back(temp_res);
                    }
                    else {
                        int target_idx = fn / out_channels_per_cts;
                        local_y_vec_ct[target_idx].push_back(temp_res);
                    }
                }

            }
    
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            for (size_t i = 0; i < y_vec_ct.size(); i++){
                y_vec_ct[i].insert(y_vec_ct[i].end(), local_y_vec_ct[i].begin(), local_y_vec_ct[i].end());
            }
        }
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

    // update the encrypted height and width 
    int start_h = -1, end_h = -1, start_w = -1, end_w = -1;
    for (int oh = 0; oh < output_h; oh++){
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            start_h = oh;
            break;
        }
    }
    for (int oh = output_h - 1; oh >= 0; oh--){
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            end_h = oh;
            break;
        }
    }
    for (int ow = 0; ow < output_w; ow++){
        if (isEncrypted_h(ow * stride_, filter_w, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            start_w = ow;
            break;
        }
    }
    for (int ow = output_w - 1; ow >= 0; ow--){
        if (isEncrypted_h(ow * stride_, filter_w, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            end_w = ow;
            break;
        }
    }
    if (start_h == -1 || end_h == -1 || start_w == -1 || end_w == -1){
        std::cout << "error: encrypted height or width not updated" << std::endl;
        return;
    }
    ENCRYPTED_HEIGHT_START = start_h;
    ENCRYPTED_HEIGHT_END = end_h;
    ENCRYPTED_WIDTH_START = start_w;
    ENCRYPTED_WIDTH_END = end_w;
    
    #define DEBUG
    #ifdef DEBUG
    cout << "after " << layer_name_ << " forward" << endl;
    cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << endl;
    cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << endl;
    cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << endl;
    cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << endl;
    #endif
}

void Conv2dBN_P::forward_C(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    cout << layer_name_ << " forward" << endl;
    cout << "data structure compact" << endl;
    #define DEBUG
    #ifdef DEBUG
    cout << "Conv2d_Partial forward" << endl;
    #endif
    int input_n = x_pts.size();
    int input_h = x_pts[0].size();
    int padded_input_h = input_h + 2 * padding_;
    int input_w = x_pts[0][0].size();
    int padded_input_w = input_w + 2 * padding_;
    int out_channels = filters_.size();
    int in_channels = filters_[0].size();
    int filter_h = filters_[0][0].size();
    int filter_w = filters_[0][0][0].size();
    int output_h = (input_h + 2 * padding_ - filter_h) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - filter_w) / stride_ + 1;
    double3d f_rows;
    f_rows.resize(out_channels);
    y_cts.clear();
    vector<double> bn_a(out_channels, 0);
    vector<double> bn_b(out_channels, 0);
    for (int i = 0; i < out_channels; i++){
        bn_a[i] = gamma_[i] / sqrt(var_[i] + epsilon_);
        bn_b[i] = beta_[i] - gamma_[i] * mean_[i] / sqrt(var_[i] + epsilon_);
    }

    #ifdef DEBUG
    cout << "output_h: " << output_h << endl;
    cout << "output_w: " << output_w << endl;
    cout << "f_n: " << out_channels << endl;
    cout << "f_h: " << filter_h << endl;
    cout << "f_w: " << filter_w << endl;
    #endif
    y_pts.resize(out_channels);
    for (int i = 0; i < out_channels; i++){
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
    for (int fn = 0; fn < out_channels; fn++){
        for (int oh = 0; oh < output_h; oh++){
            for (int ow = 0; ow < output_w; ow++){
                double sum = 0;
                // check if the output value will involve encrypted values
                if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)){
                    for (int fh = 0; fh < filter_h; fh++){
                        for (int fw = 0; fw < filter_w; fw++){
                            for (int c = 0; c < in_channels; c++){
                                sum += padded_input[c][oh*stride_ + fh][ow*stride_ + fw] * 
                                filters_[fn][c][fh][fw];
                            }
                        }
                    }
                    y_pts[fn][oh][ow] = (sum + bias_[fn]) * bn_a[fn] + bn_b[fn];                                                                                           
                }
            }
        }
    }
    //#define CHECK_UNENCRYPTED
    #ifdef CHECK_UNENCRYPTED
    cout << "value check for unencrypted parts" << endl;
    print_3d(y_pts);
    #endif
    int count = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            count++;
        }
    }
    y_cts.resize(count);
    count = 0;
    for (int ow = 0; ow < output_w; ow++){
        if(isEncrypted_h(ow * stride_, filter_w, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            count++;
        }
    }
    int output_encrypted_width = count;
    int out_channels_per_cts =  batch_size_ / output_encrypted_width; 
    REMAINING_SLOTS = out_channels_per_cts % batch_size_;
    int output_2nd_dim = (out_channels + out_channels_per_cts - 1) / out_channels_per_cts;

    int encrypted_width = ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1;
    int in_channels_per_cts = batch_size_ / encrypted_width;


    int y_cts_idx = 0;
    for (int oh = 0; oh < output_h; oh++) {
        // skip unencrypted rows 
        if (!isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            continue;
        }
        types::vector2d<Ciphertext<DCRTPoly>> y_vec_ct;
        y_vec_ct.resize(output_2nd_dim);

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            types::vector2d<Ciphertext<DCRTPoly>> local_y_vec_ct;
            local_y_vec_ct.resize(output_2nd_dim);
    
            #ifdef _OPENMP
            #pragma omp for collapse(2)
            #endif
            // each output channel
            for (int fn = 0; fn < out_channels; fn++) {
                // each output row 
                for (int ow = 0; ow < output_w; ow++) {
                    // skip if not including encrypted value
                    if (!isEncrypted(oh * stride_, ow * stride_, filter_h, filter_w, padding_)) {
                        continue;
                    }
                    double sum = 0;
                    vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey, temp_plain);    
                    for (int fh = 0; fh < filter_h; fh++) {
                        if (!isInRange(oh * stride_ + fh, ENCRYPTED_HEIGHT_START + padding_, ENCRYPTED_HEIGHT_END + padding_)) {
                            for (int fw = 0; fw < filter_w; fw++) {
                                for (int c = 0; c < in_channels; c++) {
                                    sum += padded_input[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][c][fh][fw];
                                }
                            }
                        } else {
                            // consider more than one ciphertext for each row 
                            int encrypted_height_idx = oh * stride_ + fh - ENCRYPTED_HEIGHT_START - padding_;
                            // sanity check
                            if (encrypted_height_idx < 0 || encrypted_height_idx >= static_cast<int>(x_cts.size())) {
                                continue;
                            }
                            // marking encrypted filter value range
                            std::vector<int> encrypted_filter_idxes;
                            for (int fw = 0; fw < filter_w; fw++){
                                int current_w = ow * stride_ + fw;
                                if (isInRange(current_w, ENCRYPTED_WIDTH_START + padding_, ENCRYPTED_WIDTH_END + padding_)){
                                    encrypted_filter_idxes.push_back(fw);
                                }
                                else {
                                    for (int c = 0; c < in_channels; c++) {
                                        sum += padded_input[c][oh * stride_ + fh][ow * stride_ + fw] * filters_[fn][c][fh][fw];
                                    }
                                }
                            }
                            // encrypted multiplication
                            for (int cts_idx = 0; cts_idx < static_cast<int>(x_cts[encrypted_height_idx].size()); cts_idx++){
                                auto filter_row_plain = Conv2dBN_P::GenPlainFilter_C(fn, fh, cts_idx, in_channels_per_cts, encrypted_filter_idxes, ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[encrypted_height_idx][cts_idx], filter_row_plain);
                                temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);
                            }
                        }
                    }
                    std::vector<double> mask(batch_size_, 0);
                    std::vector<double> BN_b(batch_size_, 0);
                    std::vector<double> sum_vec(batch_size_, 0);
                    int channel_idx = fn % out_channels_per_cts;
                    mask[(channel_idx * output_encrypted_width + ow) % batch_size_] = bn_a[fn];
                    BN_b[(channel_idx * output_encrypted_width + ow) % batch_size_] = bn_b[fn];
                    sum_vec[(channel_idx * output_encrypted_width + ow) % batch_size_] = sum + bias_[fn];
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
                    auto bn_b_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(BN_b);
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, batch_size_);
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    if (RESCALE_REQUIRED == true){
                        temp_res = CRYPTOCONTEXT->Rescale(temp_res);
                    }
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, bn_b_plain);
                    // get a single output value in cts
                    int target_idx = fn / out_channels_per_cts;
                    local_y_vec_ct[target_idx].push_back(temp_res);
                }
            }
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            for (size_t i = 0; i < y_vec_ct.size(); i++){
                y_vec_ct[i].insert(y_vec_ct[i].end(), local_y_vec_ct[i].begin(), local_y_vec_ct[i].end());
            }
        }
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
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



    // update the encrypted height and width 
    int start_h = -1, end_h = -1, start_w = -1, end_w = -1;
    for (int oh = 0; oh < output_h; oh++){
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            start_h = oh;
            break;
        }
    }
    for (int oh = output_h - 1; oh >= 0; oh--){
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            end_h = oh;
            break;
        }
    }
    for (int ow = 0; ow < output_w; ow++){
        if (isEncrypted_h(ow * stride_, filter_w, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            start_w = ow;
            break;
        }
    }
    for (int ow = output_w - 1; ow >= 0; ow--){
        if (isEncrypted_h(ow * stride_, filter_w, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            end_w = ow;
            break;
        }
    }
    if (start_h == -1 || end_h == -1 || start_w == -1 || end_w == -1){
        std::cout << "error: encrypted height or width not updated" << std::endl;
        return;
    }
    ENCRYPTED_HEIGHT_START = start_h;
    ENCRYPTED_HEIGHT_END = end_h;
    ENCRYPTED_WIDTH_START = start_w;
    ENCRYPTED_WIDTH_END = end_w;
    
    #define DEBUG
    #ifdef DEBUG
    cout << "after " << layer_name_ << " forward" << endl;
    cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << endl;
    cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << endl;
    cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << endl;
    cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << endl;
    #endif
}



Plaintext Conv2dBN_P::GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx, int out_channels_per_cts) {
    //cts_idx == dim_2 of cts

    std::vector<double> filter_row; 
    int padded_input_w = input_w + 2 * padding_;
    //int out_channels = filters_.size();
    int in_channels = filters_[0].size();

    int filter_w = filters_[0][0][0].size();
    // assure the batch size is multiple of input width
    //assert(batch_size_ % input_w == 0);
    int start_in_channels = out_channels_per_cts * cts_idx;
    int end_in_channels = out_channels_per_cts * (cts_idx + 1);

    for (int i = start_in_channels; i < end_in_channels && i < in_channels; i++){
        vector<double> filter_row_per_channel;
        for (int j = 0; j < padded_input_w; j++){
            if (j < filter_w){
                filter_row_per_channel.push_back(filters_[out_channel][i][height][j]);
            }else{
                filter_row_per_channel.push_back(0);
            }
        }
        filter_row_per_channel = rotateVector(filter_row_per_channel, 
            output_width_idx * stride_);
        filter_row.insert(filter_row.end(), filter_row_per_channel.begin() + padding_,
            filter_row_per_channel.end() - padding_);
    }

    return CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_row);

}

Plaintext Conv2dBN_P::GenPlainFilter_C(int out_channel, int height, int cts_idx, int in_channels_per_cts, vector<int> encrypted_filter_idxes, int encrypted_width) {

    std::vector<double> filter_row; 
    //int out_channels = filters_.size();
    int in_channels = filters_[0].size();

    // assure the batch size is multiple of input width
    //assert(batch_size_ % input_w == 0);
    int start_in_channels = in_channels_per_cts * cts_idx;
    int end_in_channels = in_channels_per_cts * (cts_idx + 1);

    for (int i = start_in_channels; i < end_in_channels && i < in_channels; i++){
        vector<double> filter_row_per_channel;
        for (size_t j = 0; j < encrypted_filter_idxes.size(); j++){
            
            filter_row_per_channel.push_back(filters_[out_channel][i][height][encrypted_filter_idxes[j]]);
        }
        filter_row.insert(filter_row.end(), filter_row_per_channel.begin(), filter_row_per_channel.end());
    }

    return CRYPTOCONTEXT->MakeCKKSPackedPlaintext(filter_row);

}

double3d GoldenConv2d(double3d input, double4d filters, vector<double> bias, int stride, int padding) {
    int in_channels = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();

    int out_channels = filters.size();
    int kernel_h = filters[0][0].size();
    int kernel_w = filters[0][0][0].size();

    // Padding input
    int padded_h = in_height + 2 * padding;
    int padded_w = in_width + 2 * padding;
    double3d padded_input(in_channels, vector<vector<double>>(padded_h, vector<double>(padded_w, 0.0)));

    for (int c = 0; c < in_channels; ++c)
        for (int h = 0; h < in_height; ++h)
            for (int w = 0; w < in_width; ++w)
                padded_input[c][h + padding][w + padding] = input[c][h][w];

    // Calculate output dimensions
    int out_h = (padded_h - kernel_h) / stride + 1;
    int out_w = (padded_w - kernel_w) / stride + 1;

    // Output tensor [out_channels][out_h][out_w]
    double3d output(out_channels, vector<vector<double>>(out_h, vector<double>(out_w, 0.0)));

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = bias[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum += padded_input[ic][ih][iw] * filters[oc][ic][kh][kw];
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }
    return output;
}

void GoldenBN(double3d& input, vector<double>& gamma, vector<double> beta, vector<double> running_mean, vector<double> running_var,
    double epsilon) {
        int channels = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
    

    
        for (int c = 0; c < channels; ++c) {
            double g = gamma[c];
            double b = beta[c];
            double mean = running_mean[c];
            double var = running_var[c];
            double eps = epsilon;
            double denom = std::sqrt(var + eps);
    
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    double norm = (input[c][h][w] - mean) / denom;
                    input[c][h][w] = g * norm + b;
                }
            }
        }
}


void update_Encrypted_Region(std::string layer_name,int output_h, int output_w, int filter, int padding, int stride){
    int start_h = -1, end_h = -1, start_w = -1, end_w = -1;
    for (int oh = 0; oh < output_h; oh++){
        if (isEncrypted_h(oh * stride, filter, padding, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            start_h = oh;
            break;
        }
    }
    for (int oh = output_h - 1; oh >= 0; oh--){
        if (isEncrypted_h(oh * stride, filter, padding, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            end_h = oh;
            break;
        }
    }
    for (int ow = 0; ow < output_w; ow++){
        if (isEncrypted_h(ow * stride, filter, padding, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            start_w = ow;
            break;
        }
    }
    for (int ow = output_w - 1; ow >= 0; ow--){
        if (isEncrypted_h(ow * stride, filter, padding, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            end_w = ow;
            break;
        }
    }
    if (start_h == -1 || end_h == -1 || start_w == -1 || end_w == -1){
        std::cout << "error: encrypted height or width not updated" << std::endl;
        return;
    }
    ENCRYPTED_HEIGHT_START = start_h;
    ENCRYPTED_HEIGHT_END = end_h;
    ENCRYPTED_WIDTH_START = start_w;
    ENCRYPTED_WIDTH_END = end_w;
    
    #define DEBUG
    #ifdef DEBUG
    //cout << "after " << layer_name << " forward" << endl;
    //cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << endl;
    //cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << endl;
    //cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << endl;
    //cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << endl;
    //cout << "new encrypted region" << ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1 << " x " << ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1 << endl;
    //cout << "encrypted new ratio: " << double((ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1) * (ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1)) /
    //double(output_h * output_w) << endl;
    cout << double((ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1) * (ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1)) / 
    double(output_h * output_w) << endl;
    #endif
}


