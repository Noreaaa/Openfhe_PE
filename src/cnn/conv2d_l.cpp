// deprecated file
#include "conv2d_l.hpp"

using std::vector;
using std::pair;
using types::double4d;
using types::double3d;
using types::double2d;
using std::vector;
using std::cout;
using std::endl;


Conv2dBN_P_L::Conv2dBN_P_L(
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
    next_pool_layer_type_(next_pool_layer_type)
    {
        CONSUMED_LEVEL++;
    }

Conv2dBN_P_L::~Conv2dBN_P_L() {}
// test layer type use parameter batch_size  RingDim = 8192
void Conv2dBN_P_L::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts){

    // the version of Conv2d BN used for ImageNet final 8 layers
    cout << layer_name_ << " forward check" << endl;

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
                    y_pts[fn][oh][ow] = sum * bn_a[fn] + bn_b[fn];
                }
            }
        }
    }

    // check value of y_pts
    //#define CHECK_Y_PTS
    #ifdef CHECK_Y_PTS
    std::cout << "y_pts:" << std::endl;

    print_3d(y_pts);
    return;
    #endif



    // do re-encryption and addition
    // ensure fixed batch size
    if (batch_size_ != 128 * 28){
        cout << "Warning: batch size is not 128 * 28." << endl;
    }
    cout << "Current batch size: " << batch_size_ << endl;

    for (int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; h++){
        std::vector<double> to_add;
        for  (int c = 0; c < input_n; c++){
            for (int w = 0; w < input_w; w++){
                to_add.push_back(x_pts[c][h][w]);
            }
        }

        for (size_t i = 0; i < to_add.size(); i+= batch_size_){
            size_t end = std::min(i + batch_size_, to_add.size());

            std::vector<double> one_batch(to_add.begin() + i, to_add.begin() + end);

            auto to_add_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(one_batch);
            x_cts[h - ENCRYPTED_HEIGHT_START][i / batch_size_] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START][i/batch_size_], to_add_plain);
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
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            count++;
        }
    }
    y_cts.resize(count);

    int output_2nd_dim = (out_channels * output_w + batch_size_ - 1)/batch_size_;

    int y_cts_idx = 0;
    for(int oh = 0; oh < output_h; oh++){
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

                    std::vector<double> temp_vec = {0};
                    auto temp_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(temp_vec);
                    auto temp_res = CRYPTOCONTEXT->Encrypt(KEYPAIR.secretKey, temp_plain);
                    double sum = 0;


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
                                auto filter_row_plain = Conv2dBN_P_L::GenPlainFilter(fn, fh, ow, input_w, cts_idx);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[encrypted_row_idx][cts_idx], filter_row_plain);
                                temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row);
                            }
                        }
                    }
                    std::vector<double> mask(batch_size_, 0);
                    std::vector<double> BN_b(batch_size_, 0);
                    std::vector<double> sum_vec(batch_size_, 0);
                    mask[(fn * output_w + ow) % batch_size_] = bn_a[fn];
                    BN_b[(fn * output_w + ow) % batch_size_] = bn_b[fn];
                    sum_vec[(fn * output_w + ow) % batch_size_] = sum;
                    auto mask_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(mask);
                    auto bn_b_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(BN_b);

                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, 4096);
                    auto sum_plain = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(sum_vec);

                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, bn_b_plain);


                    // get a single output value in cts
                    local_y_vec_ct[(fn * output_w + ow)/batch_size_].push_back(temp_res);

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



    std::cout << "Conv2dBN_P_L forward done." << std::endl;



}

Plaintext Conv2dBN_P_L::GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx){
    std::vector<double> filter_row;
    int padded_input_w = input_w + 2*padding_;
    int in_channels = filters_[0].size();
    int filter_w = filters_[0][0][0].size();
    assert(batch_size_ % in_channels == 0);

    int start_in_channels = cts_idx * batch_size_ / input_w;
    int end_in_channels = std::min((cts_idx + 1) * static_cast<int>(batch_size_) / input_w, in_channels);

    for (int i = start_in_channels; i < end_in_channels; i++){
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
