#include "sparse_conv2d.hpp"
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <set>
#include <functional>

SparseConv2d::SparseConv2d(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double3d filters, std::vector<double> biases, int stride, int padding)
    : Layer(layer_type, std::move(layer_name)),
      ring_dim_(ring_dim),
      packing_slots_(packing_slots),
      filters_(std::move(filters)),
      biases_(std::move(biases)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

void SparseConv2d::forward(std::vector<Ciphertext<DCRTPoly>>&, std::vector<Ciphertext<DCRTPoly>>&) {
    // TODO: implement with period packing_slots_ in ring ring_dim_.
    throw std::logic_error("SparseConv2d::forward is not implemented");
}

SparseConv2d_C::SparseConv2d_C(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double4d filters, int stride, int input_height, int input_width)
    : Layer(layer_type, std::move(layer_name)),
      ring_dim_(ring_dim),
      packing_slots_(packing_slots),
      filters_(std::move(filters)),
      stride_(std::move(stride)),
      input_height_(std::move(input_height)),
      input_width_(std::move(input_width)) {}

void SparseConv2d_C::forward(std::vector<Ciphertext<DCRTPoly>>&, std::vector<Ciphertext<DCRTPoly>>&) {
    // TODO: implement with period packing_slots_ in ring ring_dim_.
    throw std::logic_error("SparseConv2d_C::forward is not implemented");
}

SparseConv2d_P::SparseConv2d_P(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double3d filters, std::vector<double> biases, int stride, int padding, PLayerType next_pool_layer_type)
    : Layer(layer_type, std::move(layer_name)),
      ring_dim_(ring_dim),
      packing_slots_(packing_slots),
      filters_(std::move(filters)),
      biases_(std::move(biases)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      next_pool_layer_type_(std::move(next_pool_layer_type)) {}

void SparseConv2d_P::forward(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) {
    // TODO: implement with period packing_slots_ in ring ring_dim_.
    throw std::logic_error("SparseConv2d_P::forward is not implemented");
}

SparseConv2dBN_P::SparseConv2dBN_P(PLayerType layer_type, std::string layer_name, uint32_t ring_dim, uint32_t packing_slots, types::double4d filters, int stride, int padding, std::vector<double> gamma, std::vector<double> beta, std::vector<double> mean, std::vector<double> var, double epsilon, std::vector<double> bias, PLayerType next_pool_layer_type)
    : Layer(layer_type, std::move(layer_name)),
      ring_dim_(ring_dim),
      packing_slots_(packing_slots),
      filters_(std::move(filters)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      gamma_(std::move(gamma)),
      beta_(std::move(beta)),
      mean_(std::move(mean)),
      var_(std::move(var)),
      epsilon_(std::move(epsilon)),
      bias_(std::move(bias)),
      next_pool_layer_type_(std::move(next_pool_layer_type)) { CONSUMED_LEVEL += 2; }

void SparseConv2dBN_P::forward_reference(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    ValidateInput(x_cts, x_pts);
    std::cout << layer_name_ << " forward" << std::endl;
    #ifdef DEBUG
    std::cout << "Conv2d_Partial forward" << std::endl;
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


    y_cts.clear();
    vector<double> bn_a(out_channels, 0);
    vector<double> bn_b(out_channels, 0);
    for (int i = 0; i < out_channels; i++){
        bn_a[i] = gamma_[i] / sqrt(var_[i] + epsilon_);
        bn_b[i] = beta_[i] - gamma_[i] * mean_[i] / sqrt(var_[i] + epsilon_);
    }

    #ifdef DEBUG
    std::cout << "output_h: " << output_h << std::endl;
    std::cout << "output_w: " << output_w << std::endl;
    std::cout << "f_n: " << out_channels << std::endl;
    std::cout << "f_h: " << filter_h << std::endl;
    std::cout << "f_w: " << filter_w << std::endl;
    #endif
    y_pts.assign(out_channels, types::double2d(output_h, vector<double>(output_w, 0.0)));
    double3d padded_input(input_n, types::double2d(padded_input_h, vector<double>(padded_input_w, 0)));
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
    std::cout << "value check for unencrypted parts" << std::endl;
    print_3d(y_pts);
    #endif

    // Complete encrypted rows by adding their public entries.
    // regarding encrypted rows
    int in_channels_per_cts = packing_slots_ / input_w;
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
            auto to_add_plain = Encode(one_batch);
            x_cts[h - ENCRYPTED_HEIGHT_START][i / values_per_cts] = CRYPTOCONTEXT->EvalAdd(x_cts[h - ENCRYPTED_HEIGHT_START][i / values_per_cts], to_add_plain);
        }
    }
    

    int count = 0;
    for (int oh = 0; oh < output_h; oh++) {
        if (isEncrypted_h(oh * stride_, filter_h, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)) {
            count++;
        }
    }
    y_cts.resize(count);
    bool interleave = false; 
    int out_channels_per_cts =  packing_slots_ / output_w; 
    int output_2nd_dim = (out_channels + out_channels_per_cts - 1) / out_channels_per_cts;
    if (next_pool_layer_type_ == PLayerType::AVG_POOLING){
        int even_count = 0;
        for (int i = 0; i < output_w * out_channels; i++){
            if (i % 2 == 0){
                ++even_count;
            }
        }
        int even_batches = (even_count + packing_slots_ - 1) / packing_slots_;
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
                    Ciphertext<DCRTPoly> temp_res;
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
                                auto filter_row_plain = SparseConv2dBN_P::GenPlainFilter(fn, fh, ow, input_w, cts_idx, in_channels_per_cts);
                                auto temp_res_per_row = CRYPTOCONTEXT->EvalMult(x_cts[encrypted_row_idx][cts_idx], filter_row_plain);
                                if(RESCALE_REQUIRED == true){
                                    temp_res_per_row = CRYPTOCONTEXT->Rescale(temp_res_per_row);
                                }
                                temp_res = temp_res ? CRYPTOCONTEXT->EvalAdd(temp_res, temp_res_per_row) : temp_res_per_row;
                            }
                        }
                    }
                    std::vector<double> mask(packing_slots_, 0);
                    std::vector<double> BN_b(packing_slots_, 0);
                    std::vector<double> sum_vec(packing_slots_, 0);
                    int channel_idx = fn % out_channels_per_cts;

                    // temporarily unchanged 
                    if (interleave){
                        // interleave the output for pooling layer
                        mask[((fn * output_w + ow)/2) % packing_slots_] = bn_a[fn];
                        BN_b[((fn * output_w + ow)/2) % packing_slots_] = bn_b[fn];
                        sum_vec[((fn * output_w + ow)/2) % packing_slots_] = sum + bias_[fn];
                    }else{
                        mask[(channel_idx * output_w + ow) % packing_slots_] = bn_a[fn];
                        BN_b[(channel_idx * output_w + ow) % packing_slots_] = bn_b[fn];
                        sum_vec[(channel_idx * output_w + ow) % packing_slots_] = sum + bias_[fn];
                    }
                    auto mask_plain = Encode(mask);
                    auto bn_b_plain = Encode(BN_b);
    
                    temp_res = CRYPTOCONTEXT->EvalSum(temp_res, packing_slots_);
                    auto sum_plain = Encode(sum_vec);
    
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, sum_plain);
                    temp_res = CRYPTOCONTEXT->EvalMult(temp_res, mask_plain);
                    if (RESCALE_REQUIRED == true){
                        temp_res = CRYPTOCONTEXT->Rescale(temp_res);
                    }
                    temp_res = CRYPTOCONTEXT->EvalAdd(temp_res, bn_b_plain);
    
                    
                    // get a single output value in cts
                    if (interleave){
                        int idx = fn * output_w + ow; 
                        int batch_idx = idx / (2 * packing_slots_);
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
                y_cts[y_cts_idx].push_back((y_vec_ct[i].empty() ? CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey, Encode(vector<double>(packing_slots_, 0.0))) : CRYPTOCONTEXT->EvalAddMany(y_vec_ct[i])));
            }
            y_cts_idx++;
        }
    }
    
    

    //#define Y_CTS_CHECK
    #ifdef Y_CTS_CHECK
    std::cout << "value check for encrypted parts" << std::endl;
    for(int i = 0; i < static_cast<int>(y_cts.size()); i++){
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            Plaintext res;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &res);
            std::cout << "y_cts[" << i << "][" << j << "]: " << res << std::endl;
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
    
    #ifdef DEBUG
    std::cout << "after " << layer_name_ << " forward" << std::endl;
    std::cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << std::endl;
    std::cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << std::endl;
    std::cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << std::endl;
    std::cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << std::endl;
    #endif
}


Plaintext SparseConv2dBN_P::GenPlainFilter(int out_channel, int height, int output_width_idx, int input_w, int cts_idx, int out_channels_per_cts) {
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

    return Encode(filter_row);

}


void SparseConv2dBN_P::forward_C(types::vector2d<Ciphertext<DCRTPoly>>&, double3d&, types::vector2d<Ciphertext<DCRTPoly>>&, double3d&) {
    // TODO: implement with period packing_slots_ in ring ring_dim_.
    throw std::logic_error("SparseConv2dBN_P::forward_C is not implemented");
}

Plaintext SparseConv2dBN_P::Encode(const std::vector<double>& values) const {
    return CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, packing_slots_);
}

void SparseConv2dBN_P::ValidateInput(const types::vector2d<Ciphertext<DCRTPoly>>& cts,
                                    const double3d& pts) const {
    if (!CRYPTOCONTEXT || ring_dim_ != CRYPTOCONTEXT->GetRingDimension() ||
        !packing_slots_ || (packing_slots_ & (packing_slots_ - 1)) || packing_slots_ > ring_dim_ / 2)
        throw std::invalid_argument("SparseConv2dBN_P: invalid ring/slot configuration");
    if (pts.empty() || pts[0].empty() || pts[0][0].empty() || filters_.empty() ||
        filters_[0].empty() || filters_[0][0].empty() || filters_[0][0][0].empty() || stride_ <= 0 || padding_ < 0)
        throw std::invalid_argument("SparseConv2dBN_P: invalid tensor or convolution parameters");
    const size_t h = pts[0].size(), w = pts[0][0].size(), channels = pts.size();
    if (h + 2 * padding_ < filters_[0][0].size() || w + 2 * padding_ < filters_[0][0][0].size())
        throw std::invalid_argument("SparseConv2dBN_P: kernel exceeds padded input");
    const size_t outputWidth = (w + 2 * padding_ - filters_[0][0][0].size()) / stride_ + 1;
    if (packing_slots_ < w || packing_slots_ < outputWidth)
        throw std::invalid_argument("SparseConv2dBN_P: packing_slots must hold a complete input/output row");
    if (next_pool_layer_type_ == AVG_POOLING && outputWidth % 2 != 0)
        throw std::invalid_argument("SparseConv2dBN_P: interleaved pooling requires even output width");
    for (const auto& channel : pts) {
        if (channel.size() != h) throw std::invalid_argument("SparseConv2dBN_P: inconsistent height");
        for (const auto& row : channel)
            if (row.size() != w) throw std::invalid_argument("SparseConv2dBN_P: inconsistent width");
    }
    for (const auto& filter : filters_) {
        if (filter.size() != channels) throw std::invalid_argument("SparseConv2dBN_P: channel mismatch");
        for (const auto& channel : filter) {
            if (channel.size() != filters_[0][0].size()) throw std::invalid_argument("SparseConv2dBN_P: kernel height mismatch");
            for (const auto& row : channel)
                if (row.size() != filters_[0][0][0].size()) throw std::invalid_argument("SparseConv2dBN_P: kernel width mismatch");
        }
    }
    const size_t oc = filters_.size();
    if (gamma_.size() != oc || beta_.size() != oc || mean_.size() != oc || var_.size() != oc || bias_.size() != oc)
        throw std::invalid_argument("SparseConv2dBN_P: BN/bias channel mismatch");
    for (double v : var_) if (!(v + epsilon_ > 0)) throw std::invalid_argument("SparseConv2dBN_P: invalid BN variance");
    if (ENCRYPTED_HEIGHT_START < 0 || ENCRYPTED_HEIGHT_END < ENCRYPTED_HEIGHT_START || ENCRYPTED_HEIGHT_END >= static_cast<int>(h) ||
        ENCRYPTED_WIDTH_START < 0 || ENCRYPTED_WIDTH_END < ENCRYPTED_WIDTH_START || ENCRYPTED_WIDTH_END >= static_cast<int>(w))
        throw std::invalid_argument("SparseConv2dBN_P: invalid encrypted region");
    const size_t blocks = (channels + packing_slots_ / w - 1) / (packing_slots_ / w);
    if (cts.size() != static_cast<size_t>(ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1))
        throw std::invalid_argument("SparseConv2dBN_P: ciphertext row mismatch");
    for (const auto& row : cts) {
        if (row.size() != blocks) throw std::invalid_argument("SparseConv2dBN_P: ciphertext block mismatch");
        for (const auto& ct : row)
            if (!ct || ct->GetSlots() != packing_slots_) throw std::invalid_argument("SparseConv2dBN_P: ciphertext slot mismatch");
    }
}


void SparseConv2dBN_P::EnableFastPath(uint32_t width, bool generate_keys) {
    if (!width || width > packing_slots_ || filters_.empty() || filters_[0].empty() ||
        filters_[0][0].empty() || filters_[0][0][0].empty())
        throw std::invalid_argument("fast convolution: invalid width or filters");
    if (generate_keys && (!CRYPTOCONTEXT || !KEYPAIR.secretKey))
        throw std::invalid_argument("fast convolution: initialize context and keys first");
    if (generate_keys) {
        std::set<int32_t> indices;
        const int per = packing_slots_ / width;
        const int outputWidth = (width + 2*padding_ - filters_[0][0][0].size()) / stride_ + 1;
        const int outputPer = packing_slots_ / outputWidth;
        for (int k = 0; k < static_cast<int>(filters_[0][0][0].size()); ++k)
            if (k != padding_) indices.insert(k - padding_);
        for (int c = 1; c < std::min(outputPer, static_cast<int>(filters_.size())); ++c)
            indices.insert(-c * outputWidth);
        if (stride_ > 1) for (int ow=1;ow<outputWidth;++ow)
            indices.insert(ow*(stride_-1));
        for (int n = per; n > 1; n /= 2) {
            indices.insert((n / 2) * width);
            if (n % 2) indices.insert((n - 1) * width);
        }
        if (!indices.empty()) CRYPTOCONTEXT->EvalRotateKeyGen(KEYPAIR.secretKey,
            std::vector<int32_t>(indices.begin(), indices.end()));
    }
    fast_width_ = width;
}

void SparseConv2dBN_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x, double3d& pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y, double3d& out) {
    ValidateInput(x, pts);
    const int kw = filters_[0][0][0].size();
    if (!RESCALE_REQUIRED && fast_width_ == pts[0][0].size() && (stride_ == 1 || stride_ == 2) && 2 * padding_ == kw - 1 &&
        next_pool_layer_type_ != AVG_POOLING) forward_fast(x, pts, y, out);
    else forward_reference(x, pts, y, out);
}

void SparseConv2dBN_P::forward_fast(types::vector2d<Ciphertext<DCRTPoly>>& x, double3d& pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y, double3d& out) {
    const int ci = pts.size(), h = pts[0].size(), w = pts[0][0].size();
    const int co = filters_.size(), kh = filters_[0][0].size(), kw = filters_[0][0][0].size();
    const int ohCount = (h + 2 * padding_ - kh) / stride_ + 1;
    const int outputWidth = (w + 2 * padding_ - kw) / stride_ + 1;
    const int outputPer = packing_slots_ / outputWidth;
    const int per = packing_slots_ / w, blocks = (co + outputPer - 1) / outputPer;
    const int inputBlocks = (ci + per - 1) / per;
    int hs = -1, he = -1, ws = -1, we = -1;
    for (int i=0;i<ohCount;++i) if (isEncrypted_h(i*stride_,kh,padding_,ENCRYPTED_HEIGHT_START,ENCRYPTED_HEIGHT_END)) {
        if (hs < 0) hs = i;
        he = i;
    }
    for (int i=0;i<outputWidth;++i) if (isEncrypted_h(i*stride_,kw,padding_,ENCRYPTED_WIDTH_START,ENCRYPTED_WIDTH_END)) {
        if (ws < 0) ws = i;
        we = i;
    }
    if (hs < 0 || ws < 0) throw std::invalid_argument("fast convolution: encrypted region has no output");
    const int rows = he - hs + 1;
    std::vector<double> a(co), b(co);
    for (int c=0;c<co;++c) {
        a[c] = gamma_[c] / std::sqrt(var_[c] + epsilon_);
        b[c] = (bias_[c] - mean_[c]) * a[c] + beta_[c];
    }
    // Public values are evaluated independently. Input ciphertexts keep only private entries.
    out.assign(co, types::double2d(ohCount, std::vector<double>(outputWidth, 0)));
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int fn=0;fn<co;++fn) for (int oh=0;oh<ohCount;++oh) for (int ow=0;ow<outputWidth;++ow) {
        double sum = 0;
        for (int c=0;c<ci;++c) for (int fh=0;fh<kh;++fh) {
            const int ih = oh*stride_ + fh - padding_;
            if (ih < 0 || ih >= h) continue;
            for (int fw=0;fw<kw;++fw) {
                const int iw = ow*stride_ + fw - padding_;
                if (iw >= 0 && iw < w) sum += pts[c][ih][iw] * filters_[fn][c][fh][fw];
            }
        }
        out[fn][oh][ow] = sum * a[fn] + b[fn];
    }
    // Reuse spatial rotations for all output channels and output rows.
    types::vector2d<Ciphertext<DCRTPoly>> rotated(x.size() * inputBlocks,
        std::vector<Ciphertext<DCRTPoly>>(kw));
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t task=0;task<x.size()*inputBlocks*kw;++task) {
        const int fw = task % kw;
        const size_t index = task / kw;
        auto ct = x[index / inputBlocks][index % inputBlocks];
        rotated[index][fw] = fw == padding_ ? ct : CRYPTOCONTEXT->EvalRotate(ct, fw - padding_);
    }
    std::vector<double> select(packing_slots_, 0);
    for (int ow=ws;ow<=we;++ow) select[ow*stride_] = 1;
    auto selector = Encode(select);
    std::vector<Plaintext> compactSelectors;
    if (stride_ > 1) for (int ow=ws;ow<=we;++ow) {
        std::vector<double> one(packing_slots_, 0);
        one[ow*stride_] = 1;
        compactSelectors.push_back(Encode(one));
    }
    y.assign(rows, std::vector<Ciphertext<DCRTPoly>>(blocks));
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int fn=0;fn<co;++fn) {
        std::vector<Ciphertext<DCRTPoly>> accum(rows);
        for (int fh=0;fh<kh;++fh) for (int fw=0;fw<kw;++fw) for (int block=0;block<inputBlocks;++block) {
            std::vector<double> coefficients(packing_slots_, 0);
            for (int c=0;c<per && block*per+c<ci;++c) {
                const double coefficient = filters_[fn][block*per+c][fh][fw] * a[fn];
                for (int ow=ws;ow<=we;++ow)
                    if (ow*stride_+fw-padding_ >= 0 && ow*stride_+fw-padding_ < w)
                        coefficients[c*w+ow*stride_] = coefficient;
            }
            auto plain = Encode(coefficients);
            for (int row=0;row<rows;++row) {
                const int ih = (hs + row)*stride_ + fh - padding_;
                if (ih < ENCRYPTED_HEIGHT_START || ih > ENCRYPTED_HEIGHT_END) continue;
                const int index = (ih - ENCRYPTED_HEIGHT_START) * inputBlocks + block;
                auto product = CRYPTOCONTEXT->EvalMult(rotated[index][fw], plain);
                if (RESCALE_REQUIRED) product = CRYPTOCONTEXT->Rescale(product);
                accum[row] = accum[row] ? CRYPTOCONTEXT->EvalAdd(accum[row], product) : product;
            }
        }
        // Sum exactly 'per' channels. A power-of-two sum would wrap through tail slots.
        std::function<Ciphertext<DCRTPoly>(const Ciphertext<DCRTPoly>&, int)> reduce;
        reduce = [&](const Ciphertext<DCRTPoly>& ct, int n) -> Ciphertext<DCRTPoly> {
            if (n == 1) return ct;
            auto half = reduce(ct, n / 2);
            auto sum = CRYPTOCONTEXT->EvalAdd(half, CRYPTOCONTEXT->EvalRotate(half, (n / 2)*w));
            if (n % 2) sum = CRYPTOCONTEXT->EvalAdd(sum, CRYPTOCONTEXT->EvalRotate(ct, (n - 1)*w));
            return sum;
        };
        for (int row=0;row<rows;++row) {
            auto reduced = reduce(accum[row], per);
            if (stride_ == 1) accum[row] = CRYPTOCONTEXT->EvalMult(reduced, selector);
            else {
                Ciphertext<DCRTPoly> compact;
                for (int ow=ws;ow<=we;++ow) {
                    auto point = CRYPTOCONTEXT->EvalMult(reduced, compactSelectors[ow-ws]);
                    if (ow) point = CRYPTOCONTEXT->EvalRotate(point, ow*(stride_-1));
                    compact = compact ? CRYPTOCONTEXT->EvalAdd(compact, point) : point;
                }
                accum[row] = compact;
            }
            if (RESCALE_REQUIRED) accum[row] = CRYPTOCONTEXT->Rescale(accum[row]);
            if (fn % outputPer) accum[row] = CRYPTOCONTEXT->EvalRotate(accum[row], -(fn % outputPer)*outputWidth);
        }
        // Only the final block additions are serialized; per-channel work is independent.
        #ifdef _OPENMP
        #pragma omp critical(sparse_conv_fast_merge)
        #endif
        for (int row=0;row<rows;++row) {
            auto& target = y[row][fn / outputPer];
            target = target ? CRYPTOCONTEXT->EvalAdd(target, accum[row]) : accum[row];
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int row=0;row<rows;++row) for (int block=0;block<blocks;++block) {
        std::vector<double> publicValues(packing_slots_, 0);
        for (int c=block*outputPer;c<std::min(co,(block+1)*outputPer);++c) for (int ow=ws;ow<=we;++ow) {
            publicValues[(c%outputPer)*outputWidth+ow] = out[c][hs+row][ow];
            out[c][hs+row][ow] = 0;
        }
        y[row][block] = CRYPTOCONTEXT->EvalAdd(y[row][block], Encode(publicValues));
    }
    ENCRYPTED_HEIGHT_START=hs; ENCRYPTED_HEIGHT_END=he;
    ENCRYPTED_WIDTH_START=ws; ENCRYPTED_WIDTH_END=we;
}
