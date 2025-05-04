
#include "pool.hpp"
#include <utility>
#include <cstdlib>

using std::vector;
using std::pair;
using types::double4d;
using types::double3d;
using types::double2d;
using std::vector;
using std::cout;
using std::endl;
using namespace lbcrypto;

SumPooling::SumPooling(
    PLayerType layer_type,
    std::string layer_name,
    int kernel_size,
    int stride,
    int padding,
    uint32_t batch_size)
    : Layer(PLayerType::SUM_POOLING, layer_name),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size) {
        CONSUMED_LEVEL++;
}

SumPooling::~SumPooling() {}



void SumPooling::forward(vector<Ciphertext<DCRTPoly>>& x_cts,
vector<Ciphertext<DCRTPoly>>& y_cts) {
    #ifdef DEBUG
    cout << "SumPooling forward" << endl;
    #endif


    //int input_h = CURRENT_HEIGHT;
    //int input_w = CURRENT_WIDTH;
    //int input_c = CURRENT_CHANNEL;
    //int output_h = (input_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    //int output_w = (input_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    //int output_c = input_c;
    y_cts.clear();

    // first we do the unencrypted pooling

    for(int i = 0; i < kernel_size_; i++){
        for (int j = 0; j < kernel_size_; j++){
            
        }
    }

}



SumPooling_P::SumPooling_P(
    PLayerType layer_type,
    std::string layer_name,
    int kernel_size,
    int stride,
    int padding,
    uint32_t batch_size)
    : Layer(PLayerType::SUM_POOLING, layer_name),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size) {
        CONSUMED_LEVEL++;
}

SumPooling_P::~SumPooling_P() {}


void SumPooling_P::forward(vector<Ciphertext<DCRTPoly>>& x_cts, vector<Ciphertext<DCRTPoly>>& y_cts, double3d& x_pts, double3d& y_pts){
    #ifdef DEBUG
    cout << "SumPooling_Partial forward" << endl;
    #endif

    // first we do the unencrypted pooling
    int input_h = CURRENT_HEIGHT;
    int input_w = CURRENT_WIDTH;
    int input_c = CURRENT_CHANNEL;
    int output_h = (input_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_c = input_c;
    y_cts.clear();
    y_cts.resize(output_c);
    y_pts.resize(output_c);
    for (int i = 0; i < output_c; i++){
        y_pts[i].resize(output_h);
        for (int j = 0; j < output_h; j++){
            y_pts[i][j].resize(output_w);
        }
    }

    for (int oh = 0; oh < output_h; oh++){
        for (int ow = 0; ow < output_w; ow++){
            // check if the output value will involve encrypted data
            // to do: update the condition
            if (isEncrypted(oh * stride_, ow * stride_, kernel_size_, kernel_size_, padding_)){
                continue;
            }
            for (int c = 0; c < output_c; c++){
                double sum = 0;
                for (int fh = 0; fh < kernel_size_; fh++){
                    for (int fw = 0; fw < kernel_size_; fw++){
                        sum += x_pts[c][oh*stride_ + fh][ow*stride_ + fw];
                    }
                }
                sum /= kernel_size_ * kernel_size_;
                y_pts[c][oh][ow] = sum;
            }
        }
    }


    for (int oh = 0; oh < output_h; oh++){
        for (int ow = 0; ow < output_w; ow++){
            // check if the output value will involve encrypted data
            if (isEncrypted(oh * stride_, ow * stride_, kernel_size_, kernel_size_, padding_)){
                
            }
            
        }
    }
}

AvgPooling_P::AvgPooling_P(
    PLayerType layer_type,
    std::string layer_name,
    int kernel_size,
    int stride,
    int padding,
    uint32_t batch_size)
    : Layer(PLayerType::AVG_POOLING, layer_name),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    batch_size_(batch_size) {
        CONSUMED_LEVEL++;
}

AvgPooling_P::~AvgPooling_P() {}

//#define DEBUG
void AvgPooling_P::forward(types::vector2d<Ciphertext<DCRTPoly>>& x_cts, double3d& x_pts,
    types::vector2d<Ciphertext<DCRTPoly>>& y_cts, double3d& y_pts) {
    // Check if the input and output dimensions match
    #ifdef DEBUG
    cout << "AvgPooling_Partial forward" << endl;
    #endif


    int input_c = x_pts.size();
    int input_h = x_pts[0].size();
    int input_w = x_pts[0][0].size();
    int output_c = x_pts.size();
    int output_h = (input_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    y_cts.clear();

    y_pts.resize(output_c);

    // first we do the unencrypted pooling 
    for (int c = 0; c < output_c; c++){
        y_pts[c].resize(output_h);
        for (int oh = 0; oh < output_h; oh++){
            y_pts[c][oh].resize(output_w);
            // check if the output involves encrypted data
            for (int ow = 0; ow < output_w; ow++){
                if (!isEncrypted(oh * stride_, ow * stride_, kernel_size_, kernel_size_, padding_)){
                    double sum = 0;
                    for (int kh = 0; kh < kernel_size_; kh++){
                        for (int kw = 0; kw < kernel_size_; kw++){
                            sum += x_pts[c][oh*stride_ + kh][ow*stride_ + kw];
                        }
                    }
                    sum /= kernel_size_ * kernel_size_;
                    y_pts[c][oh][ow] = sum;
                }
            }
        }
    }

    //#define DEBUG

    #ifdef DEBUG
    cout << "value check for unencrypted pooling" << endl;
    print_3d(y_pts);
    #endif


    // now we do the encrypted pooling
    // first we need to add values if necessary
    // if encrypted column start from and end by even number nothing to be done

    int h_start_r = ENCRYPTED_HEIGHT_START % 2;
    int h_end_r = ENCRYPTED_HEIGHT_END % 2;
    int w_start_r = ENCRYPTED_WIDTH_START % 2;
    int w_end_r = ENCRYPTED_WIDTH_END % 2;

    
    int total = output_c * input_w;
    int input_w_interl = input_w / 2;
    int add_idx_s = -1;
    int add_idx_e = -1;
    // end up with odd index, we need to add the last value, which should be in the even index of y_cts
    if (w_end_r == 0){
        add_idx_e = ENCRYPTED_WIDTH_END + 1;
    }
    // start from even index, we need to add the first value, which should be in the odd index of y_cts
    if (w_start_r == 1){
        add_idx_s = ENCRYPTED_WIDTH_START - 1;
    }
    for (int h = ENCRYPTED_HEIGHT_START; h <= ENCRYPTED_HEIGHT_END; ++h){
        if (add_idx_e != -1){
            int current = add_idx_e;
            int even_batch_idx = 0;
            while (current < total){
                vector<double> batch(batch_size_, 0);
                for (size_t i = add_idx_e/2; i < batch_size_; i += input_w_interl, current += input_w) {
                    batch[i] = x_pts[current / input_w][h][current % input_w];
                }
                Plaintext batch_pt = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(batch);
                CRYPTOCONTEXT->EvalAddInPlace(batch_pt, x_cts[h - ENCRYPTED_HEIGHT_START][2 * even_batch_idx + 1]);
                even_batch_idx++;
            }
        }
        if (add_idx_s != -1){
            int current = add_idx_s;
            int odd_batch_idx = 0;
            while (current < total){
                vector<double> batch(batch_size_, 0);
                for (size_t i = add_idx_s/2; i < batch_size_; i += input_w_interl, current += input_w) {
                    batch[i] = x_pts[current / input_w][h][current % input_w];
                }
                Plaintext batch_pt = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(batch);
                CRYPTOCONTEXT->EvalAddInPlace(batch_pt, x_cts[h - ENCRYPTED_HEIGHT_START][2 * odd_batch_idx]);
                odd_batch_idx++;
            }
        }
    }

    // then we do the pooling adding each row first
    types::vector2d<Ciphertext<DCRTPoly>> y_cts_temp;
    y_cts_temp.resize(x_cts.size());
    for (size_t h = 0; h < x_cts.size(); ++h){
        for (size_t i = 0; i < x_cts[h].size(); i+=2 ){
            Ciphertext<DCRTPoly> y_ct_temp;
            y_ct_temp = CRYPTOCONTEXT->EvalAdd(x_cts[h][i], x_cts[h][i + 1]);
            y_cts_temp[h].push_back(y_ct_temp);
        }
    }

    

    size_t row_count = 0;
    size_t y_cts_size = 0;
    // add the first row
    while(row_count < y_cts_temp.size()){
        if (h_start_r == 1 && row_count == 0){
            y_cts_size++;
            row_count++;
        }
        else if (h_end_r == 0 && row_count == y_cts_temp.size() - 1){
            y_cts_size++;
            row_count++;
        }
        else{
            y_cts_size++;
            row_count += 2;
        }
    }
    y_cts.resize(y_cts_size);
    row_count = 0;
    size_t y_cts_idx = 0;

    while(row_count < y_cts_temp.size()){
        if (h_start_r == 1 && row_count == 0){
            size_t start_idx;
            size_t end_idx;
            start_idx = (w_start_r == 1) ? ENCRYPTED_WIDTH_START - 1 : ENCRYPTED_WIDTH_START;
            end_idx = (w_end_r == 0)? ENCRYPTED_WIDTH_END + 1 : ENCRYPTED_WIDTH_END;
            vector<double> row(input_w * input_c, 0);
            for (int i = 0; i < input_c; i++){
                for (size_t j = start_idx; j < end_idx; j++){
                    row[i * input_w + j] = x_pts[i][ENCRYPTED_HEIGHT_START - 1][j];
                }
            }
            // compact the row
            vector<double> compacted_row = sumAdjacentPairs(row);

            for (size_t i = 0; i < compacted_row.size(); i += batch_size_){
                vector<double> batch(compacted_row.begin() + i, compacted_row.begin() + std::min(i + batch_size_, compacted_row.size()));
                Plaintext batch_pt = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(batch);
                y_cts[y_cts_idx].push_back(CRYPTOCONTEXT->EvalAdd(batch_pt, y_cts_temp[row_count][i / batch_size_]));
            }
            row_count++;
            y_cts_idx++;
        }
        else if (h_end_r == 0 && row_count == y_cts_temp.size() - 1){
            size_t start_idx;
            size_t end_idx;
            start_idx = (w_start_r == 1) ? ENCRYPTED_WIDTH_START - 1 : ENCRYPTED_WIDTH_START;
            end_idx = (w_end_r == 0)? ENCRYPTED_WIDTH_END + 1 : ENCRYPTED_WIDTH_END;
            vector<double> row(input_w * input_c, 0);
            for (int i = 0; i < input_c; i++){
                for (size_t j = start_idx; j < end_idx; j++){
                    row[i * input_w + j] = x_pts[i][ENCRYPTED_HEIGHT_END + 1][j];
                }
            }
            // compact the row
            vector<double> compacted_row = sumAdjacentPairs(row);

            for (size_t i = 0; i < compacted_row.size(); i += batch_size_){
                vector<double> batch(compacted_row.begin() + i, compacted_row.begin() + std::min(i + batch_size_, compacted_row.size()));
                Plaintext batch_pt = CRYPTOCONTEXT->MakeCKKSPackedPlaintext(batch);
                y_cts[y_cts_idx].push_back(CRYPTOCONTEXT->EvalAdd(batch_pt, y_cts_temp[row_count][i / batch_size_]));
            }
            row_count++;
            y_cts_idx++;
        }
        else{
            // add the encrypted rows
            for (size_t i = 0; i < y_cts_temp[row_count].size(); i++){
                y_cts[y_cts_idx].push_back(CRYPTOCONTEXT->EvalAdd(y_cts_temp[row_count][i], y_cts_temp[row_count + 1][i]));
                
            }

            row_count += 2;
            y_cts_idx++;
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < y_cts.size(); i++){
        for (size_t j = 0; j < y_cts[i].size(); j++){
            CRYPTOCONTEXT->EvalMultInPlace(y_cts[i][j], 0.25);
        }
    }


    //#define ADD_CTS_CHECK
    #ifdef ADD_CTS_CHECK
    std::cout << "check cts:" << std::endl;
    for (int i = 0; i < static_cast<int>(y_cts.size()); i++){
        cout << "encrypted row: " << i << endl;
        for (int j = 0; j < static_cast<int>(y_cts[i].size()); j++){
            Plaintext plain;
            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, y_cts[i][j], &plain);
            std::cout << plain << std::endl;
        }
    }
    #endif

    // update the encrypted height an width
    // update the encrypted height and width 
    int start_h = -1, end_h = -1, start_w = -1, end_w = -1;
    for (int oh = 0; oh < output_h; oh++){
        if (isEncrypted_h(oh * stride_, kernel_size_, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            start_h = oh;
            break;
        }
    }
    for (int oh = output_h - 1; oh >= 0; oh--){
        if (isEncrypted_h(oh * stride_, kernel_size_, padding_, ENCRYPTED_HEIGHT_START, ENCRYPTED_HEIGHT_END)){
            end_h = oh;
            break;
        }
    }
    for (int ow = 0; ow < output_w; ow++){
        if (isEncrypted_h(ow * stride_, kernel_size_, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
            start_w = ow;
            break;
        }
    }
    for (int ow = output_w - 1; ow >= 0; ow--){
        if (isEncrypted_h(ow * stride_, kernel_size_, padding_, ENCRYPTED_WIDTH_START, ENCRYPTED_WIDTH_END)){
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
    //#define DEBUG
    #ifdef DEBUG
    cout << "value check for updated variables" << endl;
    cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << endl;
    cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << endl;
    cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << endl;
    cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << endl;
    cout << "check level:" << y_cts[0][0]->GetLevel() << endl;
    #endif
    
}


std::vector<double> sumAdjacentPairs(std::vector<double>& input) {
    std::vector<double> result;

    
    for (size_t i = 0; i + 1 < input.size(); i += 2) {
        result.push_back(input[i] + input[i + 1]);
    }

    return result;
}


void golden_AvgPooling(types::double3d& x_pts, int kernel_size, int stride){
        types::double3d y_pts_temp;
        int output_c = x_pts.size();
        int output_h = (x_pts[0].size() + 2 * 0 - kernel_size) / stride + 1;
        int output_w = (x_pts[0][0].size() + 2 * 0 - kernel_size) / stride + 1;

        y_pts_temp.resize(output_c);
        for (int c = 0; c < output_c; c++){
            y_pts_temp[c].resize(output_h);
            for (int oh = 0; oh < output_h; oh++){
                y_pts_temp[c][oh].resize(output_w);
                // check if the output involves encrypted data
                for (int ow = 0; ow < output_w; ow++){
                    double sum = 0;
                    for (int kh = 0; kh < kernel_size; kh++){
                        for (int kw = 0; kw < kernel_size; kw++){
                            sum += x_pts[c][oh*stride + kh][ow*stride + kw];
                        }
                    }
                    sum /= kernel_size * kernel_size;
                    y_pts_temp[c][oh][ow] = sum;
                }
            }
        }

        x_pts.clear();
        x_pts.resize(output_c);
        for (int  c = 0; c < output_c; c++){
            x_pts[c].resize(output_h);
            for (int oh = 0; oh < output_h; oh++){
                x_pts[c][oh].resize(output_w);
                for (int ow = 0; ow < output_w; ow++){
                    x_pts[c][oh][ow] = y_pts_temp[c][oh][ow];
                }
            }
        }

        
}
    