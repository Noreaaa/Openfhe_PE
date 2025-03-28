
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


#define DEBUG

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
            
        }
    }
}