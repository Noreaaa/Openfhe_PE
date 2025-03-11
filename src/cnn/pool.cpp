
#include "pool.hpp"
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


    int input_h = CURRENT_HEIGHT;
    int input_w = CURRENT_WIDTH;
    int input_c = CURRENT_CHANNEL;
    int output_h = (input_h + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_w = (input_w + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_c = input_c;
    y_cts.clear();


    for(int i = 0; i < kernel_size_; i++){
        for (int j = 0; j < kernel_size_; j++){
            
        }
    }


    
    
}