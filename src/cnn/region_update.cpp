#include "region_update.hpp"

using std::cout;
using std::endl;

void update_Encrypted_Region(std::string layer_name,int output_h, int output_w, int filter, int padding, int stride,
int start_old_h, int end_old_h, int start_old_w, int end_old_w){
    int start_h = -1, end_h = -1, start_w = -1, end_w = -1;
    for (int oh = 0; oh < output_h; oh++){
        if (isEncrypted_h(oh * stride, filter, padding, start_old_h, end_old_h)){
            start_h = oh;
            break;
        }
    }
    for (int oh = output_h - 1; oh >= 0; oh--){
        if (isEncrypted_h(oh * stride, filter, padding, start_old_h, end_old_h)){
            end_h = oh;
            break;
        }
    }
    for (int ow = 0; ow < output_w; ow++){
        if (isEncrypted_h(ow * stride, filter, padding, start_old_w, end_old_w)){
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
    cout << "after " << layer_name << " forward" << endl;
    //cout << "ENCRYPTED_HEIGHT_START: " << ENCRYPTED_HEIGHT_START << endl;
    //cout << "ENCRYPTED_HEIGHT_END: " << ENCRYPTED_HEIGHT_END << endl;
    //cout << "ENCRYPTED_WIDTH_START: " << ENCRYPTED_WIDTH_START << endl;
    //cout << "ENCRYPTED_WIDTH_END: " << ENCRYPTED_WIDTH_END << endl;
    cout << "encrypted ratio: " 
    << double((ENCRYPTED_HEIGHT_END - ENCRYPTED_HEIGHT_START + 1) * (ENCRYPTED_WIDTH_END - ENCRYPTED_WIDTH_START + 1)) / 
    double(output_h * output_w) << endl;
    #endif
}

#define RESNET_LAYERS 18

void initialize_Encrypted_Regions(encrypted_regions enc_regions, int filter, int padding, int stride,
int start_old_h, int end_old_h, int start_old_w, int end_old_w){
    // initialize the encrypted regions based on the starting encrypted region, works for conv layers and resnet-18 only 

    // initialize the input 
    encrypted_region region_0; 
    region_0.layer_name = "input";
    region_0.height_start = start_old_h;
    region_0.height_end = end_old_h;
    region_0.width_start = start_old_w;
    region_0.width_end = end_old_w;
    enc_regions.regions.push_back(region_0);

    for(int layer_count = 0; layer_count < RESNET_LAYERS; layer_count++){
        encrypted_region curr_layer = enc_regions.regions.back();
        int output_h, output_w;
        output_h = (curr_layer.height_end - curr_layer.height_start + 1 + 2 * padding - filter) / stride + 1;
        output_w = (curr_layer.width_end - curr_layer.width_start + 1 + 2 * padding - filter) / stride + 1;
        int start_h = -1, end_h = -1, start_w = -1, end_w = -1;

        for (int oh = 0; oh < output_h; oh++){
            if (isEncrypted_h(oh * stride, filter, padding, curr_layer.height_start, curr_layer.height_end)){
                start_h = oh;
                break;
            }
        }
        for (int oh = output_h - 1; oh >= 0; oh--){
            if (isEncrypted_h(oh * stride, filter, padding, curr_layer.height_start, curr_layer.height_end)){
                end_h = oh;
                break;
            }
        }
        for (int ow = 0; ow < output_w; ow++){
            if (isEncrypted_h(ow * stride, filter, padding, curr_layer.width_start, curr_layer.width_end)){
                start_w = ow;
                break;
            }
        }
        for (int ow = output_w - 1; ow >= 0; ow--){
            if (isEncrypted_h(ow * stride, filter, padding, curr_layer.width_start, curr_layer.width_end)){
                end_w = ow;
                break;
            }
        }

        if (start_h == -1 || end_h == -1 || start_w == -1 || end_w == -1){
            std::cout << "error: encrypted height or width not updated" << std::endl;
            return;
        }

        encrypted_region new_region;
        new_region.layer_name = "conv_layer_" + std::to_string(layer_count + 1);
        new_region.height_start = start_h;
        new_region.height_end = end_h;
        new_region.width_start = start_w;
        new_region.width_end = end_w;
        enc_regions.regions.push_back(new_region);
    }
    
    #define DEBUG
    #ifdef DEBUG

    #endif
}