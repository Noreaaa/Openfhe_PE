#pragma once 

#include "layer.hpp"


struct encrypted_region{
    std::string layer_name;
    int height_start;
    int height_end;
    int width_start;
    int width_end;
};

struct encrypted_regions{
    std::vector<encrypted_region> regions;
};


void update_Encrypted_Region(std::string layer_name,int output_h, int output_w, int filter, int padding, int stride);


void initialize_Encrypted_Regions(encrypted_regions enc_regions, int filter, int padding, int stride,
int start_old_h, int end_old_h, int start_old_w, int end_old_w);
