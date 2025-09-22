#pragma once 

#include "openfhe.h"
#include "types.hpp"


std::tuple<int, int, int, int> CalculateRegionHCNN(int height_start, 
    int height_end, int width_start, int width_end);

void update_region(int& enc_height_start, int& enc_height_end, int& enc_width_start, int& enc_width_end, int& area,
    std::tuple<int, int, int, int> region);

void print_3d(types::double3d& data);

void print_4d(types::double4d& data);

std::pair<int,int> findAffectedRange(int inStart, int inEnd, int kernelSize, int stride, int pad, int outSize);

std::tuple<int,int,int,int> calcNextConvAffectedRegion(
    int changedTop, int changedBottom, int changedLeft, int changedRight,
    int& inHeight, int& inWidth,
    int kernelSize, int stride, int pad
);

std::tuple<int, int, int, int> calcAffectedAvgPoolingRegion(
    int& H_in, int& W_in, int Kernel, int Stride, int Padding, int h1, int h2, int w1, int w2);

std::vector<double> rotateVector(std::vector<double> vec, int steps);
void rotateVectorInplace(std::vector<double>& vec, int steps);

bool isInRange(int value, int min_val, int max_val);

bool intervalsOverlap(int a, int b, int c, int d); 

void Gen_test_vector3d(types::double3d& data, int channel, int size_h, int size_w);

void Gen_test_vector4d(types::double4d& data, int out_channel, int in_channel, int kernel_h, int kernel_w);