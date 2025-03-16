
#include "helper.hpp"
using std::vector;

enum LayerType {
    CONV_2D,
    BATCH_NORM,
    AVERAGE_POOL,
    SQUARE_ACTIVATION,
    FULLY_CONNECTED
};
/**
 * 10x10 region 
 * top left bot right
 * height_start width_start height_end width_end
 * 
 */
std::tuple<int, int, int, int> CalculateRegionHCNN(int height_start, 
    int height_end, int width_start, int width_end) {
    
    int channel_size = 0;
    int height = 32;
    int width = 32;
    vector<int> filter_sizes = {3,3,3};
    vector<int> strides = {1,1,1};
    vector<int> paddings = {1,1,1};

    vector<int> in_channel = {3, 32, 64, 128};
    
    int enc_height_start = height_start;
    int enc_height_end = height_end;
    int enc_width_start = width_start;
    int enc_width_end = width_end;
    int area = (enc_height_end - enc_height_start) * (enc_width_end - enc_width_start);

    // model structure used for cifar 10
    /**
     * conv2d kernel 3x3 stride 1 padding 1 channel 32
     * batch norm 32 
     * square activation 
     * average pool kernel 2x2 stride 2 padding 0
     * 
     * conv2d kernel 3x3 stride 1 padding 1 channel 64
     * batch norm 32 
     * square activation 
     * average pool kernel 2x2 stride 2 padding 0
     * 
     * conv2d kernel 3x3 stride 1 padding 1 channel 128
     * batch norm 32 
     * square activation 
     * average pool kernel 2x2 stride 2 padding 0
     * 
     * fc 128*4*4 -> 256 
     * fc 256 -> 10
     * 
     */

    std::tuple<int, int, int, int> current_region = calcNextConvAffectedRegion(height_start, height_end, 
    width_start, width_end, height, width, filter_sizes[0], strides[0], paddings[0]);
    
    // update region if the region is larger
    update_region(enc_height_start, enc_height_end, enc_width_start, enc_width_end, area, current_region);
    
    std::cout << "region after conv1: " << enc_height_start << " " << enc_height_end << " " << enc_width_start << " " << enc_width_end << std::endl;

    current_region = calcAffectedAvgPoolingRegion(height, width, 2, 2, 0, enc_height_start, enc_height_end, enc_width_start, enc_width_end);
    update_region(enc_height_start, enc_height_end, enc_width_start, enc_width_end, area, current_region);
    std::cout << "region after pool1: " << enc_height_start << " " << enc_height_end << " " << enc_width_start << " " << enc_width_end << std::endl;
    

    return std::make_tuple(enc_height_start, enc_height_end, enc_width_start, enc_width_end);
}

void update_region(int& enc_height_start, int& enc_height_end, int& enc_width_start, int& enc_width_end, int& area,
    std::tuple<int, int, int, int> region){
    if((std::get<1>(region) - std::get<0>(region)) * (std::get<3>(region) - std::get<2>(region)) > area){
        enc_height_end = std::get<1>(region);
        enc_height_start = std::get<0>(region);
        enc_width_end = std::get<3>(region);
        enc_width_start = std::get<2>(region);
        area = (enc_height_end - enc_height_start) * (enc_width_end - enc_width_start);
    }
}

void print_3d(types::double3d& data){
    for (int i = 0; i < data.size(); i++){
        for (int j = 0; j < data[i].size(); j++){
            for (int k = 0; k < data[i][j].size(); k++){
                std::cout << data[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * 计算单维度上受影响的输出最小 / 最大索引。
 * 若无交集则返回 (1, 0) 这样的空区间。
 */
std::pair<int,int> findAffectedRange(int inStart, int inEnd, int kernelSize, int stride, int pad, int outSize) {
    if(inEnd < inStart) {
        // 无效区间
        return std::make_pair(1, 0);
    }

    // i_min = ceil((inStart + pad - (kernelSize - 1)) / stride)
    // i_max = floor((inEnd   + pad) / stride)
    double iMinFloat = double(inStart + pad - (kernelSize - 1)) / double(stride);
    double iMaxFloat = double(inEnd   + pad) / double(stride);

    int iMin = int(std::ceil(iMinFloat));
    int iMax = int(std::floor(iMaxFloat));

    // 裁剪到输出范围 [0, outSize - 1]
    iMin = std::max(iMin, 0);
    iMax = std::min(iMax, outSize - 1);
    return std::make_pair(iMin, iMax);
}

/**
 * 给定输入受影响区域 (changed_top, changed_left, changed_bottom, changed_right)
 * 计算下一个卷积层输出特征图 (out_height, out_width) 上的受影响区域。
 */
std::tuple<int,int,int,int> calcNextConvAffectedRegion(
    int changedTop, int changedBottom, int changedLeft, int changedRight,
    int& inHeight, int& inWidth,
    int kernelSize, int stride, int pad
) {
    int outHeight = (inHeight + 2 * pad - kernelSize) / stride + 1;
    int outWidth = (inWidth + 2 * pad - kernelSize) / stride + 1;
    // 行方向
    auto [outTop, outBottom] = findAffectedRange(
        changedTop, changedBottom,
        kernelSize, stride, pad, outHeight
    );

    // 列方向
    auto [outLeft, outRight] = findAffectedRange(
        changedLeft, changedRight,
        kernelSize, stride, pad, outWidth
    );

    inHeight = outHeight;
    inWidth = outWidth;
    return std::make_tuple(outTop, outBottom, outLeft, outRight);
    
}


/**
 * 计算在 average pooling 层中受影响的区域

    参数：
    - H_in, W_in: 输入 feature map 的高度和宽度
    - Kernel : 池化窗口的大小
    - Stride : 池化层的步长
    - Padding   : 输入 feature map 的 padding 大小
    - h1, h2    : 修改区域的height索引范围 [r1, r2]
    - w1, w2    : 修改区域的width索引范围 [c1, c2]

    返回：
    - (i_min, i_max, j_min, j_max) 受影响区域在池化层输出 feature map 的范围
 */

std::tuple<int, int, int, int> calcAffectedAvgPoolingRegion(
    int& H_in, int& W_in, int Kernel, int Stride, int Padding, int h1, int h2, int w1, int w2) 
{

    int H_pad = H_in + 2 * Padding;
    int W_pad = W_in + 2 * Padding;

    int H_out = (H_pad - Kernel) / Stride + 1;
    int W_out = (W_pad - Kernel) / Stride + 1;

    h1 += Padding;
    h2 += Padding;
    w1 += Padding;
    w2 += Padding;
    // 计算受影响的输出索引范围
    int outTop = std::max(0, (h1 - Kernel + 1 + Stride - 1) / Stride);
    int outBottom = std::min(H_out - 1, h2 / Stride);

    int outLeft = std::max(0, (w1 - Kernel + 1 + Stride - 1) / Stride);
    int outRight = std::min(W_out - 1, w2 / Stride);

    H_in = H_out;
    W_in = W_out;
    return std::make_tuple(outTop, outBottom, outLeft, outRight);
}
/**helper function:rotateVector:
 * rotate the vector by the steps
 * positive steps rotate to the right
 * negative steps rotate to the left
 */
std::vector<double> rotateVector(std::vector<double> vec, int steps) {
    if (vec.empty()) return {}; // Handle empty vector case

    int n = vec.size();
    steps = steps % n; // Normalize steps to be within the vector size

    if (steps < 0) {
        steps += n; // Convert negative steps to positive equivalent
    }

    // Rotate the vector
    std::vector<double> rotated(n);
    for (int i = 0; i < n; i++) {
        rotated[(i + steps) % n] = vec[i];
    }

    return rotated;
}

/**helper function: rotateVectorInplace: 
 * rotate the vector by the steps 
 * positive steps rotate to the right 
 * negative steps rotate to the left
 */
void rotateVectorInplace(std::vector<double>& vec, int steps) {
    if (vec.empty()) return; // Handle empty vector case

    int n = vec.size();
    steps = steps % n; // Normalize steps to be within the vector size

    if (steps < 0) {
        steps += n; // Convert negative steps to positive equivalent
    }

    // Rotate the vector
    std::vector<double> rotated(vec.end() - steps, vec.end());
    std::copy(vec.begin(), vec.end() - steps, vec.begin() + steps);
    std::copy(rotated.begin(), rotated.end(), vec.begin());
}




bool isInRange(int value, int min_val, int max_val) {
    return (value >= min_val && value <= max_val);
}


bool intervalsOverlap(int a, int b, int c, int d) {return !(b <= c || d < a);}