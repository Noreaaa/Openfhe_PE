#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>
#include <tuple>

// 简单工具函数：获取 (i, j, c) 对应在一维数组中的索引
// 输入特征图是 [H_in, W_in, C_in] 排列，行优先存储
inline int idx3D(int i, int j, int c, int width, int channels) {
    return (i * width + j) * channels + c;
}

// 同理，卷积核存储在 [K, K, C_in, C_out] 的一维向量里
inline int idx4D(int ki, int kj, int cin, int cout, 
    int K, int C_in, int C_out) {
return ((ki * K + kj) * C_in + cin) * C_out + cout;
}

/**
 * 朴素 2D 卷积 (不考虑 dilation)，输出尺寸按常规公式计算:
 *   H_out = floor((H_in + 2*pad - K) / stride) + 1
 *   W_out = floor((W_in + 2*pad - K) / stride) + 1
 *
 * 参数说明:
 *  - inputMap: 大小为 (H_in * W_in * C_in)
 *  - weight  : 大小为 (K * K * C_in * C_out)
 *  - bias    : 大小为 (C_out)，可为空指针表示无偏置
 *  - stride, pad: 整数
 *  - H_in, W_in, C_in, K, C_out: 各维度大小
 *
 * 返回值: 大小为 (H_out * W_out * C_out) 的向量
 */
std::vector<float> naiveConv2D(
    const std::vector<float> &inputMap,
    const std::vector<float> &weight,
    const std::vector<float> *bias,
    int H_in, int W_in, int C_in,
    int K, int C_out,
    int stride, int pad
) {
    // 计算输出的高度和宽度
    int H_out = (H_in + 2*pad - K) / stride + 1;
    int W_out = (W_in + 2*pad - K) / stride + 1;

    // 分配输出内存
    std::vector<float> outputMap(H_out * W_out * C_out, 0.0f);

    // 朴素卷积 5 重循环
    for(int i = 0; i < H_out; ++i) {
        for(int j = 0; j < W_out; ++j) {
            for(int co = 0; co < C_out; ++co) {
                float val = 0.0f;
                // 计算输入的起始位置
                int in_i_start = i * stride - pad;
                int in_j_start = j * stride - pad;
                for(int ki = 0; ki < K; ++ki) {
                    for(int kj = 0; kj < K; ++kj) {
                        for(int ci = 0; ci < C_in; ++ci) {
                            int in_i = in_i_start + ki;
                            int in_j = in_j_start + kj;
                            // 判断是否越界
                            if(in_i >= 0 && in_i < H_in && in_j >= 0 && in_j < W_in) {
                                float inp = inputMap[idx3D(in_i, in_j, ci, W_in, C_in)];
                                float wgt = weight[idx4D(ki, kj, ci, co, K, C_in, C_out)];
                                val += inp * wgt;
                            }
                        }
                    }
                }
                // 加上偏置
                if(bias != nullptr) {
                    val += (*bias)[co];
                }
                outputMap[idx3D(i, j, co, W_out, C_out)] = val;
            }
        }
    }
    return outputMap;
}

/**
 * 计算单维度上受影响的输出最小 / 最大索引。
 * 若无交集则返回 (1, 0) 这样的空区间。
 */
std::pair<int,int> findAffectedRange(
    int inStart, int inEnd,
    int kernelSize, int stride, int pad,
    int outSize
) {
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
    int changedTop, int changedLeft,
    int changedBottom, int changedRight,
    int inHeight, int inWidth,
    int outHeight, int outWidth,
    int kernelSize, int stride, int pad
) {
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

    return std::make_tuple(outTop, outLeft, outBottom, outRight);
}

/**
 * 在 oldOut 和 newOut 之间逐点比较，找出所有发生显著变化的位置(在任意通道上)。
 * 返回 (top, left, bottom, right) 作为最小包围框(含端点)。
 * 若无变化，返回 (1, 1, 0, 0) 表示空。
 *
 * 参数:
 *  - oldOut, newOut: 大小都是 (H_out * W_out * C_out)
 *  - eps: 判断变化的阈值
 *  - H_out, W_out, C_out: 输出特征图的三维参数
 */
std::tuple<int,int,int,int> findBoundingBoxOfChanges(
    const std::vector<float> &oldOut,
    const std::vector<float> &newOut,
    float eps,
    int H_out, int W_out, int C_out
) {
    // 找出所有有变化的位置 (i, j)
    std::vector<bool> changedMask(H_out * W_out, false);

    for(int i = 0; i < H_out; ++i) {
        for(int j = 0; j < W_out; ++j) {
            bool changedHere = false;
            for(int c = 0; c < C_out; ++c) {
                float diff = std::fabs(
                    newOut[idx3D(i, j, c, W_out, C_out)] -
                    oldOut[idx3D(i, j, c, W_out, C_out)]
                );
                if(diff > eps) {
                    changedHere = true;
                    break;
                }
            }
            changedMask[i * W_out + j] = changedHere;
        }
    }

    // 扫描 changedMask，找出 min/max 行列
    int minI = std::numeric_limits<int>::max();
    int maxI = -1;
    int minJ = std::numeric_limits<int>::max();
    int maxJ = -1;

    for(int i = 0; i < H_out; ++i) {
        for(int j = 0; j < W_out; ++j) {
            if(changedMask[i * W_out + j]) {
                if(i < minI) minI = i;
                if(i > maxI) maxI = i;
                if(j < minJ) minJ = j;
                if(j > maxJ) maxJ = j;
            }
        }
    }

    if(maxI < minI || maxJ < minJ) {
        // 没有任何变化
        return std::make_tuple(1, 1, 0, 0); // 表示空
    }
    return std::make_tuple(minI, minJ, maxI, maxJ);
}


int main() {
    // ---------------- 1. 随机数引擎准备 ----------------
    std::mt19937 gen(123);          // 固定种子以便可重复
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // ---------------- 2. 设置输入/卷积参数 ----------------
    int H_in = 10, W_in = 12, C_in = 3;   // 输入特征图大小
    int K = 3;                           // 卷积核大小(3x3)
    int C_out = 4;                       // 输出通道数
    int stride = 2;
    int pad = 1;

    // 计算输出尺寸
    int H_out = (H_in + 2*pad - K) / stride + 1;
    int W_out = (W_in + 2*pad - K) / stride + 1;

    // ---------------- 3. 随机初始化 inputMap, weight, bias ----------------
    std::vector<float> inputMap(H_in * W_in * C_in);
    for(auto &v: inputMap) v = dist(gen);

    std::vector<float> weight(K * K * C_in * C_out);
    for(auto &w: weight) w = dist(gen);

    std::vector<float> bias(C_out);
    for(auto &b: bias) b = dist(gen);

    // ---------------- 4. 做一次卷积 (oldOut) ----------------
    std::vector<float> oldOut = naiveConv2D(
        inputMap, weight, &bias,
        H_in, W_in, C_in, K, C_out,
        stride, pad
    );

    // ---------------- 5. 在输入上修改一块小矩形区域 ----------------
    int changedTop = 5, changedLeft = 5;
    int changedBottom = 7, changedRight = 7;

    // 例如给该区域所有值 +5.0
    std::vector<float> inputMapModified = inputMap;
    for(int i = changedTop; i <= changedBottom; ++i) {
        for(int j = changedLeft; j <= changedRight; ++j) {
            for(int c = 0; c < C_in; ++c) {
                inputMapModified[idx3D(i, j, c, W_in, C_in)] += 1.0f;
            }
        }
    }

    // ---------------- 6. 做第二次卷积 (newOut) ----------------
    std::vector<float> newOut = naiveConv2D(
        inputMapModified, weight, &bias,
        H_in, W_in, C_in, K, C_out,
        stride, pad
    );

    // ---------------- 7. 找出实际改变的输出区域 (realBB) ----------------
    auto realBB = findBoundingBoxOfChanges(oldOut, newOut, 1e-5f, H_out, W_out, C_out);
    auto [rTop, rLeft, rBottom, rRight] = realBB;

    // ---------------- 8. 用理论函数计算受影响区域 (calcNextConvAffectedRegion) ----------------
    auto regionBB = calcNextConvAffectedRegion(
        changedTop, changedLeft,
        changedBottom, changedRight,
        H_in, W_in,
        H_out, W_out,
        K, stride, pad
    );
    auto [cTop, cLeft, cBottom, cRight] = regionBB;

    // ---------------- 9. 打印并比较结果 ----------------
    std::cout << "== 验证示例 ==" << std::endl;
    std::cout << "输入受影响区域: top=" << changedTop 
              << ", left=" << changedLeft 
              << ", bottom=" << changedBottom 
              << ", right=" << changedRight << std::endl;

    std::cout << "实际输出改变区域: top=" << rTop 
              << ", left=" << rLeft 
              << ", bottom=" << rBottom 
              << ", right=" << rRight << std::endl;

    std::cout << "理论计算区域   : top=" << cTop
              << ", left=" << cLeft
              << ", bottom=" << cBottom
              << ", right=" << cRight << std::endl;

    // 若 realBB 是空，则可能代表计算精度或该修改没影响
    if(rTop > rBottom || rLeft > rRight) {
        std::cout << "注意: 实际输出没有任何变化 (可能是修改过小或随机参数原因)" << std::endl;
        return 0;
    }

    // 简单判断: 如果理论区域覆盖了实际区域 (realBB)，我们视为正确
    bool covered = (cTop <= rTop) && (cBottom >= rBottom) &&
                   (cLeft <= rLeft) && (cRight >= rRight);
    if(covered) {
        std::cout << "验证通过: 理论计算区域覆盖了实际改变区域!" << std::endl;
    } else {
        std::cout << "验证失败: 理论区域与实际区域不一致, 请检查!" << std::endl;
    }

    return 0;
}
