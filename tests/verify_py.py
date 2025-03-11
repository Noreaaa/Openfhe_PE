import numpy as np
import math

def naive_conv2d(input_map, weight, bias, stride, pad):
    """
    朴素实现 2D 卷积（不考虑 dilation），用于对比验证。
    - input_map:  [H_in, W_in, C_in]
    - weight:     [K, K, C_in, C_out]
    - bias:       [C_out] (可为 None 表示无偏置)
    - stride, pad: 整数
    
    返回:
    - output_map: [H_out, W_out, C_out]
    
    做法：
      output(i,j,c_out) = sum_{c_in, ki, kj} input_map(...) * weight(ki, kj, c_in, c_out) + bias[c_out]
      注意索引越界时用 0 做填充。
    """
    H_in, W_in, C_in = input_map.shape
    K, K2, C_in_w, C_out = weight.shape
    assert K == K2, "卷积核必须是方形"
    assert C_in == C_in_w, "weight 的输入通道数应与输入特征图一致"
    
    # 根据公式计算输出空间尺寸
    # 通常：H_out = floor((H_in + 2*pad - K)/stride + 1)
    #       W_out = floor((W_in + 2*pad - K)/stride + 1)
    H_out = (H_in + 2*pad - K)//stride + 1
    W_out = (W_in + 2*pad - K)//stride + 1
    
    # 准备输出
    output_map = np.zeros((H_out, W_out, C_out), dtype=np.float32)
    
    # 遍历输出每一个像素 (i, j) 和通道 c_out
    for i in range(H_out):
        for j in range(W_out):
            for c_o in range(C_out):
                val = 0.0
                # 计算输入感受野的起始顶点
                in_i_start = i*stride - pad
                in_j_start = j*stride - pad
                # 卷积核的遍历
                for ki in range(K):
                    for kj in range(K):
                        for c_i in range(C_in):
                            in_i = in_i_start + ki
                            in_j = in_j_start + kj
                            # 判断越界
                            if (0 <= in_i < H_in) and (0 <= in_j < W_in):
                                val += input_map[in_i, in_j, c_i] * weight[ki, kj, c_i, c_o]
                if bias is not None:
                    val += bias[c_o]
                output_map[i, j, c_o] = val
    
    return output_map


def find_affected_range(in_start, in_end, kernel_size, stride, pad, out_size):
    """
    计算单维度上受影响的输出最小 / 最大索引。
    若无交集则返回 (1, 0) 这样的空区间。
    """
    if in_end < in_start:
        return (1, 0)  # 无效区域
    
    i_min = math.ceil((in_start + pad - (kernel_size - 1)) / float(stride))
    i_max = math.floor((in_end + pad) / float(stride))
    
    # 裁剪到输出空间范围
    i_min = max(i_min, 0)
    i_max = min(i_max, out_size - 1)
    return (i_min, i_max)


def calc_next_conv_affected_region(
    changed_top, changed_left, 
    changed_bottom, changed_right,
    in_height, in_width, 
    out_height, out_width, 
    kernel_size, stride, pad
):
    """
    计算下一个卷积层输出特征图中受 [changed_top, changed_left, changed_bottom, changed_right]
    影响的区域 (含端点)。
    """
    # 行方向
    out_top, out_bottom = find_affected_range(
        changed_top, changed_bottom, kernel_size, stride, pad, out_height
    )
    # 列方向
    out_left, out_right = find_affected_range(
        changed_left, changed_right, kernel_size, stride, pad, out_width
    )
    return (out_top, out_left, out_bottom, out_right)


def find_bounding_box_of_changes(old_out, new_out, eps=1e-7):
    """
    给定两份输出 (H_out, W_out, C_out)，找出哪些位置在“任何输出通道”上出现了显著变化。
    返回 (top, left, bottom, right) 作为空间维度上的最小包围框（含端点）。
    如果没有任何变化，返回 (1,1,0,0) 代表空区域。
    """
    # 先计算各位置(不含通道)是否有变化
    diff_map = np.abs(new_out - old_out)
    # 在任何通道上有变化就算改动
    changed_mask = np.any(diff_map > eps, axis=-1)  # 变成 [H_out, W_out] 的布尔矩阵

    ys, xs = np.where(changed_mask)  # 找到改动的位置
    if len(ys) == 0:
        # 说明没任何改动
        return (1, 1, 0, 0)  # 空
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    return (top, left, bottom, right)


def verify_calc_next_conv_affected_region():
    """
    用一个随机示例来验证 calc_next_conv_affected_region 的正确性。
    我们做以下步骤：
      1. 构造随机输入、卷积核、偏置
      2. 做一次卷积得到 old_out
      3. 在输入上某个小矩形区域修改数值
      4. 再做一次卷积得到 new_out
      5. 找出 new_out 与 old_out 的最小空间变化区域（称为 real_bb）
      6. 用 calc_next_conv_affected_region 计算理论受影响范围 region_bb
      7. 比较 real_bb 与 region_bb 是否一致（或者至少 region_bb 覆盖 real_bb）
    """
    np.random.seed(123)
    
    # 1. 随机构造输入、权重、偏置参数
    H_in, W_in, C_in = 10, 12, 3   # 输入特征图大小 (10 x 12 x 3)
    K = 3
    C_out = 4
    stride = 1
    pad = 1
    
    # 根据卷积公式计算输出高宽
    H_out = (H_in + 2*pad - K)//stride + 1
    W_out = (W_in + 2*pad - K)//stride + 1

    input_map = np.random.randn(H_in, W_in, C_in).astype(np.float32)
    weight = np.random.randn(K, K, C_in, C_out).astype(np.float32)
    bias = np.random.randn(C_out).astype(np.float32)
    
    # 2. 做一次卷积
    old_out = naive_conv2d(input_map, weight, bias, stride, pad)
    
    # 3. 在输入上修改一块小矩形区域
    changed_top, changed_left = 3, 3
    changed_bottom, changed_right = 5, 5
    # 随机加一些值
    input_map_modified = input_map.copy()
    input_map_modified[changed_top:changed_bottom+1, changed_left:changed_right+1, :] += 5.0
    
    # 4. 做第二次卷积
    new_out = naive_conv2d(input_map_modified, weight, bias, stride, pad)
    
    # 5. 找出真实改变的输出区域
    real_bb = find_bounding_box_of_changes(old_out, new_out, eps=1e-5)
    
    # 6. 用我们的函数计算理论受影响范围
    region_bb = calc_next_conv_affected_region(
        changed_top, changed_left, changed_bottom, changed_right,
        H_in, W_in, H_out, W_out,
        kernel_size=K, stride=stride, pad=pad
    )
    
    print("== 验证示例 ==")
    print(f"输入受影响区域: top={changed_top}, left={changed_left}, bottom={changed_bottom}, right={changed_right}")
    print(f"实际输出改变区域: {real_bb} (top, left, bottom, right)")
    print(f"理论计算区域   : {region_bb} (top, left, bottom, right)")
    
    # 7. 做一个简易判断：理论区域若能覆盖实际区域，就认为正确
    rt, rl, rb, rr = real_bb
    ct, cl, cb, cr = region_bb
    
    # 如果 real_bb 是空区域，直接说明没有变化
    if rt > rb or rl > rr:
        print("输出完全没有变化，可能是因为参数或修改太小 / 计算精度等原因。")
        return
    
    # 判断覆盖关系
    covered = (ct <= rt) and (cb >= rb) and (cl <= rl) and (cr >= rr)
    
    if covered:
        print("验证通过：理论计算区域覆盖了实际改变区域！")
    else:
        print("验证失败：理论区域与实际区域不一致，请检查具体值！")


if __name__ == "__main__":
    verify_calc_next_conv_affected_region()
