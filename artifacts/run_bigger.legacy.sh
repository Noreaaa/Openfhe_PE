#!/bin/bash
# 中心方形区域加密脚本
# 从 50x50 递增 10 到 100x100

set -e

# 中心点（由 top=87 bottom=137 left=87 right=137 推得）
CENTER=112

# 固定参数
NUMS=1
MODEL="ResNet-18"
DATASET="ImageNet"
R=2048
B=0

# 循环：10 -> 100，步长 10
for size in $(seq 10 10 100); do
    half=$((size / 2))
    top=$((CENTER - half))
    bottom=$((CENTER + half))
    left=$((CENTER - half))
    right=$((CENTER + half))

    out="center_${size}x${size}.txt"

    echo "[*] size=${size}x${size}  top=$top bottom=$bottom left=$left right=$right"
    echo "    -> $out"

    ./test \
        --top "$top" \
        --bottom "$bottom" \
        --left "$left" \
        --right "$right" \
        --nums "$NUMS" \
        -m "$MODEL" \
        -d "$DATASET" \
        -r "$R" \
        -b "$B" \
        > "$out"
done

echo "[+] All done."