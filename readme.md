# PFHE: Partial Encryted FHE for CNN 

This is the REPO for PFHE, we accelerated encrypted CNN through a partial encryption only on sensitive contents. For more details, check the paper.

## Features
- Partially encrypted Conv2D
- Supports CKKS scheme (OpenFHE)
- Hybrid plaintext / ciphertext computation
- Reduced memory and latency compared to full FHE

## Requirements
- OS: Linux
- CMake >= 3.5.1
- OpenFHE
- OpenMP
- cnpy

## Datasets
Model is ResNet-18 tuned using Imagenet-mini, to test ResNet-18 performance. Download ImageNet-mini and 
place in datasets/imagenet-mini/

## Build
under Openfhe_PE/build
```bash
cmake ..
make -j$(nproc) 2>errors.log
```
## Run

to run a single image with 5x5 encrypted region on the upperleft of image with ImageNet data.

Notice: we only support a rectangle encrypted region
```bash
./test --top 0 --bottom 4 --left 0 --right 4 --nums 1 -m ResNet-18 -d ImageNet -r 4096 -> test_result.txt
```
## Command-line Arguments

| Argument | Description |
|--------|-------------|
| `--top` | Top boundary (row index) of the encrypted region in the input image (inclusive). |
| `--bottom` | Bottom boundary (row index) of the encrypted region (inclusive). |
| `--left` | Left boundary (column index) of the encrypted region (inclusive). |
| `--right` | Right boundary (column index) of the encrypted region (inclusive). |
| `--nums` | Number of test samples to evaluate. |
| `-m` | Model name. Currently supports `ResNet-18`. |
| `-d` | Dataset used for evaluation (e.g., `ImageNet`). |
| `-r` | Ring dimension used in the CKKS scheme. |
| `--act` | ReLu activation type 0 for scheme switching, 1 for polynomial approximation (degree 7) |
| `>` | Redirects the program output to a file. |
| `test_result.txt` | File used to store the execution results. |
