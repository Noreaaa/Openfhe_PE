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

## Build
under Openfhe_PE/build
```bash

run command 


./test --top 0 --bottom 27 --left 0 --right 27 --nums 1 -m ResNet-18 -d ImageNet -r 8192 -b 0 -> test_result.txt

to compile, in Openfhe_PE/build/ : make -j$(nproc) 2>errors.log

to launch a test run and store the result in build directory :

run 
