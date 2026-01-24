This is the REPO for PFHE



./test --top 0 --bottom 27 --left 0 --right 27 --nums 1 -m ResNet-18 -d ImageNet -r 8192 -b 0 -> test_result.txt

to compile, in Openfhe_PE/build/ : make -j$(nproc) 2>errors.log

to launch a test run and store the result in build directory :

run 
