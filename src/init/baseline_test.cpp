#include "baseline_test.hpp"
#include "../cnn/conv2d.hpp"
#include "../cnn/activation.hpp"
#include "../cnn/pool.hpp"
#include "../cnn/linear.hpp"
#include "../encryption/encryption.hpp"
#include "../utils/dataloader.hpp"
#include "../utils/helper.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace lbcrypto;

void InitBaselineTest(Network& model, const std::string& resnet18_path, const std::string& imagenet_image_path, uint32_t numSlots, int test_nums, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair, int height_start, int height_end, int width_start, int width_end) {
    std::vector<int32_t> rotate_index;



        // load transformed image 
        types::double4d filter_4d_1, filter_4d_2, filter_4d_3, filter_4d_4, filter_4d_5, filter_4d_6;
        std::vector<double> gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6;
        std::vector<double> beta_1, beta_2, beta_3, beta_4, beta_5, beta_6;
        std::vector<double> mean_1, mean_2, mean_3, mean_4, mean_5, mean_6;
        std::vector<double> var_1, var_2, var_3, var_4, var_5, var_6;
        types::double2d linear_weight_1, linear_weight_2;
        std::vector<double> linear_bias_1, linear_bias_2;  

        CONV_BIAS = false;

        double3d input_3d = load_bin_image_double(imagenet_image_path + "ILSVRC2012_val_00000029.bin");
        std::cout << "Shape of input_3d: " << input_3d.size() << "x" << input_3d[0].size() << "x" << input_3d[0][0].size() << std::endl;

        std::cout << "ResNet-18 model is being loaded..." << std::endl;


        // convbn1
        filter_4d_1 = LoadConv2dWeight(resnet18_path + "conv1_weight.npy");
        LoadConv2dBias(resnet18_path + "bn1_weight.npy", gamma_1);
        LoadConv2dBias(resnet18_path + "bn1_bias.npy", beta_1);
        LoadConv2dBias(resnet18_path + "bn1_running_mean.npy", mean_1);
        LoadConv2dBias(resnet18_path + "bn1_running_var.npy", var_1);

        filter_4d_2 = LoadConv2dWeight(resnet18_path + "maxpool_0_weight.npy");
        LoadConv2dBias(resnet18_path + "maxpool_1_weight.npy", gamma_2);
        LoadConv2dBias(resnet18_path + "maxpool_1_bias.npy", beta_2);
        LoadConv2dBias(resnet18_path + "maxpool_1_running_mean.npy", mean_2);
        LoadConv2dBias(resnet18_path + "maxpool_1_running_var.npy", var_2);

        int stride = 1;
        int kernel_size = 3;
        int test_height = height_end - height_start + 1;
        int test_width = width_end - width_start + 1;
        int test_channels = test_nums;

        for (int32_t i = 0; i < test_height * test_height; i+=test_height){
            for (int32_t j = 0; j < kernel_size; j++){
                rotate_index.push_back(i + j);
            }
        }
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_index);

        std::vector<double> bias_1(test_channels,0.0);
        Gen_test_vector3d(input_3d, test_channels, test_height, test_width);
        print_3d(input_3d);

        Gen_test_vector4d(filter_4d_1, test_channels, test_channels, kernel_size, kernel_size);

        std::vector<Ciphertext<DCRTPoly>> x_ctxt = Encrypt_baseline(input_3d, numSlots, test_height, test_channels, CRYPTOCONTEXT, keyPair);

        model.add_layer(std::make_shared<Conv2d_C>(CONV_2D_C, "conv1", filter_4d_1, stride, test_height, test_height, numSlots));
        //model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu1"), privateKeyFHEW, test_height*test_height));
        model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu1")));
        update_index_maps(test_height, test_width, stride, 3, true);
        auto start = std::chrono::high_resolution_clock::now();
        types::double3d result =model.predict(x_ctxt);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        types::double3d golden_res = GoldenConv2d(input_3d, filter_4d_1, bias_1, stride, 0);
        golden_Relu(golden_res);

        std::cout << "golden result:" << std::endl;
        print_3d(golden_res);
        // calculate precentage loss between result and golden_res
        double sum = 0;
        for (int i = 0; i < result.size(); i++){
            for (int j = 0; j < result[0].size(); j++){
                for (int k = 0; k < result[0][0].size(); k++){

                    double pct_loss = std::abs(result[i][j][k] - golden_res[i][j][k])/std::abs(golden_res[i][j][k]);
                    sum += pct_loss;
                }
            }
        }
        std::cout << "average percentage loss: " << sum / (result.size() * result[0].size() * result[0][0].size()) << std::endl;

    
}
