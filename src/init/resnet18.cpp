#include "resnet18.hpp"
#include "../cnn/conv2d.hpp"
#include "../cnn/sparse_conv2d.hpp"
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

void InitResNet18(Network& model, const std::string& resnet18_path, const std::string& imagenet_image_path, uint32_t numSlots, double eps, int act_type, lbcrypto::LWEPrivateKey privateKeyFHEW, lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair, int height_start, int height_end, int width_start, int width_end) {
    types::vector2d<Ciphertext<DCRTPoly>> x_ctxt_2d;

        // load transformed image 
        types::double4d filter_4d_1, filter_4d_2, filter_4d_3, filter_4d_4, filter_4d_5, filter_4d_6;
        std::vector<double> gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6;
        std::vector<double> beta_1, beta_2, beta_3, beta_4, beta_5, beta_6;
        std::vector<double> mean_1, mean_2, mean_3, mean_4, mean_5, mean_6;
        std::vector<double> var_1, var_2, var_3, var_4, var_5, var_6;
        types::double2d linear_weight_1, linear_weight_2;
        std::vector<double> linear_bias_1, linear_bias_2;  
        std::vector<double> bias_1(64,0.0);

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

        filter_4d_3 = LoadConv2dWeight(resnet18_path + "layer1_0_conv1_weight.npy");
        LoadConv2dBias(resnet18_path + "layer1_0_bn1_weight.npy", gamma_3);
        LoadConv2dBias(resnet18_path + "layer1_0_bn1_bias.npy", beta_3);
        LoadConv2dBias(resnet18_path + "layer1_0_bn1_running_mean.npy", mean_3);
        LoadConv2dBias(resnet18_path + "layer1_0_bn1_running_var.npy", var_3);

        filter_4d_4 = LoadConv2dWeight(resnet18_path + "layer1_0_conv2_weight.npy");
        LoadConv2dBias(resnet18_path + "layer1_0_bn2_weight.npy", gamma_4);
        LoadConv2dBias(resnet18_path + "layer1_0_bn2_bias.npy", beta_4);
        LoadConv2dBias(resnet18_path + "layer1_0_bn2_running_mean.npy", mean_4);
        LoadConv2dBias(resnet18_path + "layer1_0_bn2_running_var.npy", var_4);

        filter_4d_5 = LoadConv2dWeight(resnet18_path + "layer1_1_conv1_weight.npy");
        LoadConv2dBias(resnet18_path + "layer1_1_bn1_weight.npy", gamma_5);
        LoadConv2dBias(resnet18_path + "layer1_1_bn1_bias.npy", beta_5);
        LoadConv2dBias(resnet18_path + "layer1_1_bn1_running_mean.npy", mean_5);
        LoadConv2dBias(resnet18_path + "layer1_1_bn1_running_var.npy", var_5);

        filter_4d_6 = LoadConv2dWeight(resnet18_path + "layer1_1_conv2_weight.npy");
        LoadConv2dBias(resnet18_path + "layer1_1_bn2_weight.npy", gamma_6);
        LoadConv2dBias(resnet18_path + "layer1_1_bn2_bias.npy", beta_6);
        LoadConv2dBias(resnet18_path + "layer1_1_bn2_running_mean.npy", mean_6);
        LoadConv2dBias(resnet18_path + "layer1_1_bn2_running_var.npy", var_6);

        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn1", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_1, 2, 3, gamma_1, beta_1, mean_1, var_1, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu1"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu1")));
        }
        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn2", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_2, 2, 1, gamma_2, beta_2, mean_2, var_2, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu2"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu2")));
        }
        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn3", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_3, 1, 1, gamma_3, beta_3, mean_3, var_3, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu3"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu3")));
        }
        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn4", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_4, 1, 1, gamma_4, beta_4, mean_4, var_4, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu4"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu4")));
        }
        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn5", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_5, 1, 1, gamma_5, beta_5, mean_5, var_5, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu5"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu5")));
        }
        model.add_layer(std::make_shared<SparseConv2dBN_P>(CONV_2D, "conv_bn6", CRYPTOCONTEXT->GetRingDimension(), numSlots, filter_4d_6, 1, 1, gamma_6, beta_6, mean_6, var_6, eps, bias_1, CONV_2D_BN));
        if (act_type == 0){
            model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu6"), privateKeyFHEW, numSlots));
        }
        else{
            model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION, std::string("relu6")));
        }
        types::double3d golden_res = GoldenConv2d(input_3d, filter_4d_1, bias_1, 2, 3);
        GoldenBN(golden_res, gamma_1, beta_1, mean_1, var_1, eps);
        golden_Relu(golden_res);
        //print_3d(golden_res);

        USE_COMPACT = false;
        if (USE_COMPACT == true){
            Encrypt_MCSR_P_COMPACT(input_3d, numSlots, CRYPTOCONTEXT, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);
        }
        else{
            Encrypt_MCSR_P(input_3d, numSlots, CRYPTOCONTEXT, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);
        }

        auto start = std::chrono::high_resolution_clock::now();
        int predict_label = model.predict_P(x_ctxt_2d, input_3d);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    
}
