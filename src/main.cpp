#define PROFILE
#include "cmdline.h"
#include "openfhe.h"
#include "utils/types.hpp"
#include "utils/dataloader.hpp"
#include "utils/helper.hpp"
#include "encryption/encryption.hpp"
#include "cnn/model.hpp"
#include "cnn/conv2d.hpp"
#include "cnn/conv2d_l.hpp"
#include "cnn/activation.hpp"
#include "cnn/pool.hpp"
#include "cnn/bootstrap.hpp"
#include "cnn/linear.hpp"
#include <chrono>


#include <iostream>
#include <vector>

using namespace lbcrypto;
using std::vector;
CCParams<CryptoContextCKKSRNS> parameters; 
SchSwchParams params;
void SetParamCKKS(uint32_t RingDim);
void test_ablation();
void SetParam_TFHE(uint32_t RingDim);

std::string data_path = "../python_model/saved_models/";
std::string hcnn_path = data_path + "HCNN/";
std::string resnet18_path = data_path + "ResNet-18/parameters/";
std::string cifar_image_path = "../datasets/cifar-10/test_batch.bin";
std::string resnet_8_feature_path = "../datasets/imagenet-2000/feature-maps/";
std::string imagenet_image_path = "../datasets/imagenet-mini/";


int main(int argc, char *argv[]) {
    cmdline::parser parser;
    parser.add<int>("top", 0, "top of the encryption region 0-31", false, 0);
    parser.add<int>("bottom", 0, "bottom of the encryption region 0-31", false, 0);
    parser.add<int>("left", 0, "left of the encryption region 0-31", false, 0);
    parser.add<int>("right", 0, "right of the encryption region 0-31", false, 0);
    parser.add<int>("nums", 0, "number of images to be tested", false, 0);
    parser.add<std::string>("model", 'm', "model type: HCNN or ResNet-18", true, "ResNet-18");
    parser.add<std::string>("dataset", 'd', "dataset: CIFAR10, MNIST or ImageNet", false, "CIFAR10");
    parser.add<int>("ringDim", 'r', "ring dimension for CKKS scheme", false, 64);
    parser.add<int>("batch", 'b', "batch size for cts", false, 64);

    parser.parse_check(argc, argv);
    int height_start = parser.get<int>("top");
    int height_end = parser.get<int>("bottom");
    int width_start = parser.get<int>("left");
    int width_end = parser.get<int>("right");
    int test_nums = parser.get<int>("nums");
    std::string MODEL_TYPE = parser.get<std::string>("model");
    std::string DATASET = parser.get<std::string>("dataset");


    ENCRYPTED_HEIGHT_START = height_start;
    ENCRYPTED_HEIGHT_END = height_end;
    ENCRYPTED_WIDTH_START = width_start;
    ENCRYPTED_WIDTH_END = width_end;
    


    int ringDim = parser.get<int>("ringDim");
    SetParamCKKS(static_cast<uint32_t>(ringDim));
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};
    uint32_t numSlots = ringDim / 2;

    

    // set cryptocontext for CKKS
    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

    // Enable features that you wish to use. Note, we must enable FHE to use bootstrapping.
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);
    cryptoContext->Enable(SCHEMESWITCH);

    ringDim = cryptoContext->GetRingDimension();
    std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;

    cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    // Generate bootstrapping keys.
    cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    // set cryptocontext for FHEW
    lbcrypto::LWEPrivateKey privateKeyFHEW;
    bool USE_SS = true;
    if(USE_SS == true){
        SetParam_TFHE(ringDim);
        privateKeyFHEW = cryptoContext->EvalSchemeSwitchingSetup(params);
        auto ccLWE          = cryptoContext->GetBinCCForSchemeSwitch();
        CCLWE = cryptoContext->GetBinCCForSchemeSwitch();

        ccLWE->BTKeyGen(privateKeyFHEW);
        cryptoContext->EvalSchemeSwitchingKeyGen(keyPair, privateKeyFHEW);
        uint32_t logQ_ccLWE = 25;
        std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
        std::cout << ", logQ " << logQ_ccLWE;
        std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;


        auto pLWE1           = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
        auto modulus_LWE     = 1 << logQ_ccLWE;
        auto beta            = ccLWE->GetBeta().ConvertToInt();
        auto pLWE2           = modulus_LWE / (2 * beta);  // Large precision
        double scaleSignFHEW = 1.0;
        cryptoContext->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);
    }
    bool USE_APPX = false;
    if (USE_APPX == true){
        RESCALE_REQUIRED = true;
    }
    else{
        RESCALE_REQUIRED = false;
    }
    std::vector<int32_t> rotate_index;

    // create the rotate key
    int image_size = 32;

    for (int32_t i = 0; i < image_size; i++){
        rotate_index.push_back(i);
    }
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_index);

    Ciphertext<DCRTPoly> x_ctxt;
    types::vector2d<Ciphertext<DCRTPoly>> x_ctxt_2d;

    std::cout << "create the model:" << std::endl;
    // create the model
    Network model;

    double eps = 1e-5;

    CRYPTOCONTEXT = cryptoContext;
    KEYPAIR = keyPair;

    if (MODEL_TYPE == "HCNN"){

        types::double4d filter_4d_1, filter_4d_2, filter_4d_3;
        std::vector<double> gamma_1, gamma_2, gamma_3;
        std::vector<double> beta_1, beta_2, beta_3;
        std::vector<double> mean_1, mean_2, mean_3;
        std::vector<double> var_1, var_2, var_3;
        types::double2d linear_weight_1, linear_weight_2;
        std::vector<double> linear_bias_1, linear_bias_2;  
        std::vector<double> bias_1, bias_2, bias_3;

        std::cout << "HCNN model is being loaded..." << std::endl;

        filter_4d_1 = LoadConv2dWeight(hcnn_path + "conv1.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv1.0.bias.npy", bias_1);
        LoadConv2dBias(hcnn_path + "conv1.1.weight.npy", gamma_1);
        LoadConv2dBias(hcnn_path + "conv1.1.bias.npy", beta_1);
        LoadConv2dBias(hcnn_path + "conv1.1.running_mean.npy", mean_1);
        LoadConv2dBias(hcnn_path + "conv1.1.running_var.npy", var_1);


        filter_4d_2 = LoadConv2dWeight(hcnn_path + "conv2.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv2.0.bias.npy", bias_2);
        LoadConv2dBias(hcnn_path + "conv2.1.weight.npy", gamma_2);
        LoadConv2dBias(hcnn_path + "conv2.1.bias.npy", beta_2);
        LoadConv2dBias(hcnn_path + "conv2.1.running_mean.npy", mean_2);
        LoadConv2dBias(hcnn_path + "conv2.1.running_var.npy", var_2);

        filter_4d_3 = LoadConv2dWeight(hcnn_path + "conv3.0.weight.npy");
        LoadConv2dBias(hcnn_path + "conv3.0.bias.npy", bias_3);
        LoadConv2dBias(hcnn_path + "conv3.1.weight.npy", gamma_3);
        LoadConv2dBias(hcnn_path + "conv3.1.bias.npy", beta_3);
        LoadConv2dBias(hcnn_path + "conv3.1.running_mean.npy", mean_3);
        LoadConv2dBias(hcnn_path + "conv3.1.running_var.npy", var_3);
        
        LoadLinearWeight(hcnn_path + "fc1.weight.npy", linear_weight_1);
        LoadLinearWeight(hcnn_path + "fc2.weight.npy", linear_weight_2);
        LoadConv2dBias(hcnn_path + "fc1.bias.npy", linear_bias_1);
        LoadConv2dBias(hcnn_path + "fc2.bias.npy", linear_bias_2);

        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv1_bn1", filter_4d_1, 1, 1, numSlots, gamma_1, beta_1, mean_1, var_1, eps, bias_1, AVG_POOLING));
        //model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square1")));
        //model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu1"), privateKeyFHEW));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool1"), 2, 2, 0, numSlots));
        

        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv2_bn2", filter_4d_2, 1, 1, numSlots, gamma_2, beta_2, mean_2, var_2, eps, bias_2, AVG_POOLING));
        //model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square2")));
        //model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu2"), privateKeyFHEW));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool2"), 2, 2, 0, numSlots));

        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv3_bn3", filter_4d_3, 1, 1, numSlots, gamma_3, beta_3, mean_3, var_3, eps, bias_3, AVG_POOLING));
        //model.add_layer(std::make_shared<Square>(SQUARE_ACTIVATION, std::string("square3")));
        //model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu3"), privateKeyFHEW));
        model.add_layer(std::make_shared<AvgPooling_P>(AVG_POOLING, std::string("avgpool3"), 2, 2, 0, numSlots));

        model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear1", linear_weight_1, linear_bias_1, numSlots));
        model.add_layer(std::make_shared<Linear_P>(LINEAR, "linear2", linear_weight_2, linear_bias_2, numSlots));
    }
    else if (MODEL_TYPE == "ResNet-18"){
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
        //std::cout << "shape of gamma_2: " << gamma_2.size() << std::endl;
        //std::cout << "shape of beta_2: " << beta_2.size() << std::endl;
        //std::cout << "shape of mean_2: " << mean_2.size() << std::endl;
        //std::cout << "shape of var_2: " << var_2.size() << std::endl;


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

        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn1", filter_4d_1, 2, 3, numSlots, gamma_1, beta_1, mean_1, var_1, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu1"), privateKeyFHEW, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn2", filter_4d_2, 2, 1, numSlots, gamma_2, beta_2, mean_2, var_2, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu2"), privateKeyFHEW, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn3", filter_4d_3, 1, 1, numSlots, gamma_3, beta_3, mean_3, var_3, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu3"), privateKeyFHEW, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn4", filter_4d_4, 1, 1, numSlots, gamma_4, beta_4, mean_4, var_4, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu4"), privateKeyFHEW, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn5", filter_4d_5, 1, 1, numSlots, gamma_5, beta_5, mean_5, var_5, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu5"), privateKeyFHEW, numSlots));
        model.add_layer(std::make_shared<Conv2dBN_P>(CONV_2D, "conv_bn6", filter_4d_6, 1, 1, numSlots, gamma_6, beta_6, mean_6, var_6, eps, bias_1, CONV_2D_BN));
        model.add_layer(std::make_shared<Relu_ss>(RELU_SS_ACTIVATION, std::string("relu6"), privateKeyFHEW, numSlots));

        // partial encryption
        Encrypt_MCSR_P(input_3d, numSlots, 3, CRYPTOCONTEXT, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);
        //print_3d(input_3d);



        auto start = std::chrono::high_resolution_clock::now();
        int predict_label = model.predict_P(x_ctxt_2d, input_3d);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        return 0;


    }


    // test the encrypt mcsr p 
/*

    int label = 0;
    long long total_time = 0;
    int correct_count = 0; 
    CRYPTOCONTEXT = cryptoContext;
    KEYPAIR = keyPair;
    

    if (MODEL_TYPE == "HCNN"){
        for (int i = 0; i < test_nums; i++){
            LoadImageCifar(cifar_image_path, image_3d, label, i);
            NormalizeImage(image_3d);
            LoadImageCifar(cifar_image_path, verification_3d, label, i);
            NormalizeImage(verification_3d);
            Encrypt_MCSR_P(image_3d, numSlots, 2, cryptoContext, height_start, height_end, width_start, width_end, keyPair, x_ctxt_2d);

            //KEYPAIR = keyPair;
            ENCRYPTED_HEIGHT_START = height_start;
            ENCRYPTED_HEIGHT_END = height_end;
            ENCRYPTED_WIDTH_START = width_start;
            ENCRYPTED_WIDTH_END = width_end;
            std::cout << "image[" << i <<"]:" << std::endl;
            //print_3d(image_3d);


            auto start = std::chrono::high_resolution_clock::now();

            int predict_label = model.predict_P(x_ctxt_2d, image_3d);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "prediction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            //std::cout << "true result: " << label << std::endl;
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                double3d result_1 = GoldenConv2d(verification_3d, filter_4d_1, bias_1, 1, 1);
                GoldenBN(result_1, gamma_1, beta_1, mean_1, var_1, eps);
                //golden_Square(result_1);
                golden_Relu(result_1);
                golden_AvgPooling(result_1, 2, 2);
                double3d result_2 = GoldenConv2d(result_1, filter_4d_2, bias_2, 1, 1);
                GoldenBN(result_2, gamma_2, beta_2, mean_2, var_2, eps);
                //golden_Square(result_2);
                golden_Relu(result_2);
                golden_AvgPooling(result_2, 2, 2);
                double3d result_3 = GoldenConv2d(result_2, filter_4d_3, bias_3, 1, 1);
                GoldenBN(result_3, gamma_3, beta_3, mean_3, var_3, eps);
                //golden_Square(result_3);
                golden_Relu(result_3);
                golden_AvgPooling(result_3, 2, 2);
                std::vector<double> golden_output_1d;
                GoldenLinear_3d_input(result_3, linear_weight_1, linear_bias_1, golden_output_1d);
                GoldenLinear(golden_output_1d, linear_weight_2, linear_bias_2);

                print_3d(result_3);
                double max = -1000;
                int golden_label = -1;
                if(label == predict_label){
                    correct_count++;
                }
                std::cout << "predicted label: " << predict_label << ", true label: " << label << ", golden label:" << golden_label << std::endl;

            }

        }
    }
    else if (MODEL_TYPE == "ResNet-8"){
        
    }

    std::cout << "average predicted time: " << total_time/test_nums << "ms" << std::endl;
    double accuracy = (double)correct_count / (double)test_nums;
    std::cout << "accuracy: " << accuracy * 100 << "%" << std::endl;



*/


    return 0;
    

}


void SetParamCKKS(uint32_t RingDim){
    // Step 1: Set CryptoContext

    // A. Specify main parameters
    /*  A1) Secret key distribution
    * The secret key distribution for CKKS should either be SPARSE_TERNARY or UNIFORM_TERNARY.
    * The SPARSE_TERNARY distribution was used in the original CKKS paper,
    * but in this example, we use UNIFORM_TERNARY because this is included in the homomorphic
    * encryption standard.
    */
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    //parameters.SetBatchSize(BatchSize);

    /*  A2) Desired security level based on FHE standards.
    * In this example, we use the "NotSet" option, so the example can run more quickly with
    * a smaller ring dimension. Note that this should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices. In production-like environments, we recommend using
    * HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic for 128-bit, 192-bit,
    * or 256-bit security, respectively. If you choose one of these as your security level,
    * you do not need to set the ring dimension.
    */
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(RingDim);

    /*  A3) Key switching parameters.
    * By default, we use HYBRID key switching with a digit size of 3.
    * Choosing a larger digit size can reduce complexity, but the size of keys will increase.
    * Note that you can leave these lines of code out completely, since these are the default values.
    */
    parameters.SetNumLargeDigits(3);
    //parameters.SetKeySwitchTechnique(HYBRID);

    /*  A4) Scaling parameters.
    * By default, we set the modulus sizes and rescaling technique to the following values
    * to obtain a good precision and performance tradeoff. We recommend keeping the parameters
    * below unless you are an FHE expert.
    */
#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    // Currently, only FIXEDMANUAL and FIXEDAUTO modes are supported for 128-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint dcrtBits               = 78;
    usint firstMod               = 89;
#else
    // All modes are supported for 64-bit CKKS bootstrapping.
    ScalingTechnique rescaleTech = FIXEDAUTO;
    usint scaleModSize               = 51;
    usint firstMod               = 60;
#endif

    parameters.SetScalingModSize(scaleModSize);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    /*  A4) Bootstrapping parameters.
    * We set a budget for the number of levels we can consume in bootstrapping for encoding and decoding, respectively.
    * Using larger numbers of levels reduces the complexity and number of rotation keys,
    * but increases the depth required for bootstrapping.
	* We must choose values smaller than ceil(log2(slots)). A level budget of {4, 4} is good for higher ring
    * dimensions (65536 and higher).
    */
    std::vector<uint32_t> levelBudget = {3, 3};

    /* We give the user the option of configuring values for an optimization algorithm in bootstrapping.
    * Here, we specify the giant step for the baby-step-giant-step algorithm in linear transforms
    * for encoding and decoding, respectively. Either choose this to be a power of 2
    * or an exact divisor of the number of slots. Setting it to have the default value of {0, 0} allows OpenFHE to choose
    * the values automatically.
    */
    std::vector<uint32_t> bsgsDim = {0, 0};

     /*  A5) Multiplicative depth.
    * The goal of bootstrapping is to increase the number of available levels we have, or in other words,
    * to dynamically increase the multiplicative depth. However, the bootstrapping procedure itself
    * needs to consume a few levels to run. We compute the number of bootstrapping levels required
    * using GetBootstrapDepth, and add it to levelsAvailableAfterBootstrap to set our initial multiplicative
    * depth.
    */
    uint32_t levelsAvailableAfterBootstrap = 26;
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    std::cout << "depth: " << depth << std::endl;
    parameters.SetMultiplicativeDepth(depth);
}

void SetParam_TFHE(uint32_t RingDim){

    uint32_t logQ_ccLWE = 25; 
    params.SetSecurityLevelCKKS(HEStd_NotSet);
    params.SetSecurityLevelFHEW(TOY);
    params.SetNumSlotsCKKS(RingDim / 2);
    params.SetNumValues(RingDim / 2);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    
}

void test_ablation(){
    for (int i = 5; i < 13; i++){
        SetParamCKKS(1 << i);
        CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
        cryptoContext->Enable(PKE);
        cryptoContext->Enable(KEYSWITCH);
        cryptoContext->Enable(LEVELEDSHE);
        cryptoContext->Enable(ADVANCEDSHE);
        cryptoContext->Enable(FHE);
        
    
        int ringDim = cryptoContext->GetRingDimension();
        std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl << std::endl;
        auto keyPair = cryptoContext->KeyGen();
        cryptoContext->EvalMultKeyGen(keyPair.secretKey);
        cryptoContext->EvalSumKeyGen(keyPair.secretKey);

        std::vector<double> vec(ringDim/2);

        Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(vec);
        Ciphertext<DCRTPoly> ctxt = cryptoContext->Encrypt(keyPair.publicKey, ptxt);
        auto start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalAdd(ctxt, ptxt);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Add in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalMult(ctxt, ptxt);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Mult in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        ctxt = cryptoContext->EvalSum(ctxt, ringDim/2);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "ringDim: " << ringDim << " Eval Sum in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
}