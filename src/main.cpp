#define PROFILE
#include "cmdline.h"
#include "openfhe.h"
#include <sys/resource.h>
#include "utils/types.hpp"
#include "utils/dataloader.hpp"
#include "utils/helper.hpp"
#include "encryption/encryption.hpp"
#include "encryption/setparam.hpp"
#include "init/hcnn.hpp"
#include "init/resnet18.hpp"
#include "init/baseline_test.hpp"
#include "cnn/model.hpp"
#include "cnn/conv2d.hpp"
#include "cnn/conv2d_l.hpp"
#include "cnn/activation.hpp"
#include "cnn/pool.hpp"
#include "cnn/bootstrap.hpp"
#include "cnn/linear.hpp"
#include "cnn/region_update.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>


#include <iostream>
#include <vector>

using namespace lbcrypto;
using std::vector;
CCParams<CryptoContextCKKSRNS> parameters; 
SchSwchParams params;
void test_ablation();

std::string data_path = "../python_model/saved_models/";
std::string hcnn_path = data_path + "HCNN/";
std::string resnet18_path = data_path + "ResNet-18/parameters/";
std::string cifar_image_path = "../datasets/cifar-10/test_batch.bin";
std::string resnet_8_feature_path = "../datasets/imagenet-2000/feature-maps/";
std::string imagenet_image_path = "../datasets/imagenet-mini/";


int main(int argc, char *argv[]) {
    cmdline::parser parser;
    parser.add<int>("top", 0, "top of the encryption region", false, 0);
    parser.add<int>("bottom", 0, "bottom of the encryption region", false, 0);
    parser.add<int>("left", 0, "left of the encryption region", false, 0);
    parser.add<int>("right", 0, "right of the encryption region", false, 0);
    parser.add<int>("nums", 0, "number of images to be tested", false, 0);
    parser.add<std::string>("model", 'm', "model type: HCNN or ResNet-18", true, "ResNet-18");
    parser.add<std::string>("dataset", 'd', "dataset: CIFAR10, MNIST or ImageNet", false, "CIFAR10");
    parser.add<int>("ringDim", 'r', "ring dimension for CKKS scheme", false, 16384);
    parser.add<int>("batch", 'b', "logical packing slots (0 selects ringDim/2)", false, 1024);
    parser.add<int>("act", 'a', "activation type 0 for scheme switching, 1 for polynomial approximation", false, 0);

    parser.add("check-config", 0, "validate parameters and exit before generating contexts or keys");
    parser.add("verify", 0, "compare each ResNet prefix against a plaintext reference (outside timed sections)");
    parser.add<int>("layers", 0, "number of sequential ResNet convolutions (1..10)", false, 6);
    parser.add<int>("depth", 0, "CKKS multiplicative depth (0 selects sufficient depth for the prefix)", false, 0);
    parser.add<int>("encrypt-size", 0, "center square size in the 224x224 ResNet input", false, 10);
    parser.add<std::string>("profile", 0, "write per-layer and cumulative prefix timings to CSV", false, "");
    parser.add<std::string>("data-root", 0, "repository root containing datasets and python_model", false,
        std::filesystem::exists("python_model") ? "." : "..");
    parser.parse_check(argc, argv);
    int height_start = parser.get<int>("top");
    int height_end = parser.get<int>("bottom");
    int width_start = parser.get<int>("left");
    int width_end = parser.get<int>("right");
    int test_nums = parser.get<int>("nums");
    std::string MODEL_TYPE = parser.get<std::string>("model");
    std::string DATASET = parser.get<std::string>("dataset");
    int act_type = parser.get<int>("act");
    const int convLayers=parser.get<int>("layers"), requestedDepth=parser.get<int>("depth");
    if(convLayers<1 || convLayers>10 || requestedDepth<0) {
        std::cerr<<"layers must be 1..10 and depth must be non-negative"<<std::endl; return 1;
    }
    if(MODEL_TYPE=="ResNet-18" && !parser.exist("top") && !parser.exist("bottom") &&
       !parser.exist("left") && !parser.exist("right")) {
        const int size=parser.get<int>("encrypt-size");
        if(size<1 || size>224) { std::cerr<<"encrypt-size must be 1..224"<<std::endl; return 1; }
        height_start=width_start=(224-size)/2;
        height_end=width_end=height_start+size-1;
    }
    const std::string repository=parser.get<std::string>("data-root");
    resnet18_path=repository+"/python_model/saved_models/ResNet-18/parameters/";
    imagenet_image_path=repository+"/datasets/imagenet-mini/";



    ENCRYPTED_HEIGHT_START = height_start;
    ENCRYPTED_HEIGHT_END = height_end;
    ENCRYPTED_WIDTH_START = width_start;
    ENCRYPTED_WIDTH_END = width_end;
    


    int ringDim = parser.get<int>("ringDim");
    int requestedSlots = parser.get<int>("batch");
    if (ringDim < 2 || requestedSlots < 0) {
        std::cerr << "Invalid ringDim or batch" << std::endl;
        return 1;
    }
    uint32_t numSlots = requestedSlots == 0 ? ringDim / 2 : requestedSlots;
    SetParamCKKS(static_cast<uint32_t>(ringDim), parameters, numSlots);
    if(MODEL_TYPE=="ResNet-18" && act_type==0) {
        const int minimumDepth=3*convLayers+4;
        if(requestedDepth && requestedDepth<minimumDepth) {
            std::cerr<<"Continuous Conv+scheme-switch ReLU prefix requires depth >= "<<minimumDepth
                     <<" (no bootstrap/refresh in this path)"<<std::endl; return 1;
        }
        parameters.SetMultiplicativeDepth(requestedDepth ? requestedDepth : std::max(23,3*convLayers+6));
    } else if(requestedDepth) parameters.SetMultiplicativeDepth(requestedDepth);
    const bool needsBootstrap=MODEL_TYPE!="ResNet-18" || act_type!=0;
    std::cout<<"RUN_CONFIG layers="<<convLayers<<" depth="<<parameters.GetMultiplicativeDepth()
             <<" ring="<<ringDim<<" slots="<<numSlots<<" input_encrypted_h="<<height_start<<':'<<height_end
             <<" input_encrypted_w="<<width_start<<':'<<width_end<<std::endl;
    if(parser.exist("check-config")) return 0;
    std::vector<uint32_t> levelBudget = {3, 3};
    std::vector<uint32_t> bsgsDim = {0, 0};


    

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
    std::cout << "CKKS scheme is using ring dimension " << ringDim
              << ", logical packing slots " << numSlots << std::endl << std::endl;

    if(needsBootstrap) cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots, 10);

    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);
    // Generate bootstrapping keys.
    if(needsBootstrap) cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    // set cryptocontext for FHEW
    lbcrypto::LWEPrivateKey privateKeyFHEW;
    bool USE_SS = act_type == 0;
    if(USE_SS == true){
        SetParam_TFHE(ringDim, params, numSlots);
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
    bool USE_APPX = act_type != 0;
    if (USE_APPX == true){
        RESCALE_REQUIRED = true;
    }
    else{
        RESCALE_REQUIRED = false;
    }
    std::vector<int32_t> rotate_index;

    // create the rotate key
    int image_size = 32;
    if(MODEL_TYPE != "baseline_test"){
        for (int32_t i = 0; i < image_size; i++){
            rotate_index.push_back(i);
        }
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotate_index);
    }
    else{
        
    }


    std::cout << "create the model:" << std::endl;
    // create the model
    Network model;

    double eps = 1e-5;

    CRYPTOCONTEXT = cryptoContext;
    KEYPAIR = keyPair;

    if (MODEL_TYPE == "HCNN"){
        InitHCNN(model, hcnn_path, numSlots, eps);
    }
    else if (MODEL_TYPE == "ResNet-18"){
        InitResNet18(model, resnet18_path, imagenet_image_path, numSlots, eps, act_type, privateKeyFHEW, keyPair, height_start, height_end, width_start, width_end, convLayers, parser.get<std::string>("profile"), parser.exist("verify"));
    }
    else if (MODEL_TYPE == "baseline_test"){
        InitBaselineTest(model, resnet18_path, imagenet_image_path, numSlots, test_nums, cryptoContext, keyPair, height_start, height_end, width_start, width_end);
    } /*else if (MODEL_TYPE == "encryption_estimate"){
        encrypted_regions enc_regions;
        vector<int> filter_sizes = {};
        vector<int> stride_sizes = {};
        vector<int> padding_sizes = {};
        initialize_Encrypted_Regions(enc_regions, 3, 1, )

    }*/

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double peak_gb = usage.ru_maxrss / 1024.0 / 1024.0;  // KB -> GB
    std::cout << "Peak memory: " << peak_gb << " GB" << std::endl;

    return 0;
    

}


void test_ablation(){
    for (int i = 5; i < 13; i++){
        SetParamCKKS(1 << i, parameters);
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