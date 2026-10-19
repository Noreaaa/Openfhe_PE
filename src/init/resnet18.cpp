#include "resnet18.hpp"
#include "../cnn/sparse_conv2d.hpp"
#include "../cnn/conv2d.hpp"
#include "../cnn/activation.hpp"
#include "../cnn/sparse_activation.hpp"
#include "../encryption/encryption.hpp"
#include "../utils/dataloader.hpp"
#include <chrono>
#include <set>
#include <tuple>
#include <stdexcept>

using namespace lbcrypto;

void InitResNet18(Network& model, const std::string& root, const std::string& imageRoot,
    uint32_t slots, double eps, int actType, LWEPrivateKey fheKey, KeyPair<DCRTPoly>& keyPair,
    int hs, int he, int ws, int we, int convLayers, const std::string& profileCsv, bool verify) {
    if(convLayers<1 || convLayers>10) throw std::invalid_argument("ResNet convolution layers must be 1..10");
    struct Spec { const char* weight; const char* bn; int stride, padding; };
    static const Spec specs[] = {
        {"conv1","bn1",2,3}, {"maxpool_0","maxpool_1",2,1},
        {"layer1_0_conv1","layer1_0_bn1",1,1}, {"layer1_0_conv2","layer1_0_bn2",1,1},
        {"layer1_1_conv1","layer1_1_bn1",1,1}, {"layer1_1_conv2","layer1_1_bn2",1,1},
        {"layer2_0_conv1","layer2_0_bn1",2,1}, {"layer2_0_conv2","layer2_0_bn2",1,1},
        {"layer2_1_conv1","layer2_1_bn1",1,1}, {"layer2_1_conv2","layer2_1_bn2",1,1}
    };
    auto input=load_bin_image_double(imageRoot+"ILSVRC2012_val_00000029.bin");
    if(hs<0 || he<hs || he>=static_cast<int>(input[0].size()) ||
       ws<0 || we<ws || we>=static_cast<int>(input[0][0].size()))
        throw std::invalid_argument("ResNet encrypted region is outside the input image");
    int width=input[0][0].size(), channels=input.size();
    std::set<std::tuple<int,int,int,int,int>> prepared;
    CONV_BIAS=false; USE_COMPACT=false;
    for(int i=0;i<convLayers;++i) {
        const auto& spec=specs[i];
        auto weights=LoadConv2dWeight(root+spec.weight+"_weight.npy");
        if(weights.empty() || weights[0].size()!=static_cast<size_t>(channels))
            throw std::runtime_error("ResNet convolution channel mismatch at layer "+std::to_string(i+1));
        std::vector<double> gamma,beta,mean,var;
        const std::string prefix=root+spec.bn;
        LoadConv2dBias(prefix+"_weight.npy",gamma); LoadConv2dBias(prefix+"_bias.npy",beta);
        LoadConv2dBias(prefix+"_running_mean.npy",mean); LoadConv2dBias(prefix+"_running_var.npy",var);
        auto conv=std::make_shared<SparseConv2dBN_P>(CONV_2D,"conv_bn"+std::to_string(i+1),
            CRYPTOCONTEXT->GetRingDimension(),slots,weights,spec.stride,spec.padding,
            gamma,beta,mean,var,eps,std::vector<double>(weights.size(),0.0),CONV_2D_BN);
        if(actType==0) {
            auto key=std::make_tuple(width,weights.size(),weights[0][0][0].size(),spec.stride,spec.padding);
            conv->EnableFastPath(width,prepared.insert(key).second);
        }
        model.add_layer(conv);
        if(actType==0) model.add_layer(std::make_shared<sparse_activation>(RELU_SS_ACTIVATION,
            "relu"+std::to_string(i+1),fheKey,CRYPTOCONTEXT->GetRingDimension(),slots));
        else model.add_layer(std::make_shared<Relu_appx>(RELU_APPX_ACTIVATION,"relu"+std::to_string(i+1)));
        int nextWidth=(width+2*spec.padding-weights[0][0][0].size())/spec.stride+1;
        std::cout<<"RESNET_LAYER index="<<i+1<<" weight="<<spec.weight<<" input="<<channels<<'x'<<width<<'x'<<width
                 <<" output="<<weights.size()<<'x'<<nextWidth<<'x'<<nextWidth<<" stride="<<spec.stride<<std::endl;
        channels=weights.size(); width=nextWidth;
    }
    if(verify) model.set_feature_observer(
        [reference=input,root,eps,slots](int count,const types::vector2d<Ciphertext<DCRTPoly>>& cts,
                                       const double3d& publicValues) mutable {
            const auto& spec=specs[count-1];
            auto weights=LoadConv2dWeight(root+spec.weight+"_weight.npy");
            std::vector<double> gamma,beta,mean,var;
            const std::string prefix=root+spec.bn;
            LoadConv2dBias(prefix+"_weight.npy",gamma); LoadConv2dBias(prefix+"_bias.npy",beta);
            LoadConv2dBias(prefix+"_running_mean.npy",mean); LoadConv2dBias(prefix+"_running_var.npy",var);
            reference=GoldenConv2d(reference,weights,std::vector<double>(weights.size(),0),spec.stride,spec.padding);
            GoldenBN(reference,gamma,beta,mean,var,eps);
            for(auto& channel:reference) for(auto& row:channel) for(auto& value:row) value=std::max(0.0,value);
            auto actual=publicValues;
            const int width=actual[0][0].size(),per=slots/width;
            for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h)
                for(size_t block=0;block<cts[h-ENCRYPTED_HEIGHT_START].size();++block) {
                    Plaintext decoded;
                    CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,cts[h-ENCRYPTED_HEIGHT_START][block],&decoded);
                    if(cts[h-ENCRYPTED_HEIGHT_START][block]->GetSlots()!=slots)
                        throw std::runtime_error("ResNet prefix changed logical slots");
                    decoded->SetLength(slots); const auto values=decoded->GetRealPackedValue();
                    for(int c=block*per;c<std::min(static_cast<int>(actual.size()),static_cast<int>((block+1)*per));++c)
                        for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w)
                            actual[c][h][w]=values[(c%per)*width+w];
                }
            double maxError=0; long double squareError=0; size_t values=0;
            for(size_t c=0;c<reference.size();++c) for(size_t h=0;h<reference[c].size();++h)
                for(size_t w=0;w<reference[c][h].size();++w) {
                    if(!std::isfinite(actual[c][h][w])) throw std::runtime_error("Non-finite ResNet prefix output");
                    const double error=std::abs(actual[c][h][w]-reference[c][h][w]);
                    maxError=std::max(maxError,error); squareError+=error*error; ++values;
                }
            std::cout<<"PREFIX_ACCURACY layers="<<count<<" max_abs="<<maxError
                     <<" rmse="<<std::sqrt(squareError/values)<<std::endl;
        });
    ENCRYPTED_HEIGHT_START=hs; ENCRYPTED_HEIGHT_END=he;
    ENCRYPTED_WIDTH_START=ws; ENCRYPTED_WIDTH_END=we;
    types::vector2d<Ciphertext<DCRTPoly>> encrypted;
    Encrypt_MCSR_P(input,slots,CRYPTOCONTEXT,hs,he,ws,we,keyPair,encrypted);
    auto start=std::chrono::steady_clock::now();
    model.predict_P(encrypted,input,profileCsv);
    std::cout<<"prediction time: "<<std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count()<<"ms"<<std::endl;
}
