#include "../../src/cnn/sparse_conv2d.hpp"
#include "../../src/cnn/conv2d.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

int main() {
    try {
        for (auto config : {std::pair<uint32_t, uint32_t>{1024, 8}, {1024, 16}, {1024, 512}, {16384, 1024}}) {
            const auto ring = config.first, slots = config.second;
            CCParams<CryptoContextCKKSRNS> p;
            p.SetSecurityLevel(HEStd_NotSet); p.SetRingDim(ring);
            p.SetBatchSize(slots); p.SetMultiplicativeDepth(6);
            p.SetScalingModSize(40); p.SetFirstModSize(50);
            CRYPTOCONTEXT = GenCryptoContext(p);
            for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE}) CRYPTOCONTEXT->Enable(feature);
            KEYPAIR = CRYPTOCONTEXT->KeyGen();
            CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
            CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);
            for (int stride : {1, 2}) for (int kernel : {1, 3, 7}) {
                const int padding = kernel / 2;
                const int channels = slots==1024 ? 7 : 5, height = slots==1024 ? 9 : 4, width = slots==1024 ? 31 : 5, outputs = 4;
                const int encWs=slots==1024 ? 7 : 1, encWe=slots==1024 ? 22 : 3;
                double3d original(channels, types::double2d(height, std::vector<double>(width)));
                for (int c=0;c<channels;++c) for(int h=0;h<height;++h) for(int w=0;w<width;++w)
                    original[c][h][w] = (c*20+h*5+w-30)*0.01;
                auto pts = original;
                ENCRYPTED_HEIGHT_START=1; ENCRYPTED_HEIGHT_END=2;
                ENCRYPTED_WIDTH_START=encWs; ENCRYPTED_WIDTH_END=encWe;
                RESCALE_REQUIRED=false;
                types::vector2d<Ciphertext<DCRTPoly>> cts(2), result;
                int per=slots/width;
                for(int h=1;h<=2;++h) for(int start=0;start<channels;start+=per) {
                    std::vector<double> values(slots,0);
                    for(int c=start;c<std::min(channels,start+per);++c) for(int w=encWs;w<=encWe;++w) {
                        values[(c-start)*width+w]=pts[c][h][w]; pts[c][h][w]=0;
                    }
                    cts[h-1].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
                        CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
                }
                types::double4d weights(outputs, double3d(channels, types::double2d(kernel,std::vector<double>(kernel))));
                for(int o=0;o<outputs;++o) for(int c=0;c<channels;++c) for(int h=0;h<kernel;++h) for(int w=0;w<kernel;++w)
                    weights[o][c][h][w]=(o+c+h-w+1)*0.02;
                std::vector<double> gamma(outputs,1.2),beta(outputs,-0.1),mean(outputs,0.03),var(outputs,0.8),bias(outputs,0.07);
                auto oldCts = cts; auto oldPts = pts;
                SparseConv2dBN_P layer(CONV_2D,"test",ring,slots,weights,stride,padding,gamma,beta,mean,var,1e-5,bias,CONV_2D_BN);
                layer.EnableFastPath(width);
                double3d y;
                layer.forward(cts,pts,result,y);
                // Keep packed output and public entries separate for the next layer.
                auto chainedCts = result; auto chainedPts = y;
                auto reconstruct = [&](const types::vector2d<Ciphertext<DCRTPoly>>& encrypted, double3d publicValues) {
                    int widthOut = publicValues[0][0].size(), perOut = slots / widthOut;
                    for (int h=ENCRYPTED_HEIGHT_START; h<=ENCRYPTED_HEIGHT_END; ++h)
                        for (size_t block=0; block<encrypted[h-ENCRYPTED_HEIGHT_START].size(); ++block) {
                            Plaintext decoded;
                            CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey, encrypted[h-ENCRYPTED_HEIGHT_START][block], &decoded);
                            decoded->SetLength(slots);
                            auto values = decoded->GetRealPackedValue();
                            for (int c=block*perOut; c<std::min(static_cast<int>(publicValues.size()), static_cast<int>((block+1)*perOut)); ++c)
                                for (int w=ENCRYPTED_WIDTH_START; w<=ENCRYPTED_WIDTH_END; ++w)
                                    publicValues[c][h][w] = values[(c%perOut)*widthOut+w];
                        }
                    return publicValues;
                };
                if (slots == 512) {
                    ENCRYPTED_HEIGHT_START=1; ENCRYPTED_HEIGHT_END=2;
                    ENCRYPTED_WIDTH_START=encWs; ENCRYPTED_WIDTH_END=encWe;
                    Conv2dBN_P oldLayer(CONV_2D, "legacy", weights, stride, padding, slots,
                        gamma, beta, mean, var, 1e-5, bias, CONV_2D_BN);
                    types::vector2d<Ciphertext<DCRTPoly>> oldResult; double3d oldY;
                    oldLayer.forward(oldCts, oldPts, oldResult, oldY);
                    auto legacy = reconstruct(oldResult, oldY);
                    auto sparse = reconstruct(result, y);
                    double difference = 0;
                    for (size_t c=0;c<sparse.size();++c) for(size_t h=0;h<sparse[c].size();++h)
                        for(size_t w=0;w<sparse[c][h].size();++w) {
                            if (!std::isfinite(legacy[c][h][w])) throw std::runtime_error("non-finite legacy output");
                            difference=std::max(difference,std::abs(legacy[c][h][w]-sparse[c][h][w]));
                        }
                    if (difference>1e-3) throw std::runtime_error("legacy/sparse mismatch");
                    std::cout << "legacy_difference=" << difference << '\n';
                }
                int ow=y[0][0].size(), outPer=slots/ow;
                for(int h=ENCRYPTED_HEIGHT_START;h<=ENCRYPTED_HEIGHT_END;++h)
                    for(size_t block=0;block<result[h-ENCRYPTED_HEIGHT_START].size();++block) {
                        Plaintext decoded; CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,result[h-ENCRYPTED_HEIGHT_START][block],&decoded);
                        decoded->SetLength(slots); auto values=decoded->GetRealPackedValue();
                        for(int c=block*outPer;c<std::min(outputs,static_cast<int>((block+1)*outPer));++c)
                            for(int w=ENCRYPTED_WIDTH_START;w<=ENCRYPTED_WIDTH_END;++w) y[c][h][w]=values[(c%outPer)*ow+w];
                    }
                double maxError=0;
                for(int o=0;o<outputs;++o) for(size_t h=0;h<y[o].size();++h) for(int w=0;w<ow;++w) {
                    double expected=bias[o];
                    for(int c=0;c<channels;++c) for(int kh=0;kh<kernel;++kh) for(int kw=0;kw<kernel;++kw) {
                        int ih=h*stride+kh-padding, iw=w*stride+kw-padding;
                        if(ih>=0&&ih<height&&iw>=0&&iw<width) expected+=original[c][ih][iw]*weights[o][c][kh][kw];
                    }
                    expected=(expected-mean[o])*gamma[o]/std::sqrt(var[o]+1e-5)+beta[o];
                    if(!std::isfinite(y[o][h][w])) throw std::runtime_error("non-finite output");
                    maxError=std::max(maxError,std::abs(expected-y[o][h][w]));
                }
                if(maxError>1e-3) throw std::runtime_error("sparse convolution mismatch");
                std::cout << "slots=" << slots << " stride=" << stride << " kernel=" << kernel << " max_error=" << maxError << '\n';
                // A second 3x3 convolution consumes the first layer's sparse ciphertexts.
                types::double4d secondWeights(outputs, double3d(outputs, types::double2d(3, std::vector<double>(3, 0.02))));
                SparseConv2dBN_P second(CONV_2D, "second", ring, slots, secondWeights, 1, 1,
                    gamma, beta, mean, var, 1e-5, bias, CONV_2D_BN);
                second.EnableFastPath(ow);
                auto expectedSecond = GoldenConv2d(y, secondWeights, bias, 1, 1);
                GoldenBN(expectedSecond, gamma, beta, mean, var, 1e-5);
                types::vector2d<Ciphertext<DCRTPoly>> secondCts; double3d secondPts;
                second.forward(chainedCts, chainedPts, secondCts, secondPts);
                auto actualSecond = reconstruct(secondCts, secondPts);
                double chainError = 0;
                for (int c=0;c<outputs;++c) for(size_t h=0;h<actualSecond[c].size();++h)
                    for(size_t w=0;w<actualSecond[c][h].size();++w) {
                        if (!std::isfinite(actualSecond[c][h][w])) throw std::runtime_error("non-finite chained output");
                        chainError=std::max(chainError,std::abs(expectedSecond[c][h][w]-actualSecond[c][h][w]));
                    }
                if(chainError>1e-3) throw std::runtime_error("chained convolution mismatch");
                std::cout << "chain_error=" << chainError << '\n';
            }
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
