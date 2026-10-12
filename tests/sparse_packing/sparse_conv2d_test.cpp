#include "../../src/cnn/sparse_conv2d.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

int main() {
    try {
        for (uint32_t slots : {8u, 16u}) {
            CCParams<CryptoContextCKKSRNS> p;
            p.SetSecurityLevel(HEStd_NotSet); p.SetRingDim(1024);
            p.SetBatchSize(slots); p.SetMultiplicativeDepth(6);
            p.SetScalingModSize(40); p.SetFirstModSize(50);
            CRYPTOCONTEXT = GenCryptoContext(p);
            for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE}) CRYPTOCONTEXT->Enable(feature);
            KEYPAIR = CRYPTOCONTEXT->KeyGen();
            CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);
            CRYPTOCONTEXT->EvalSumKeyGen(KEYPAIR.secretKey);
            for (int stride : {1, 2}) {
                const int channels = 5, height = 4, width = 5, outputs = 4;
                double3d original(channels, types::double2d(height, std::vector<double>(width)));
                for (int c=0;c<channels;++c) for(int h=0;h<height;++h) for(int w=0;w<width;++w)
                    original[c][h][w] = (c*20+h*5+w-30)*0.01;
                auto pts = original;
                ENCRYPTED_HEIGHT_START=1; ENCRYPTED_HEIGHT_END=2;
                ENCRYPTED_WIDTH_START=1; ENCRYPTED_WIDTH_END=3;
                RESCALE_REQUIRED=false;
                types::vector2d<Ciphertext<DCRTPoly>> cts(2), result;
                int per=slots/width;
                for(int h=1;h<=2;++h) for(int start=0;start<channels;start+=per) {
                    std::vector<double> values(slots,0);
                    for(int c=start;c<std::min(channels,start+per);++c) for(int w=1;w<=3;++w) {
                        values[(c-start)*width+w]=pts[c][h][w]; pts[c][h][w]=0;
                    }
                    cts[h-1].push_back(CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,
                        CRYPTOCONTEXT->MakeCKKSPackedPlaintext(values,1,0,nullptr,slots)));
                }
                types::double4d weights(outputs, double3d(channels, types::double2d(3,std::vector<double>(3))));
                for(int o=0;o<outputs;++o) for(int c=0;c<channels;++c) for(int h=0;h<3;++h) for(int w=0;w<3;++w)
                    weights[o][c][h][w]=(o+c+h-w+1)*0.02;
                std::vector<double> gamma(outputs,1.2),beta(outputs,-0.1),mean(outputs,0.03),var(outputs,0.8),bias(outputs,0.07);
                SparseConv2dBN_P layer(CONV_2D,"test",1024,slots,weights,stride,1,gamma,beta,mean,var,1e-5,bias,CONV_2D_BN);
                double3d y;
                layer.forward(cts,pts,result,y);
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
                    for(int c=0;c<channels;++c) for(int kh=0;kh<3;++kh) for(int kw=0;kw<3;++kw) {
                        int ih=h*stride+kh-1, iw=w*stride+kw-1;
                        if(ih>=0&&ih<height&&iw>=0&&iw<width) expected+=original[c][ih][iw]*weights[o][c][kh][kw];
                    }
                    expected=(expected-mean[o])*gamma[o]/std::sqrt(var[o]+1e-5)+beta[o];
                    if(!std::isfinite(y[o][h][w])) throw std::runtime_error("non-finite output");
                    maxError=std::max(maxError,std::abs(expected-y[o][h][w]));
                }
                if(maxError>1e-3) throw std::runtime_error("sparse convolution mismatch");
                std::cout << "slots=" << slots << " stride=" << stride << " max_error=" << maxError << '\n';
            }
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
