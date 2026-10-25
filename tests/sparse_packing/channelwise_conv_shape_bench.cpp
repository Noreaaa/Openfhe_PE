// Fully encrypted channel-wise shape microbenchmark; same convolution helper as the prefix baseline.
#include "channelwise_bench_common.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/experiment_phase.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <omp.h>

int main(int argc,char**argv){try{
 std::string root=argc>1?argv[1]:".";int target=argc>2?std::stoi(argv[2]):3;uint32_t ring=argc>3?std::stoul(argv[3]):16384,slots=argc>4?std::stoul(argv[4]):ring/2;int depth=argc>5?std::stoi(argv[5]):30;std::string profile=argc>6?argv[6]:"channelwise_shape.csv";
 if(target<1||target>18)throw std::invalid_argument("invalid layer");omp_set_dynamic(0);omp_set_num_threads(96);
 struct Spec{std::string w,bn;int s,p;};std::vector<Spec> specs={{"conv1","bn1",2,3},{"maxpool_0","maxpool_1",2,1}};
 for(int stage=1;stage<=4;++stage)for(int block=0;block<2;++block)for(int conv=1;conv<=2;++conv){std::string name="layer"+std::to_string(stage)+"_"+std::to_string(block);specs.push_back({name+"_conv"+std::to_string(conv),name+"_bn"+std::to_string(conv),(stage>1&&block==0&&conv==1)?2:1,1});}
 const std::string wr=root+"/python_model/saved_models/ResNet-18/parameters/";WriteExperimentPhase(profile,"reference_setup",target);
 auto input=load_bin_image_double(root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin");std::vector<double>g,b,m,v;types::double4d f;
 for(int i=0;i<target;++i){auto spec=specs[i];f=LoadConv2dWeight(wr+spec.w+"_weight.npy");LoadConv2dBias(wr+spec.bn+"_weight.npy",g);LoadConv2dBias(wr+spec.bn+"_bias.npy",b);LoadConv2dBias(wr+spec.bn+"_running_mean.npy",m);LoadConv2dBias(wr+spec.bn+"_running_var.npy",v);if(i==target-1)break;input=GoldenConv2d(input,f,std::vector<double>(f.size(),0),spec.s,spec.p);GoldenBN(input,g,b,m,v,1e-5);for(auto&c:input)for(auto&row:c)for(auto&x:row)x=std::max(0.,x);}
 int ci=input.size(),height=input[0].size(),width=input[0][0].size(),co=f.size();auto spec=specs[target-1];
 if(static_cast<uint64_t>(height)*width>slots)throw std::invalid_argument("spatial feature map does not fit one channel ciphertext");
 std::cout<<"SHAPE_CONFIG layer="<<target<<" weight="<<spec.w<<" input="<<ci<<'x'<<height<<'x'<<width<<" output_channels="<<co<<" kernel="<<f[0][0].size()<<" stride="<<spec.s<<" padding="<<spec.p<<" ring="<<ring<<" slots="<<slots<<" depth="<<depth<<" input_level=0 input_degree=1 fully_encrypted=1 packing=contiguous_per_channel"<<std::endl;
 WriteExperimentPhase(profile,"key_setup",target);CCParams<CryptoContextCKKSRNS> p;SetParamCKKS(ring,p,slots);p.SetMultiplicativeDepth(depth);p.SetScalingModSize(30);p.SetFirstModSize(40);CRYPTOCONTEXT=GenCryptoContext(p);for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE})CRYPTOCONTEXT->Enable(feature);KEYPAIR=CRYPTOCONTEXT->KeyGen();CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);rotations({width},{1});RESCALE_REQUIRED=false;USE_COMPACT=false;CONV_BIAS=false;
 Layout layout{height,width,width,1};
 WriteExperimentPhase(profile,"input_encryption",target);auto x=encrypt(input,layout,slots);
 std::vector<double>bias(co,0);
 WriteExperimentPhase(profile,"server_conv",target);auto t=Clock::now();auto y=convolution(x,layout,f,spec.s,spec.p,g,b,m,v,slots,1.0);double convms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();WriteExperimentPhase(profile,"verification",target);
 int oh=layout.h,ow=layout.w;size_t ciphertexts=y.size();
 auto golden=GoldenConv2d(input,f,bias,spec.s,spec.p);GoldenBN(golden,g,b,m,v,1e-5);auto errors=verify(y,layout,golden);double error=errors.max,rmse=errors.rmse;
 auto ct=y[0];rusage usage{};getrusage(RUSAGE_SELF,&usage);
 std::ofstream csv(profile);csv<<"layer,ring,slots,depth,input_channels,input_height,input_width,output_channels,output_height,output_width,kernel,stride,encrypted_height,encrypted_width,conv_ms,ciphertexts,output_level,towers,max_abs,rmse,whole_peak_gib\n"<<std::setprecision(12)<<target<<','<<ring<<','<<slots<<','<<depth<<','<<ci<<','<<height<<','<<width<<','<<co<<','<<oh<<','<<ow<<','<<f[0][0].size()<<','<<spec.s<<','<<oh<<','<<ow<<','<<convms<<','<<ciphertexts<<','<<ct->GetLevel()<<','<<ct->GetElements()[0].GetNumOfElements()<<','<<error<<','<<rmse<<','<<usage.ru_maxrss/1048576.<<'\n';csv.flush();
 std::cout<<"SHAPE_RESULT layer="<<target<<" ring="<<ring<<" conv_s="<<convms/1000<<" peak_gib="<<usage.ru_maxrss/1048576.<<" max_abs="<<error<<" rmse="<<rmse<<std::endl;
 if(error>.02||rmse>.001)throw std::runtime_error("channel-wise convolution numeric check failed");WriteExperimentPhase(profile,"finished",target);return 0;
}catch(const std::exception&e){std::cerr<<"SHAPE_ERROR "<<e.what()<<std::endl;return 1;}}
