// Channel-wise Conv+BN + normalized polynomial + parallel genuine CKKS bootstrap.
#include "channelwise_bench_common.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/experiment_phase.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
#include <omp.h>
#include <exception>

int main(int argc,char**argv){try{
 std::string root=argc>1?argv[1]:".";int target=argc>2?std::stoi(argv[2]):3;uint32_t ring=argc>3?std::stoul(argv[3]):16384,slots=argc>4?std::stoul(argv[4]):ring/2;int depth=argc>5?std::stoi(argv[5]):30;std::string profile=argc>6?argv[6]:"channelwise_shape.csv";
 int outputLimit=argc>7?std::stoi(argv[7]):0;
 if(target<1||target>18)throw std::invalid_argument("invalid layer");omp_set_dynamic(0);omp_set_num_threads(96);
 struct Spec{std::string w,bn;int s,p;};std::vector<Spec> specs={{"conv1","bn1",2,3},{"maxpool_0","maxpool_1",2,1}};
 for(int stage=1;stage<=4;++stage)for(int block=0;block<2;++block)for(int conv=1;conv<=2;++conv){std::string name="layer"+std::to_string(stage)+"_"+std::to_string(block);specs.push_back({name+"_conv"+std::to_string(conv),name+"_bn"+std::to_string(conv),(stage>1&&block==0&&conv==1)?2:1,1});}
 const std::string wr=root+"/python_model/saved_models/ResNet-18/parameters/";WriteExperimentPhase(profile,"reference_setup",target);
 auto input=load_bin_image_double(root+"/datasets/imagenet-mini/ILSVRC2012_val_00000029.bin");std::vector<double>g,b,m,v;types::double4d f;
 for(int i=0;i<target;++i){auto spec=specs[i];f=LoadConv2dWeight(wr+spec.w+"_weight.npy");LoadConv2dBias(wr+spec.bn+"_weight.npy",g);LoadConv2dBias(wr+spec.bn+"_bias.npy",b);LoadConv2dBias(wr+spec.bn+"_running_mean.npy",m);LoadConv2dBias(wr+spec.bn+"_running_var.npy",v);if(i==target-1)break;input=GoldenConv2d(input,f,std::vector<double>(f.size(),0),spec.s,spec.p);GoldenBN(input,g,b,m,v,1e-5);for(auto&c:input)for(auto&row:c)for(auto&x:row)x=std::max(0.,x);}
 int ci=input.size(),height=input[0].size(),width=input[0][0].size(),co=f.size();auto spec=specs[target-1];
 if(static_cast<uint64_t>(height)*width>slots)throw std::invalid_argument("spatial feature map does not fit one channel ciphertext");
 std::cout<<"SHAPE_CONFIG layer="<<target<<" weight="<<spec.w<<" input="<<ci<<'x'<<height<<'x'<<width<<" output_channels="<<co<<" kernel="<<f[0][0].size()<<" stride="<<spec.s<<" padding="<<spec.p<<" ring="<<ring<<" slots="<<slots<<" depth="<<depth<<" input_level=0 input_degree=1 fully_encrypted=1 packing=contiguous_per_channel"<<std::endl;
 if(outputLimit>0&&outputLimit<co){co=outputLimit;f.resize(co);g.resize(co);b.resize(co);m.resize(co);v.resize(co);}
 std::vector<double>bias(co,0);auto pre=plainconv(input,f,spec.s,spec.p,g,b,m,v);double R=std::max(1.,1.1*maxabs(pre));auto expectedConv=pre,expectedPoly=pre,exactRelu=pre;
 for(int c=0;c<co;++c)for(size_t h=0;h<pre[c].size();++h)for(size_t w=0;w<pre[c][h].size();++w){double value=pre[c][h][w];expectedConv[c][h][w]=value/R;expectedPoly[c][h][w]=R*polynomial(value/R);exactRelu[c][h][w]=std::max(0.,value);}
 double B=std::max(1.,1.1*maxabs(expectedPoly));std::vector<uint32_t>budget{3,3};int reserve=FHECKKSRNS::GetBootstrapDepth(budget,UNIFORM_TERNARY);
 std::cout<<"PIPELINE_CONFIG scale_bits=59 first_bits=60 bootstrap_budget=3,3 bootstrap_reserve="<<reserve<<" output_limit="<<outputLimit<<" output_channels="<<co<<" parallel_bootstrap_workers="<<std::min(co,96)<<" R="<<R<<" B="<<B<<std::endl;
 WriteExperimentPhase(profile,"key_setup",target);auto t=Clock::now();CCParams<CryptoContextCKKSRNS> p;SetParamCKKS(ring,p,slots);p.SetMultiplicativeDepth(depth);p.SetScalingModSize(59);p.SetFirstModSize(60);CRYPTOCONTEXT=GenCryptoContext(p);for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,FHE})CRYPTOCONTEXT->Enable(feature);CRYPTOCONTEXT->EvalBootstrapSetup(budget,{0,0},slots,0);KEYPAIR=CRYPTOCONTEXT->KeyGen();CRYPTOCONTEXT->EvalMultKeyGen(KEYPAIR.secretKey);CRYPTOCONTEXT->EvalBootstrapKeyGen(KEYPAIR.secretKey,slots);rotations({width},{1});double keyms=elapsed(t);
 Layout layout{height,width,width,1};WriteExperimentPhase(profile,"input_encryption",target);t=Clock::now();auto x=encrypt(input,layout,slots);double encms=elapsed(t);
 WriteExperimentPhase(profile,"server_conv",target);t=Clock::now();auto y=convolution(x,layout,f,spec.s,spec.p,g,b,m,v,slots,R);double convms=elapsed(t);x.clear();WriteExperimentPhase(profile,"verification",target);auto ce=verify(y,layout,expectedConv);if(ce.rmse>.001)throw std::runtime_error("convolution arithmetic check failed");
 auto indices=active(layout);std::vector<CT> activated(co);std::exception_ptr failure;
 WriteExperimentPhase(profile,"server_relu",target);t=Clock::now();
#pragma omp parallel for
 for(int c=0;c<co;++c){try{auto result=CRYPTOCONTEXT->EvalPoly(y[c],{0.,.5*R,1.4340124*R,0.,-2.089185*R,0.,1.1551725*R});std::vector<double>constant(slots,0);for(auto pos:indices)constant[pos]=.0229645*R;activated[c]=CRYPTOCONTEXT->EvalAdd(result,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(constant,result->GetNoiseScaleDeg(),result->GetLevel(),nullptr,slots));}catch(...){
#pragma omp critical(channelwise_pipeline_failure)
 {if(!failure)failure=std::current_exception();}}}
 if(failure)std::rethrow_exception(failure);double polyms=elapsed(t);y.clear();WriteExperimentPhase(profile,"verification",target);auto pe=verify(activated,layout,expectedPoly);if(pe.rmse>.001*std::max(1.,maxabs(expectedPoly)))throw std::runtime_error("polynomial arithmetic check failed");
 auto beforeLevel=activated[0]->GetLevel();std::vector<CT> refreshed(co);
 WriteExperimentPhase(profile,"server_bootstrap",target);t=Clock::now();
#pragma omp parallel for
 for(int c=0;c<co;++c){try{auto normalized=CRYPTOCONTEXT->EvalMult(activated[c],1./B);if(normalized->GetNoiseScaleDeg()>1)CRYPTOCONTEXT->GetScheme()->ModReduceInternalInPlace(normalized,normalized->GetNoiseScaleDeg()-1);auto towers=normalized->GetElements()[0].GetNumOfElements();if(towers>2)CRYPTOCONTEXT->GetScheme()->LevelReduceInternalInPlace(normalized,towers-2);if(normalized->GetElements()[0].GetNumOfElements()!=2)throw std::runtime_error("bootstrap input not depleted");auto result=CRYPTOCONTEXT->EvalBootstrap(normalized,1,0);if(result->GetElements()[0].GetNumOfElements()<=2)throw std::runtime_error("bootstrap did not restore towers");refreshed[c]=CRYPTOCONTEXT->EvalMult(result,B);}catch(...){
#pragma omp critical(channelwise_pipeline_failure)
 {if(!failure)failure=std::current_exception();}}}
 if(failure)std::rethrow_exception(failure);double bootms=elapsed(t);activated.clear();WriteExperimentPhase(profile,"verification",target);auto be=verify(refreshed,layout,expectedPoly),re=verify(refreshed,layout,exactRelu);auto ct=refreshed[0];rusage usage{};getrusage(RUSAGE_SELF,&usage);
 std::ofstream csv(profile);csv<<"layer,ring,slots,depth,input_channels,input_height,input_width,output_channels,output_height,output_width,kernel,stride,key_setup_ms,encryption_ms,conv_ms,activation_ms,bootstrap_ms,operator_total_ms,pipeline_ms,ciphertexts,pre_bootstrap_level,output_level,towers,R,B,conv_max_abs,conv_rmse,poly_max_abs,poly_rmse,max_abs,rmse,vs_relu_rmse,whole_peak_gib\n"<<std::setprecision(12)<<target<<','<<ring<<','<<slots<<','<<depth<<','<<ci<<','<<height<<','<<width<<','<<co<<','<<layout.h<<','<<layout.w<<','<<f[0][0].size()<<','<<spec.s<<','<<keyms<<','<<encms<<','<<convms<<','<<polyms<<','<<bootms<<','<<convms+polyms+bootms<<','<<keyms+encms+convms+polyms+bootms<<','<<co<<','<<beforeLevel<<','<<ct->GetLevel()<<','<<ct->GetElements()[0].GetNumOfElements()<<','<<R<<','<<B<<','<<ce.max<<','<<ce.rmse<<','<<pe.max<<','<<pe.rmse<<','<<be.max<<','<<be.rmse<<','<<re.rmse<<','<<usage.ru_maxrss/1048576.<<'\n';csv.flush();
 std::cout<<"PIPELINE_RESULT layer="<<target<<" ring="<<ring<<" conv_s="<<convms/1000<<" polynomial_s="<<polyms/1000<<" bootstrap_s="<<bootms/1000<<" peak_gib="<<usage.ru_maxrss/1048576.<<" bootstrap_rmse="<<be.rmse<<" vs_relu_rmse="<<re.rmse<<std::endl;
 if(be.rmse>.005*std::max(1.,maxabs(expectedPoly)))throw std::runtime_error("post-bootstrap arithmetic check failed");WriteExperimentPhase(profile,"finished",target);return 0;
}catch(const std::exception&e){std::cerr<<"PIPELINE_ERROR "<<e.what()<<std::endl;return 1;}}
