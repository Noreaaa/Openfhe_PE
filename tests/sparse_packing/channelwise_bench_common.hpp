#pragma once
// Channel-wise baseline derived from Conv2d_C: one ciphertext per channel,
// row-major spatial indices retained across strided convolutions.
#include "../../src/cnn/conv2d.hpp"
#include "../../src/cnn/encrypted_relu.hpp"
#include "../../src/encryption/setparam.hpp"
#include "../../src/utils/dataloader.hpp"
#include "../../src/utils/experiment_phase.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sys/resource.h>
#include <omp.h>
using namespace lbcrypto;
using CT=Ciphertext<DCRTPoly>;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
struct Layout {int h,w,pitch,step;int pos(int h0,int w0)const{return h0*step*pitch+w0*step;}};
static std::vector<size_t> active(const Layout& l){std::vector<size_t> a;for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w)a.push_back(l.pos(h,w));return a;}
static void bnload(const std::string& root,const std::string& name,std::vector<double>& g,std::vector<double>& b,std::vector<double>& m,std::vector<double>& v){LoadConv2dBias(root+name+"_weight.npy",g);LoadConv2dBias(root+name+"_bias.npy",b);LoadConv2dBias(root+name+"_running_mean.npy",m);LoadConv2dBias(root+name+"_running_var.npy",v);}
static types::double3d plainconv(const types::double3d& x,types::double4d& f,int s,int p,std::vector<double>&g,std::vector<double>&b,std::vector<double>&m,std::vector<double>&v){auto y=GoldenConv2d(x,f,std::vector<double>(f.size(),0),s,p);GoldenBN(y,g,b,m,v,1e-5);return y;}
static double maxabs(const types::double3d& x){double r=0;for(auto&c:x)for(auto&row:c)for(auto v:row)r=std::max(r,std::abs(v));return r;}
static double polynomial(double x){double z=x*x;return .0229645+.5*x+z*(1.4340124+z*(-2.089185+z*1.1551725));}
static std::vector<CT> encrypt(const types::double3d& x,const Layout& l,uint32_t slots){std::vector<CT> out(x.size());
#pragma omp parallel for
for(int c=0;c<(int)x.size();++c){std::vector<double> v(slots);for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w)v[l.pos(h,w)]=x[c][h][w];out[c]=CRYPTOCONTEXT->Encrypt(KEYPAIR.publicKey,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(v,1,0,nullptr,slots));}return out;}
// Reuse rotations across output channels; each plaintext incorporates padding,
// sampling anchors, trained kernel and folded BN. No ciphertext channel mixing.
static std::vector<CT> convolution(const std::vector<CT>&x,Layout& l,types::double4d&f,int stride,int padding,std::vector<double>&g,std::vector<double>&b,std::vector<double>&m,std::vector<double>&v,uint32_t slots,double divisor){
 int kh=f[0][0].size(),kw=f[0][0][0].size(),ci=x.size(),co=f.size();
 Layout out{(l.h+2*padding-kh)/stride+1,(l.w+2*padding-kw)/stride+1,l.pitch,l.step*stride};
 std::vector<std::vector<CT>> rotated(ci,std::vector<CT>(kh*kw));
#pragma omp parallel for
 for(int c=0;c<ci;++c)for(int h=0;h<kh;++h)for(int w=0;w<kw;++w){int off=(h-padding)*l.step*l.pitch+(w-padding)*l.step;rotated[c][h*kw+w]=off?CRYPTOCONTEXT->EvalRotate(x[c],off):x[c];}
 std::vector<std::vector<int>> positions(kh*kw);
 for(int h=0;h<kh;++h)for(int w=0;w<kw;++w)for(int oh=0;oh<out.h;++oh)for(int ow=0;ow<out.w;++ow){int ih=oh*stride+h-padding,iw=ow*stride+w-padding;if(ih>=0&&ih<l.h&&iw>=0&&iw<l.w)positions[h*kw+w].push_back(out.pos(oh,ow));}
 std::vector<CT> y(co);
#pragma omp parallel for
 for(int oc=0;oc<co;++oc){CT sum;double scale=g[oc]/std::sqrt(v[oc]+1e-5)/divisor;
   for(int ic=0;ic<ci;++ic)for(int h=0;h<kh;++h)for(int w=0;w<kw;++w){std::vector<double> vals(slots,0);double k=f[oc][ic][h][w]*scale;for(int pos:positions[h*kw+w])vals[pos]=k;
     auto pt=CRYPTOCONTEXT->MakeCKKSPackedPlaintext(vals,1,x[ic]->GetLevel(),nullptr,slots);auto term=CRYPTOCONTEXT->EvalMult(rotated[ic][h*kw+w],pt);if(sum)CRYPTOCONTEXT->EvalAddInPlace(sum,term);else sum=term;
   }
   std::vector<double> bias(slots,0);double biasv=(b[oc]-g[oc]*m[oc]/std::sqrt(v[oc]+1e-5))/divisor;for(auto pos:active(out))bias[pos]=biasv;
   y[oc]=CRYPTOCONTEXT->EvalAdd(sum,CRYPTOCONTEXT->MakeCKKSPackedPlaintext(bias,sum->GetNoiseScaleDeg(),sum->GetLevel(),nullptr,slots));
 }
 l=out;return y;
}
struct Errors{double max=0,rmse=0;};
static Errors verify(const std::vector<CT>&cts,const Layout&l,const types::double3d&ref){long n=0;long double sq=0;Errors e;
 for(size_t c=0;c<cts.size();++c){Plaintext p;CRYPTOCONTEXT->Decrypt(KEYPAIR.secretKey,cts[c],&p);auto vals=p->GetRealPackedValue();for(int h=0;h<l.h;++h)for(int w=0;w<l.w;++w){double actual=vals.at(l.pos(h,w)),err=std::abs(actual-ref[c][h][w]);if(!std::isfinite(actual))throw std::runtime_error("nonfinite decrypted output");e.max=std::max(e.max,err);sq+=err*err;++n;}}
 e.rmse=std::sqrt(sq/n);return e;}
static void rotations(const std::vector<int>& pitches,const std::vector<int>&steps){std::set<int32_t> keys;for(int pitch:pitches)for(int step:steps)for(int h=-1;h<=1;++h)for(int w=-1;w<=1;++w)if(int o=h*step*pitch+w*step)keys.insert(o);CRYPTOCONTEXT->EvalRotateKeyGen(KEYPAIR.secretKey,std::vector<int32_t>(keys.begin(),keys.end()));}
