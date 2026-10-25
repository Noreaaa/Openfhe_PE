#!/usr/bin/env python3
"""Forward binary dependency masks through the project's modified full ResNet."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
PARAM=ROOT/'python_model/saved_models/ResNet-18/parameters'
BASE=ROOT/'benchmarks/resnet_client_first_slots8192_20261024'
OUT=BASE/'figures/full_resnet18'
OUT.mkdir(parents=True,exist_ok=True)

def conv(mask,name,stride,pad):
    # Positive unit kernel: any nonzero input makes the output sensitive.
    shape=np.load(PARAM/(name+'_weight.npy'),mmap_mode='r').shape
    kh,kw=shape[-2:]
    padded=np.pad(mask,((pad,pad),(pad,pad)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,(kh,kw))[::stride,::stride]
    return windows.any(axis=(-2,-1))

records=[];series={};saved={};labels={0:'Input'}
for area in range(10,81,10):
    mask=np.zeros((224,224),dtype=bool);start=(224-area)//2
    mask[start:start+area,start:start+area]=True
    values=[];idx=0
    def record(name,m):
        labels[idx]=name;values.append(float(m.mean()*100))
        saved[f'area{area}_layer{idx}']=m
        ys,xs=np.nonzero(m)
        records.append(dict(initial_area=area,layer=idx,name=name,height=m.shape[0],width=m.shape[1],nonzero=int(m.sum()),total=m.size,percent=float(m.mean()*100),h_start=int(ys.min()),h_end=int(ys.max()),w_start=int(xs.min()),w_end=int(xs.max())))
    record('Input',mask)
    for name,stride,pad in [('conv1',2,3),('maxpool_0',2,1)]:
        mask=conv(mask,name,stride,pad);idx+=1;record(name,mask)
    for stage in range(1,5):
        for block in range(2):
            stride=2 if stage>1 and block==0 else 1
            identity=mask.copy()
            prefix=f'layer{stage}_{block}'
            mask=conv(mask,prefix+'_conv1',stride,1);idx+=1;record(prefix+'_conv1',mask)
            mask=conv(mask,prefix+'_conv2',1,1)
            if stride==2:identity=conv(identity,prefix+'_downsample_0',2,0)
            assert mask.shape==identity.shape
            mask=np.logical_or(mask,identity)
            idx+=1;record(prefix+'_conv2 + residual',mask)
    assert idx==18
    # Existing measured prefix must match mask-forward results exactly.
    with (BASE/f'area{area:02d}/layers.csv').open() as f:
        for r in csv.DictReader(f):
            m=saved[f"area{area}_layer{r['conv_layers']}"]
            expected=np.zeros(m.shape,dtype=bool)
            expected[int(r['encrypted_h_start']):int(r['encrypted_h_end'])+1,int(r['encrypted_w_start']):int(r['encrypted_w_end'])+1]=True
            assert np.array_equal(m,expected),(area,r['conv_layers'])
    series[area]=values
np.savez_compressed(OUT/'binary_masks.npz',**saved)
with (OUT/'coverage_geometry.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
with (OUT/'coverage_percent.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['Layer','Operation']+[f'{a}x{a} (%)' for a in series])
    for i in range(19):w.writerow([i,labels[i]]+[round(v[i],6) for v in series.values()])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.labelcolor':'#5f6368','xtick.color':'#5f6368','ytick.color':'#5f6368'})
fig,ax=plt.subplots(figsize=(14,7.5),facecolor='white');fig.subplots_adjust(left=.08,right=.84,top=.79,bottom=.25)
colors=['#4285F4','#DB4437','#F4B400','#0F9D58','#AB47BC','#00ACC1','#FF7043','#5C6BC0']
for (area,v),c in zip(series.items(),colors):ax.plot(range(19),v,color=c,lw=2,marker='o',ms=3.5,label=f'{area} × {area}',alpha=.9)
ax.set_axisbelow(True);ax.grid(axis='y',color='#e8eaed');ax.set_xlim(-.2,18.2);ax.set_ylim(0,104)
ax.set_xticks(range(19));ax.set_yticks(range(0,101,20));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
ax.set_xlabel('Convolution index (0 = input; 2 = replacement for max pooling)',labelpad=14);ax.set_ylabel('Encrypted spatial coverage',labelpad=12)
ax.spines[['top','right']].set_visible(False)
for side in ['left','bottom']:ax.spines[side].set_color('#dadce0')
ax.tick_params(length=0,pad=8)
ax.legend(title='Initial center region',loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,labelspacing=1.1)
fig.text(.08,.925,'Encrypted coverage through the full ResNet-18 structure',fontsize=21,color='#202124')
fig.text(.08,.87,'Binary mask forward pass  •  Modified pooling convolution + 8 residual blocks  •  224 × 224 input',fontsize=11.5,color='#5f6368')
fig.text(.08,.07,'Unit positive convolution kernels; threshold after every operation; residual branches combined by OR.\nCoverage = nonzero spatial positions / all spatial positions. All channels share the mask.\nIndices 1–18 count main-path convolutions in this modified model; global pooling and the classifier follow index 18.',fontsize=10,color='#5f6368',linespacing=1.7)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'coverage_full_resnet18.{ext}',dpi=220,facecolor='white')
plt.close(fig)
lines=['# Full ResNet binary-mask propagation','', 'Input center pixels are 1; other pixels are 0. Each convolution uses a positive unit spatial kernel, followed by thresholding at >0. This propagates potential sensitive-input dependence without trained-weight cancellation, BN offsets or ReLU value gating. Pointwise BN/activations preserve the dependency mask. All channels share the same mask, allowing an exact single-spatial-plane representation for this structural propagation.','', 'Architecture follows exported parameter kernel sizes: conv1 7x7 stride 2 pad 3; replacement maxpool_0 convolution 3x3 stride 2 pad 1; two residual blocks in each of layer1..layer4, each with two 3x3 convolutions. The first block of stages 2..4 downsamples with stride 2 and a parallel 1x1 stride-2 projection. Residual outputs use the union of both masks. There are 18 main-path convolutions in this modified model (projection convolutions not counted); this is not standard ResNet naming with its final FC counted as layer 18.','', 'Validated all mask outputs at convolution indices 2..8 against the recorded encrypted rectangles for all eight areas (56 exact mask comparisons). Later layers are structural simulations, not measured homomorphic runs. Global average pooling yields a sensitive 1x1 output for every initial region; structurally, the following dense classifier outputs are sensitive too.','', '| Initial area | First 100% spatial coverage at convolution |','|---|---:|']
for a,v in series.items():
    first=next((i for i,x in enumerate(v) if x==100),None)
    lines.append(f'| {a}x{a} | {first} |')
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[-10:]));print('56 measured-prefix mask checks passed. Output:',OUT)
