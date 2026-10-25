#!/usr/bin/env python3
"""Produce a final memory timeline/report from existing samples; no reruns."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/resnet_ring_comparison_slots2048_20261023'
def read(name):
    with (OUT/name).open()as f:return list(csv.DictReader(f))
def main():
    samples=[r for r in read('rss_samples.csv')if r['pid']=='4007206']
    stages=[r for r in read('rss_stage_samples.csv')if r['pid']=='4007206']
    profile=read('ring16384/repeat1/area80/layers8/ring16384/layers.csv')
    grouped=defaultdict(list)
    for r in stages:grouped[r['stage']].append(float(r['rss_gib']))
    summary=next(r for r in json.loads((OUT/'rss_summary.json').read_text())if r['pid']==4007206)
    above={}
    for limit in (35,60,70):
        above[limit]=sum(float(b['monotonic_s'])-float(a['monotonic_s']) for a,b in zip(samples,samples[1:])if 0<float(b['monotonic_s'])-float(a['monotonic_s'])<=30 and (float(a['rss_gib'])+float(b['rss_gib']))/2>limit)/60
    lines=['# 实验完成后的内存监控结论','',
      '实验于北京时间 2026-10-24 02:29 完成，16/16 项均完成，无内存回退。本文重点分析最后一项：ringDim=16384、80×80、2048 slots、30/40、digits=6、depth=40、全局 96 线程、直接旋转、无并发限制、8 层卷积。', '',
      '## 统计范围','',
      '5 秒间隔读取进程当前 RSS；最后一项获取了 1171 个模型进程样本，覆盖约 97.88 分钟，进程总耗时约 98.04 分钟。已排除启动瞬间误采的一条 GNU time 包装进程记录。阶段从已刷新日志推断，切换定位精度约一个采样间隔。', '',
      '这些是同进程实验的 RSS：客户端准备、明文参考对象和分配器保留页未被完全隔离。它不是独立云服务进程的严格内存测量。5 秒采样最大值和内核历史高水位不是同一指标。', '',
      '## 主要结果','',
      f"- 全观察窗口时间加权平均 RSS：{summary['time_weighted_mean_gib']:.2f} GiB。",
      f"- 中位数：{summary['median_gib']:.2f} GiB；P95：{summary['p95_gib']:.2f} GiB。",
      f"- 采样最大值：{summary['max_sampled_gib']:.2f} GiB；进程历史峰值：{summary['process_peak_gib']:.2f} GiB。",
      f"- RSS 超过 60 GiB 约 {above[60]:.1f} 分钟，超过 70 GiB 约 {above[70]:.1f} 分钟。阈值持续时间按相邻样本区间近似，不能视为连续精确测量。", '',
      '密钥准备阶段采到的最大 RSS 约 18.6 GiB，输入加密/第一层准备阶段约 26.5 GiB。随后第一层卷积逐步升到 72.5 GiB；卷积至 ReLU 的切换附近，内核历史高水位升到 73.05 GiB。高内存不是只有密钥生成时出现，也不是持续几秒的尖峰。', '',
      '## 逐层数据','',
      '| 层 | Conv+BN 时间 分钟 | Conv 采样均值 GiB | Conv 采样最大 GiB | ReLU 时间 分钟 | ReLU 采样均值 GiB | ReLU 采样最大 GiB |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for r in profile:
        n=int(r['conv_layers']);c=grouped[f'conv_{n}_encrypted'];a=grouped[f'relu_{n}']
        lines.append(f"| {n} | {float(r['conv_ms'])/60000:.2f} | {mean(c):.2f} | {max(c):.2f} | {float(r['relu_ms'])/60000:.2f} | {mean(a):.2f} | {max(a):.2f} |")
    lines+=['', '## 已确认的规律与原因边界','',
      '1. 第一层卷积的 RSS 均值约 58.6 GiB，且第一层 ReLU 在约 71.5 GiB 平台持续约 19.6 分钟。此前将高值称为短时尖峰不准确。',
      '2. 第一层 ReLU 结束后，样本降至约 39.1 GiB；第二层 ReLU 约 30.2 GiB；后续 ReLU 多为 24–30 GiB。你看到的约 28 GiB 主要反映后续阶段，不能代表全程平均。',
      '3. 第一层同时处理 64 个输出通道、43 个受影响输出行。以一个高层级密文约 10.25 MiB 粗算，仅通道累加器的容量量级就可达 64×43×10.25 MiB≈27.5 GiB，此外还有输入旋转缓存、输出、密钥和密钥切换临时工作区。该估算解释峰值来源量级，不是分项实测。',
      '4. 第一层 ReLU 期间的 smaps 显示，约 67.9 GiB 集中在 1122 个接近 64 MiB 的匿名可写映射中，与多线程 glibc 堆分配区形态相符。结合卷积局部缓存生命周期，分配器保留/碎片是合理怀疑；但是没有读取分配器已用/空闲统计，不能确认其中多少空间已经空闲，也不能排除仍存活的转换对象。',
      '5. 没有执行 malloc_trim、调试器注入、减少线程或更换算法来干预本轮测量。进程结束后，整机已用内存回到约 2 GiB。退出后释放并不证明运行期没有泄漏。', '',
      '## 内存曲线','',
      '![RSS timeline](memory_80_rss.png)', '',
      '蓝线为当前 RSS；虚线是从进程启动累计的历史高水位，不能理解为各层的当前用量。下图比较各层 Conv/ReLU 的采样均值。', '',
      '[原始 RSS](rss_samples.csv) · [阶段采样](rss_stage_samples.csv) · [完整实验表](REPORT.md) · [内存统计口径](MEMORY_ACCOUNTING.md)']
    (OUT/'FINAL_MEMORY_REPORT.md').write_text('\n'.join(lines)+'\n')
    sys.path.insert(0,'/root/deps/memory-report-python')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=datetime.fromisoformat('2026-10-23T16:51:37.901922+00:00')
    x=[(datetime.fromisoformat(r['utc'])-start).total_seconds()/60 for r in samples]
    fig,(ax,bars)=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[2,1]},layout='constrained')
    ax.plot(x,[float(r['rss_gib'])for r in samples],label='Current RSS',lw=1.5)
    ax.plot(x,[float(r['process_peak_gib'])for r in samples],label='Lifetime high-water RSS',ls='--',lw=1,color='gray')
    segments=[]
    for r in stages:
        name=r['stage']
        if not(name.startswith('conv_')or name.startswith('relu_')):continue
        t=(datetime.fromisoformat(r['utc'])-start).total_seconds()/60
        if not segments or segments[-1][0]!=name:segments.append([name,t,t])
        else:segments[-1][2]=t
    for name,a,b in segments:
        ax.axvspan(a,b,alpha=.08,color='tab:blue'if name.startswith('conv_')else'tab:orange')
    ax.set(xlabel='Minutes since process start',ylabel='Resident memory (GiB)',title='16384 ring / 80x80 private region / 2048 slots / depth 40')
    ax.grid(alpha=.2);ax.legend(loc='upper right');ax.set_ylim(bottom=0)
    ax.annotate('First ReLU: about 71.5 GiB for 19.6 min',xy=(31,71.5),xytext=(43,61),arrowprops={'arrowstyle':'->'})
    import numpy as np
    n=np.arange(1,9);cv=[mean(grouped[f'conv_{i}_encrypted'])for i in n];rv=[mean(grouped[f'relu_{i}'])for i in n]
    bars.bar(n-.18,cv,width=.36,label='Conv+BN sampled mean');bars.bar(n+.18,rv,width=.36,label='ReLU sampled mean')
    bars.set(xlabel='Convolution stage',ylabel='RSS (GiB)',xticks=n);bars.legend();bars.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'memory_80_rss.png',dpi=160);fig.savefig(OUT/'memory_80_rss.svg');plt.close(fig)
    print('Final memory report and plots generated.')
if __name__=='__main__':main()
