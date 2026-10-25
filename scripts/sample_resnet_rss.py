#!/usr/bin/env python3
"""Sample running campaign RSS without restarting or changing its computation."""
import csv
import json
import math
import os
import re
from pathlib import Path
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/resnet_ring_comparison_slots2048_20261023'
FIELDS=['utc','monotonic_s','ring','area','pid','completed_layers','rss_gib','process_peak_gib']

def current_stage(live):
    path=Path(live['attempt_dir'])/'run.log'
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-32768));text=f.read().decode(errors='replace')
    events=list(re.finditer(r'^(?:Layer conv_bn(\d+) forward|Layer relu(\d+) forward|PREFIX_ACCURACY layers=(\d+)|CONV_MEMORY |RESNET_LAYER |Plaintext\[)',text,re.MULTILINE))
    if not events:return 'context_and_key_setup'
    last=events[-1]
    if last.group(1):return 'relu_'+last.group(1)
    if last.group(2):return 'plaintext_reference_'+last.group(2)
    if last.group(3):return 'conv_'+str(int(last.group(3))+1)+'_prepare' if int(last.group(3))<8 else 'finishing'
    if last.group().startswith('CONV_MEMORY'):
        completed=[int(e.group(3))for e in events if e.group(3)]
        return 'conv_'+str((completed[-1]if completed else 0)+1)+'_encrypted'
    if last.group().startswith('RESNET_LAYER'):return 'model_and_rotation_key_setup'
    return 'input_encryption_or_conv_1_prepare'

def stage_report():
    path=OUT/'rss_stage_samples.csv'
    if not path.exists():return
    with path.open()as f:data=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for r in data:grouped[(r['ring'],r['area'],r['stage'])].append(r)
    lines=['# RSS and execution-stage observations','',
      'RSS sampling interval: 5 seconds. Stage labels are inferred from flushed execution logs; transitions can lag by one sample. Initial 70x70 coverage starts after its historical peak. The 80x80 case will be observed from near process start. Historical per-stage peak columns are cumulative high-water RSS, not current RSS.', '',
      '| Ring | Area | Stage | Samples | Mean sampled RSS (GiB) | Min | Max |', '|---:|---:|---|---:|---:|---:|---:|']
    for (ring,area,stage),records in grouped.items():
        vals=[float(r['rss_gib'])for r in records]
        lines.append(f'| {ring} | {area} | {stage} | {len(vals)} | {sum(vals)/len(vals):.3f} | {min(vals):.3f} | {max(vals):.3f} |')
    lines+=['', '## Observed new process peaks', '', '| UTC | Ring | Area | Inferred stage | New process peak (GiB) | Current RSS (GiB) |', '|---|---:|---:|---|---:|---:|']
    previous={}
    for r in data:
        key=r['pid'];peak=float(r['process_peak_gib'])
        if key in previous and peak>previous[key]+.01:
            lines.append(f"| {r['utc']} | {r['ring']} | {r['area']} | {r['stage']} | {peak:.3f} | {float(r['rss_gib']):.3f} |")
        previous[key]=max(previous.get(key,0),peak)
    temp=OUT/'RSS_ANALYSIS.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'RSS_ANALYSIS.md')
    # Use sampled current RSS, never lifetime VmHWM, for inference-window summaries.
    inference=defaultdict(list)
    for r in data:
        if r['stage'].startswith(('conv_','relu_')):
            inference[(r['ring'],r['area'],r['pid'])].append(r)
    cloud=['# 推理阶段 RSS 采样（同进程实验）','',
      '排除日志标记的密钥准备与明文校验时段；采样间隔为 5 秒，阶段由日志推断。这里的最大值是推理窗口中实际采到的 RSS 最大值，不是进程历史 VmHWM，也不是严格隔离客户端/明文参考对象后的云服务峰值。70×70 仅覆盖运行后半段；80×80 将覆盖更多阶段。详见 [统计口径更正](MEMORY_ACCOUNTING.md)。', '',
      '| 环维度 | 区域 | 采样数 | 窗口内均值 GiB | 中位数 GiB | 采样最大值 GiB | 首次 UTC | 最后 UTC |',
      '|---:|---:|---:|---:|---:|---:|---|---|']
    for (ring,area,pid),records in inference.items():
        vals=[float(r['rss_gib'])for r in records]
        cloud.append(f"| {ring} | {area} | {len(vals)} | {sum(vals)/len(vals):.3f} | {median(vals):.3f} | {max(vals):.3f} | {records[0]['utc']} | {records[-1]['utc']} |")
    temp=OUT/'INFERENCE_RSS.md.tmp';temp.write_text('\n'.join(cloud)+'\n');temp.replace(OUT/'INFERENCE_RSS.md')


def summarize():
    with (OUT/'rss_samples.csv').open() as f:data=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in data:groups[(int(r['ring']),int(r['area']),int(r['pid']))].append(r)
    result=[]
    for (ring,area,pid),records in groups.items():
        values=[float(r['rss_gib'])for r in records];seconds=0;integral=0
        for a,b in zip(records,records[1:]):
            dt=float(b['monotonic_s'])-float(a['monotonic_s'])
            if 0<dt<=30:seconds+=dt;integral+=dt*(float(a['rss_gib'])+float(b['rss_gib']))/2
        ordered=sorted(values)
        result.append(dict(ring=ring,area=area,pid=pid,samples=len(values),covered_seconds=seconds,
          first_utc=records[0]['utc'],last_utc=records[-1]['utc'],time_weighted_mean_gib=integral/seconds if seconds else values[0],
          median_gib=median(values),p95_gib=ordered[max(0,math.ceil(.95*len(values))-1)],min_gib=min(values),max_sampled_gib=max(values),
          process_peak_gib=max(float(r['process_peak_gib'])for r in records)))
    temp=OUT/'rss_summary.json.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(OUT/'rss_summary.json')
    with (OUT/'rss_summary.csv').open('w',newline='')as f:
        if result:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)

def main():
    save=dict(sampler_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),interval_s=5,note='Only the observed window is measured. Historical RSS cannot be reconstructed. Initial case has partial-run coverage.')
    (OUT/'rss_sampling.json').write_text(json.dumps(save,indent=2)+'\n')
    path=OUT/'rss_samples.csv';new=not path.exists()
    stage_path=OUT/'rss_stage_samples.csv';stage_new=not stage_path.exists()
    with path.open('a',newline='')as f, stage_path.open('a',newline='')as stage_file:
        stage_writer=csv.DictWriter(stage_file,fieldnames=FIELDS+['stage','rss_anon_gib','rss_file_gib'])
        if stage_new:stage_writer.writeheader();stage_file.flush()
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if new:writer.writeheader();f.flush()
        while True:
            campaign=json.loads((OUT/'live.json').read_text())
            if campaign['status']!='running':break
            try:
                live=json.loads(Path(campaign['phase_status_path']).read_text());pid=live['model_pid'];info={}
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    key,_,value=line.partition(':')
                    if key in ('VmRSS','VmHWM','RssAnon','RssFile'):info[key]=int(value.split()[0])
                if 'VmRSS'in info:
                    row=dict(utc=datetime.now(timezone.utc).isoformat(),monotonic_s=time.monotonic(),ring=live['ring'],area=live['area'],pid=pid,completed_layers=live['completed_layers'],rss_gib=info['VmRSS']/1024**2,process_peak_gib=info['VmHWM']/1024**2)
                    writer.writerow(row);f.flush();summarize()
                    stage_writer.writerow(dict(row,stage=current_stage(live),rss_anon_gib=info.get('RssAnon',0)/1024**2,rss_file_gib=info.get('RssFile',0)/1024**2));stage_file.flush();stage_report()
            except (OSError,ValueError,KeyError):pass
            time.sleep(5)
    summarize()

if __name__=='__main__':main()
