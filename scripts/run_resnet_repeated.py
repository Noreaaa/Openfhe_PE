#!/usr/bin/env python3
"""ResNet convolution-prefix repeats, with per-attempt artifacts and memory fallback."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev

ROOT = Path(os.environ.get('RESNET_REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
PRIMARY_RING = int(os.environ.get('RESNET_RING_DIM', '32768'))
if PRIMARY_RING not in (16384, 32768):
    raise ValueError('RESNET_RING_DIM must be 16384 or 32768')
BINARY = Path(os.environ.get('RESNET_BINARY', ROOT/'build/test')).resolve()
SLOTS = 2048
OUT = Path(os.environ.get('RESNET_OUTPUT', ROOT/f'benchmarks/resnet_repeated_ring{PRIMARY_RING}_30_40_d6_depth40_slots2048_20261023')).resolve()
FIELDS = ['repeat','area','target_layers','ring','status','wall_s','forward_s','conv_s','relu_s','peak_gib','exit_code','attempt_dir']

def stamp():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)

def read_profile(path):
    try:
        with path.open() as f: return list(csv.DictReader(f))
    except OSError: return []

def memory(pid):
    try:
        info={}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('VmRSS','VmHWM','Threads'): info[k]=int(v.split()[0])
        if not info or 'VmRSS' not in info: return None
        return info
    except (OSError,ValueError): return None

def available_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])
    return 0

def write_csv(path, rows, fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    tmp.replace(path)

def aggregate():
    runs=[json.loads(p.read_text()) for p in sorted(OUT.glob('repeat*/area*/layers*/result.json'))]
    write_csv(OUT/'runs.csv', runs, FIELDS)
    prefixes=[]; layers=[]
    for run in runs:
        if run['status']!='success': continue
        for r in read_profile(Path(run['attempt_dir'])/'layers.csv'):
            common={k:run[k] for k in ('repeat','area','target_layers','ring')}
            prefixes.append(dict(common,layer=int(r['conv_layers']),seconds=float(r['total_ms'])/1000,peak_gib=float(r['peak_mib'])/1024))
            layers.append(dict(common,layer=int(r['conv_layers']),conv_s=float(r['conv_ms'])/1000,relu_s=float(r['relu_ms'])/1000,total_s=(float(r['conv_ms'])+float(r['relu_ms']))/1000))
    if prefixes: write_csv(OUT/'prefix_samples.csv',prefixes,list(prefixes[0]))
    if layers: write_csv(OUT/'layer_samples.csv',layers,list(layers[0]))
    def grouped(rows, keys, metrics, filename):
        groups=defaultdict(list)
        for r in rows: groups[tuple(r[k] for k in keys)].append(r)
        result=[]
        for key,items in sorted(groups.items()):
            r=dict(zip(keys,key));r['samples']=len(items)
            for metric in metrics:
                values=[x[metric] for x in items]
                r[metric+'_mean']=mean(values);r[metric+'_std']=stdev(values) if len(values)>1 else 0
                r[metric+'_min']=min(values);r[metric+'_max']=max(values)
            result.append(r)
        if result: write_csv(OUT/filename,result,list(result[0]))
        return result
    good=[r for r in runs if r['status']=='success']
    rs=grouped(good,['area','target_layers','ring'],['wall_s','forward_s','conv_s','relu_s','peak_gib'],'run_means.csv')
    ps=grouped(prefixes,['area','layer','ring'],['seconds','peak_gib'],'prefix_means.csv')
    grouped(prefixes,['area','target_layers','layer','ring'],['seconds','peak_gib'],'prefix_means_by_target.csv')
    ls=grouped(layers,['area','layer','ring'],['conv_s','relu_s','total_s'],'layer_means.csv')
    grouped(layers,['area','target_layers','layer','ring'],['conv_s','relu_s','total_s'],'layer_means_by_target.csv')
    lines=['# ResNet repeated prefix benchmark','',f'Updated UTC: {stamp()}', '',f'Completed successful cases: {len(good)}/8; recorded cases: {len(runs)}.', '',
      'Configuration: 30/40 bits, depth 40, HYBRID digits 6, 2048 logical slots, 96 OpenMP threads, direct rotation keys, unrestricted channel concurrency, no CKKS bootstrap. Center square sides 10..80 step 10; target 8 convolutions; one run per configuration. Each case uses a fresh process and fresh keys. Plaintext verification enabled in every case.', '',
      'Whole-process wall time includes setup, encryption and verification. Forward time sums Conv+BN and ReLU timers and excludes the plaintext observer. Peak RSS includes setup and verification. Per-layer peak is cumulative process high-water RSS, not isolated layer working memory.', '',
      'The prefix is sequential Conv+BN+ReLU without residual additions/downsample branches. Layer 2 replaces maxpool with convolution. Layers 6/7/8 map to layer1.1.conv2 / layer2.0.conv1 / layer2.0.conv2. This experiment does not establish full ResNet classification accuracy or 128-bit security.', '',
      'Ring dimensions are summarized separately. Memory-pressure fallback triggers below 12 GiB host available RAM, on std::bad_alloc, or an OOM kill; each affected case retries with ring 16384. Failed and partial attempts remain available.', '',
      '## Single-run measurements', '', '| Area | Target | Ring | n | Wall mean (s) | Forward mean (s) | Conv mean (s) | ReLU mean (s) | Peak mean / max (GiB) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rs: lines.append(f"| {r['area']}×{r['area']} | {r['target_layers']} | {r['ring']} | {r['samples']} | {r['wall_s_mean']:.3f} | {r['forward_s_mean']:.3f} | {r['conv_s_mean']:.3f} | {r['relu_s_mean']:.3f} | {r['peak_gib_mean']:.3f} / {r['peak_gib_max']:.3f} |")
    lines+=['','## Pooled prefix timings','', 'At completion, per area and absent fallback/failure: every prefix from 1 through 8 has 1 sample. Prefix samples from one process are correlated; CSV also provides target-stratified means.', '', '| Area | Prefix | Ring | n | Mean (s) | Std (s) | Min (s) | Max (s) | Peak max (GiB) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ps:
        if r['layer']>=1: lines.append(f"| {r['area']}×{r['area']} | {r['layer']} | {r['ring']} | {r['samples']} | {r['seconds_mean']:.3f} | {r['seconds_std']:.3f} | {r['seconds_min']:.3f} | {r['seconds_max']:.3f} | {r['peak_gib_max']:.3f} |")
    lines+=['', '## Per-layer measurements', '', '| Area | Layer | Ring | n | Conv+BN mean (s) | ReLU mean (s) | Layer mean (s) | Layer std (s) |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in ls: lines.append(f"| {r['area']}×{r['area']} | {r['layer']} | {r['ring']} | {r['samples']} | {r['conv_s_mean']:.3f} | {r['relu_s_mean']:.3f} | {r['total_s_mean']:.3f} | {r['total_s_std']:.3f} |")
    lines+=['', '## Individual runs', '', '| Repeat | Area | Target | Ring | Status | Process wall (s) | Forward (s) | Peak RSS (GiB) |', '|---:|---|---:|---:|---|---:|---:|---:|']
    for r in runs: lines.append(f"| {r['repeat']} | {r['area']}×{r['area']} | {r['target_layers']} | {r['ring']} | {r['status']} | {r['wall_s']:.3f} | {r['forward_s']:.3f} | {r['peak_gib']:.3f} |")
    attempts=[json.loads(p.read_text()) for p in sorted(OUT.glob('repeat*/area*/layers*/ring*/result.json'))]
    lines+=['', f"Sum of completed attempt process wall times (including failed/fallback attempts): {sum(r['wall_s'] for r in attempts):.3f} s.", f"Sum of successful forward times: {sum(r['forward_s'] for r in good):.3f} s."]
    save(OUT/'totals.json',dict(successful_cases=len(good),planned_cases=8,completed_attempts=len(attempts),sum_attempt_wall_s=sum(r['wall_s'] for r in attempts),sum_success_forward_s=sum(r['forward_s'] for r in good),max_attempt_peak_gib=max((r['peak_gib'] for r in attempts),default=0)))
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def stats(path, profile, meta, exit_code, peak, elapsed, pressure):
    rows=read_profile(profile)
    text=path.read_text() if path.exists() else ''
    for line in text.splitlines():
        if 'Maximum resident set size (kbytes):' in line: peak=max(peak,int(line.split(':')[-1]))
        if line.startswith('wall_seconds='): elapsed=float(line.split('=')[1])
        if line.startswith('peak_rss_kib='): peak=max(peak,int(line.split('=')[1]))
    last=rows[-1] if rows else {}
    ok=exit_code==0 and len(rows)==meta['target_layers'] and int(last.get('conv_layers',0))==meta['target_layers']
    return dict(meta,status='success' if ok else ('memory_pressure' if pressure else 'failed'),exit_code=exit_code,
        wall_s=elapsed,forward_s=float(last.get('total_ms',0))/1000,conv_s=float(last.get('cumulative_conv_ms',0))/1000,
        relu_s=float(last.get('cumulative_relu_ms',0))/1000,peak_gib=peak/1024**2,finished_utc=stamp(),completed_layers=len(rows))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    aggregate()
    save(OUT/'configuration.json',dict(runner_pid=os.getpid(),started_utc=stamp(),ring=PRIMARY_RING,fallback_ring=16384 if PRIMARY_RING==32768 else None,scale_bits=30,first_bits=40,digits=6,depth=40,threads=96,conv_scratch_mib=0,slots=SLOTS,areas=list(range(10,81,10)),layers=[8],repeats=1,cases=8,binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest()))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='96',OMP_DYNAMIC='FALSE',LD_LIBRARY_PATH='/root/local/openFHE/lib:/root/local/cnpy/lib')
    order=[(rep,area,8) for rep in range(1,2) for area in range(10,81,10)]
    for rep,area,target in order:
        case=OUT/f'repeat{rep}'/f'area{area:02d}'/f'layers{target}'
        case.mkdir(parents=True,exist_ok=True)
        if (case/'result.json').exists() and json.loads((case/'result.json').read_text())['status']=='success':continue
        for ring in ((32768,16384) if PRIMARY_RING==32768 else (16384,)):
            attempt=case/f'ring{ring}';attempt.mkdir(exist_ok=True)
            meta=dict(repeat=rep,area=area,target_layers=target,ring=ring,attempt_dir=str(attempt),started_utc=stamp())
            peak=0;pressure=False;proc=None
            command=[str(BINARY),'--model','ResNet-18','--ringDim',str(ring),'--batch',str(SLOTS),'--layers',str(target),'--depth','40','--scale-bits','30','--first-bits','40','--encrypt-size',str(area),'--bootstrap-after','0','--conv-scratch-mib','0','--verify','--data-root',str(ROOT),'--profile',str(attempt/'layers.csv')]
            meta['command']=command
            adopt_path=os.environ.get('RESNET_ADOPT_CASE_PATH')
            adopt=json.loads(Path(adopt_path).read_text()) if adopt_path and Path(adopt_path).exists() else {}
            adopted=all(adopt.get(k)==meta[k] for k in ('repeat','area','target_layers','ring'))
            if adopted:
                meta['started_utc']=adopt['started_utc']
                meta['adopted_existing_process']=True
                pid=adopt['model_pid'];group_pid=adopt['wrapper_pid']
                log=None
            else:
                log=(attempt/'run.log').open('w')
                proc=subprocess.Popen(['/usr/bin/time','-f','wall_seconds=%e\npeak_rss_kib=%M\nexit_code=%x','-o',str(attempt/'time.txt'),*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pid=proc.pid;group_pid=proc.pid
            save(attempt/'configuration.json',meta)
            start=time.monotonic()-(datetime.now(timezone.utc)-datetime.fromisoformat(meta['started_utc'])).total_seconds();print('START',rep,area,target,ring,'adopted',adopted,flush=True)
            while True:
                modelpid=pid
                if proc:
                    if proc.poll() is not None:break
                    try:
                        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
                        if children:modelpid=int(children[0])
                    except OSError:pass
                info=memory(modelpid)
                if adopted and not info:
                    break
                if info:peak=max(peak,info.get('VmHWM',0))
                rows=read_profile(attempt/'layers.csv')
                save(OUT/'live.json',dict(meta,model_pid=modelpid,elapsed_monitor_s=round(time.monotonic()-start,1),rss_gib=(info or {}).get('VmRSS',0)/1024**2,peak_gib=peak/1024**2,completed_layers=len(rows),updated_utc=stamp(),available_gib=available_kib()/1024**2))
                if available_kib()<12*1024**2:
                    pressure=True
                    os.killpg(group_pid,signal.SIGTERM)
                    if proc:
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(5)
            if proc:
                code=proc.wait();log.close()
            else:
                if pressure:
                    for _ in range(15):
                        if not memory(pid):break
                        time.sleep(1)
                    if memory(pid):os.killpg(group_pid,signal.SIGKILL)
                time.sleep(1)
                text=(attempt/'time.txt').read_text() if (attempt/'time.txt').exists() else ''
                codes=[line.split('=',1)[1] for line in text.splitlines() if line.startswith('exit_code=')]
                code=int(codes[-1]) if codes else (-15 if pressure else 1)
            row=stats(attempt/'time.txt',attempt/'layers.csv',meta,code,peak,time.monotonic()-start,pressure)
            text=(attempt/'run.log').read_text(errors='replace')[-12000:]
            oom=pressure or code in (-9,137) or 'bad_alloc' in text or 'Cannot allocate memory' in text
            row['memory_failure']=oom
            save(attempt/'result.json',row);save(case/'result.json',row);aggregate()
            print('FINISH',rep,area,target,ring,row['status'],'wall',row['wall_s'],'peak',row['peak_gib'],flush=True)
            if row['status']=='success' or not oom or ring==16384:break
            print('FALLBACK to ring16384',flush=True)
    aggregate()
    save(OUT/'completion.json',dict(finished_utc=stamp(),runner_wall_s=time.monotonic()-started))
    save(OUT/'live.json',dict(status='finished',finished_utc=stamp()))

if __name__=='__main__':main()
