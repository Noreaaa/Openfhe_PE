#!/usr/bin/env python3
"""One six-stage encrypted-mask ReLU run for each center area at ring 4096."""
import csv,fcntl,hashlib,json,os,resource,shutil
from pathlib import Path
import run_resnet_client_first as bench
ROOT=bench.ROOT
OUT=ROOT/'benchmarks/resnet_ring4096_depth30_encrypted_mask_areas10_40_layers6_20261025'
AREAS=[10,20,30,40]
def report():
 results=[];layer_records=[]
 for area in AREAS:
  folder=OUT/f'area{area:02d}'
  if (folder/'result.json').exists():results.append(json.loads((folder/'result.json').read_text()))
  for row in bench.read_csv(folder/'layers.csv'):layer_records.append(dict(area=area,**row))
 lines=['# Ring 4096: Encrypted-Mask ReLU, Six-Stage Center-Area Sweep','',f'Updated UTC: {bench.now()}','',
 '## Configuration and scope','',
 'Ring dimension 4096; logical slots 2048; multiplicative depth 30; 30/40-bit scaling/first moduli; HYBRID digits 6; 96 OpenMP threads. Original input center areas: 10×10, 20×20, 30×30 and 40×40. One run per area, in separate sequential processes. Original convolutions 1–6, BN and ReLU execute on the server. The second convolution replaces max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap.','',
 'ReLU uses CKKS → FHEW → encrypted sign → CKKS negative indicator → positive mask → multiplication with the original CKKS ciphertext. Activation does not decrypt FHEW signs. Reference decryption occurs only in verification outside operator timers. The normal production executable is used, not the decrypted-sign fixture.','',
 'Sparse convolution includes occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation and working-level plaintext encoding. The latter is newer than earlier ring-scaling experiments, so those archived timings are not an identical-code comparison. Direct rotation keys and unrestricted output-channel concurrency are retained. An explicit experimental-depth option fixes depth at 30 below the conservative default margin; numerical checks validate each completed prefix.','',
 '## Measurement','',
 'Conv+BN and ReLU operator wall times exclude key generation, input encryption and verification. RSS is sampled every 0.5 seconds with phase markers. Operator-phase RSS includes resident keys and retained allocations from the same process, and is not an isolated cloud-process measurement. GNU time records whole-process wall time and lifetime peak RSS separately. No allocator trimming, network measurement or repeated-run averaging.','',
 '## Summary','', '| Center area | Status | Conv+BN (s) | ReLU (s) | Forward total (s / min) | Operator RSS mean / peak (GiB) | Whole-process peak (GiB) | Max absolute feature error |','|---|---|---:|---:|---|---|---:|---:|']
 for area in AREAS:
  r=next((r for r in results if r['area']==area),None)
  if not r:
   count=sum(int(x['area'])==area for x in layer_records);lines.append(f'| {area}×{area} | pending / running ({count}/6) | — | — | — | — | — | — |');continue
  layers=bench.read_csv(OUT/f'area{area:02d}/layers.csv');conv=sum(float(x['conv_ms']) for x in layers)/1000;relu=sum(float(x['relu_ms']) for x in layers)/1000
  lines.append(f"| {area}×{area} | {r['status']} | {conv:.3f} | {relu:.3f} | {r['server_forward_s']:.3f} / {r['server_forward_s']/60:.3f} | {r['server_sampled_mean_gib']:.3f} / {r['server_sampled_peak_gib']:.3f} | {r['whole_process_peak_gib']:.3f} | {r['max_abs_error']:.6g} |")
 lines+=['','## Per-layer measurements','', '| Area | Layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Ciphertexts | Level / towers | Operator RSS mean / peak (GiB) |','|---:|---:|---:|---:|---:|---:|---:|---|---|']
 for x in layer_records:
  r=next((r for r in results if r['area']==int(x['area'])),{})
  mem=r.get('layer_memory',{}).get(x['conv_layers']);mt=f"{mem['mean_gib']:.3f} / {mem['peak_gib']:.3f}" if mem else 'pending'
  a=float(x['conv_ms'])/1000;b=float(x['relu_ms'])/1000
  lines.append(f"| {x['area']} | {x['conv_layers']} | {a:.3f} | {b:.3f} | {a+b:.3f} | {float(x['total_ms'])/1000:.3f} | {x['ciphertexts']} | {x['level']} / {x['remaining_towers']} | {mt} |")
 lines+=['','Raw records per area: `layers.csv`, `run.log`, `rss.csv`, `time.txt`, `result.json`, `configuration.json`. Per-layer feature comparisons are retained in logs/results; they are not classifier accuracy. Binary hash and source snapshot are retained at the sweep root. Incomplete layers are not valid completed measurements.']
 temp=OUT/'REPORT.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'REPORT.md')
 if layer_records:
  with (OUT/'all_layers.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(layer_records[0]));w.writeheader();w.writerows(layer_records)
 bench.save(OUT/'summary.json',results)
def layer_memory(folder):
 samples=bench.read_csv(folder/'rss.csv');memory={}
 for layer in range(1,7):
  ops=[s for s in samples if s['phase'] in ('server_conv','server_relu') and int(s['layer'])==layer];duration=integral=0
  for a,b in zip(ops,ops[1:]):
   dt=float(b['monotonic_s'])-float(a['monotonic_s'])
   if a['phase']==b['phase'] and 0<dt<2:duration+=dt;integral+=dt*(float(a['rss_gib'])+float(b['rss_gib']))/2
  memory[str(layer)]=dict(mean_gib=integral/duration if duration else 0,peak_gib=max((float(s['rss_gib']) for s in ops),default=0))
 return memory
def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0));bench.OUT=OUT
 binary=OUT/'resnet_experiment'
 if not binary.exists():shutil.copy2(ROOT/'build/test',binary)
 config=dict(started_utc=bench.now(),runner_pid=os.getpid(),ring=4096,slots=2048,depth=30,scale_bits=30,first_bits=40,digits=6,threads=96,areas=AREAS,layers=6,repeats=1,client_first_layer=False,activation='encrypted sign -> FHEW-to-CKKS -> ciphertext mask multiplication',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
 bench.save(OUT/'configuration.json',config)
 snap=OUT/'source';snap.mkdir(exist_ok=True)
 for p in ['src/main.cpp','src/init/resnet18.cpp','src/cnn/sparse_conv2d.cpp','src/cnn/sparse_conv2d.hpp','src/cnn/sparse_activation.cpp','src/cnn/encrypted_relu.hpp','scripts/run_resnet_client_first.py','scripts/run_resnet_ring4096_encrypted_mask_sweep.py']:shutil.copy2(ROOT/p,snap/Path(p).name)
 report()
 for area in AREAS:
  folder=OUT/f'area{area:02d}'
  if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='success':continue
  result=bench.run_case(binary,area,6,folder,ring=4096,slots=2048,depth=30,client_first=False,experimental_depth=True)
  result['layer_memory']=layer_memory(folder);bench.save(folder/'result.json',result);report()
  if result['status']!='success':bench.save(OUT/'live.json',dict(status='failed',area=area,result=result));return
 bench.save(OUT/'completion.json',dict(status='finished',finished_utc=bench.now()));bench.save(OUT/'live.json',dict(status='finished',finished_utc=bench.now()));report()
if __name__=='__main__':main()
