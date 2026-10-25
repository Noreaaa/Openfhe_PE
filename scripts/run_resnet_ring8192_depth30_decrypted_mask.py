#!/usr/bin/env python3
"""Six server stages with the historical decrypted-sign plaintext-mask ReLU."""
import csv,fcntl,hashlib,json,os,resource,shutil
from pathlib import Path
import run_resnet_client_first as bench
ROOT=bench.ROOT
OUT=ROOT/'benchmarks/resnet_ring8192_depth30_decrypted_mask_area10_layers6_20261025'
def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 binary=OUT/'resnet_experiment';shutil.copy2(ROOT/'build/test_decrypted_mask',binary)
 bench.OUT=OUT
 config=dict(started_utc=bench.now(),runner_pid=os.getpid(),ring=8192,slots=4096,depth=30,scale_bits=30,first_bits=40,digits=6,threads=96,area=10,layers=6,client_first_layer=False,bootstrap=False,activation='CKKS -> FHEW sign -> sign decryption -> plaintext mask multiplication',convolution='current optimized sparse fast path',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
 bench.save(OUT/'configuration.json',config)
 snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
 for path in ['src/main.cpp','src/cnn/sparse_conv2d.cpp','src/cnn/encrypted_relu.hpp','src/cnn/sparse_activation.cpp','scripts/run_resnet_ring8192_depth30_decrypted_mask.py','src/init/resnet18.cpp','tests/sparse_packing/baseline/baseline_sparse_activation.cpp','tests/sparse_packing/baseline/baseline_sparse_activation.hpp','scripts/run_resnet_client_first.py']:shutil.copy2(ROOT/path,snapshot/Path(path).name)
 (OUT/'REPORT.md').write_text('# Ring 8192 / depth 30 / 10×10 / six server stages\n\nStatus: running. Ring 8192, 4096 slots, depth 30, 30/40-bit moduli, digits 6, 96 threads. Original stages 1–6 all run on the server; no client first-layer offload and no CKKS bootstrap. ReLU decrypts FHEW signs and multiplies by a plaintext mask. This uses the current optimized sparse convolution, not the historical Conv2dBN_P. Detailed results are populated after completion.\n')
 result=bench.run_case(binary,10,6,OUT/'area10',ring=8192,slots=4096,depth=30,client_first=False,experimental_depth=True)
 folder=OUT/'area10';layers=bench.read_csv(folder/'layers.csv');samples=bench.read_csv(folder/'rss.csv')
 ops=[r for r in samples if r['phase'] in ('server_conv','server_relu')]
 memory={}
 for layer in range(1,7):
  selected=[r for r in ops if int(r['layer'])==layer];integral=duration=0
  for a,b in zip(selected,selected[1:]):
   dt=float(b['monotonic_s'])-float(a['monotonic_s'])
   if 0<dt<2 and a['phase']==b['phase']:duration+=dt;integral+=dt*(float(a['rss_gib'])+float(b['rss_gib']))/2
  memory[str(layer)]=dict(mean_gib=integral/duration if duration else 0,peak_gib=max((float(r['rss_gib']) for r in selected),default=0))
 result['layer_memory']=memory;bench.save(folder/'result.json',result)
 conv=sum(float(x['conv_ms']) for x in layers)/1000;relu=sum(float(x['relu_ms']) for x in layers)/1000
 lines=['# Ring 8192 / Depth 30: Decrypted-Sign Mask Baseline','',f"Status: **{result['status']}**. Finished UTC: {result['finished_utc']}",'',
 '## Configuration','', 'Ring dimension 8192; 4096 logical slots; multiplicative depth 30; scaling/first modulus bits 30/40; HYBRID digits 6; 96 OpenMP threads. Initial encrypted center: 10×10 in a 224×224 image. All six original Conv+BN+ReLU stages execute on the server, including conv1 and the convolution replacing max pooling. No client first-layer offload, residual additions, classifier or CKKS bootstrap. One run.','',
 'Convolution uses the current optimized SparseConv2dBN_P path (occupied-channel reduction, one-level grouped stride-2 compaction, owned in-place accumulation). ReLU amplifies by 1000, converts CKKS to FHEW, evaluates signs, decrypts those signs with the FHEW secret key, encodes a plaintext positive mask and multiplies the original CKKS ciphertext by that mask. No FHEW-to-CKKS conversion is evaluated in activation. This benchmark explicitly uses the sign-decryption fixture, not the production encrypted activation. Conversion tasks, sign tasks and mask multiplication are parallelized by the fixture; the encrypted-mask comparison processes ciphertext tasks sequentially. Scheme-switch setup/key generation is unchanged for comparability, so resident reverse-conversion keys may remain even though activation does not use them.','',
 'An explicit experimental-depth option permits depth 30 below the default conservative minimum 32; no modulus depth is silently increased. The estimated experimental minimum is 27. Per-layer numeric checks determine whether this particular run completes successfully. The run does not reproduce the historical 168-second implementation: convolution, activation and possibly precision/thread settings differ.','',
 '## Timing and memory','',f'- Conv+BN total: {conv:.3f} s.',f'- Decrypted-sign mask ReLU total: {relu:.3f} s.',f"- Server forward total: {result['server_forward_s']:.3f} s ({result['server_forward_s']/60:.3f} min).",f"- Whole-process wall time: {result['wall_s']:.3f} s.",f"- Operator-phase sampled RSS mean / peak: {result['server_sampled_mean_gib']:.3f} / {result['server_sampled_peak_gib']:.3f} GiB.",f"- Whole-process peak RSS: {result['whole_process_peak_gib']:.3f} GiB.",'',
 'Operator totals exclude key generation, encryption and verification. Operator-phase RSS is sampled every 0.5 seconds and excludes their time windows, but remains a same-process measure including resident keys and retained allocations. Lifetime peak is separately obtained from GNU time. No network transfer or allocator trimming.','',
 '| Original layer | Conv+BN (s) | ReLU (s) | Layer total (s) | Cumulative (s) | Output ciphertexts | Level / remaining towers | Operator RSS mean / peak (GiB) |','|---:|---:|---:|---:|---:|---:|---|---|']
 for x in layers:
  m=memory[x['conv_layers']];a=float(x['conv_ms'])/1000;b=float(x['relu_ms'])/1000
  lines.append(f"| {x['conv_layers']} | {a:.3f} | {b:.3f} | {a+b:.3f} | {float(x['total_ms'])/1000:.3f} | {x['ciphertexts']} | {x['level']} / {x['remaining_towers']} | {m['mean_gib']:.3f} / {m['peak_gib']:.3f} |")
 lines+=['','## Numeric checks','',f"Structure: {result['structure_ok']}; packing geometry: {result['geometry_ok']}; numerical check thresholds: {result['accuracy_ok']}. Maximum absolute feature error: {result['max_abs_error']:.6g}; maximum RMSE / max(1, reference maximum absolute value): {result['max_relative_rmse']:.6g}. This is a feature consistency check, not classifier accuracy.",'','Raw records: `area10/{layers.csv,run.log,rss.csv,time.txt,result.json}`. Binary and source snapshots are retained.']
 old=ROOT/'benchmarks/resnet_ring2048_depth30_decrypted_mask_area10_layers6_20261025/area10'
 previous=json.loads((old/'result.json').read_text());oldlayers=bench.read_csv(old/'layers.csv')
 lines+=['','## Comparison with ring dimension 2048 / 1024 slots','', '| Metric | Ring 2048 / slots 1024 | Ring 8192 / slots 4096 |','|---|---:|---:|',f"| Server forward (s) | {previous['server_forward_s']:.3f} | {result['server_forward_s']:.3f} |",f"| Conv+BN (s) | {sum(float(x['conv_ms']) for x in oldlayers)/1000:.3f} | {conv:.3f} |",f"| ReLU (s) | {sum(float(x['relu_ms']) for x in oldlayers)/1000:.3f} | {relu:.3f} |",f"| Operator mean RSS (GiB) | {previous['server_sampled_mean_gib']:.3f} | {result['server_sampled_mean_gib']:.3f} |",f"| Whole-process peak (GiB) | {previous['whole_process_peak_gib']:.3f} | {result['whole_process_peak_gib']:.3f} |",'', 'Depth, modulus bit sizes, convolution code, decrypted-sign ReLU, thread count, image and original sensitive area are held fixed. Compared with the ring-2048 baseline, ring dimension and logical slot count both increase fourfold; ciphertext block counts and scheme-switch conversion sizes may therefore change. This is a ring-plus-packing comparison, not an isolated ring-dimension microbenchmark.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 bench.save(OUT/'live.json',dict(status='finished' if result['status']=='success' else 'failed',result=result))
 if result['status']=='success':bench.save(OUT/'completion.json',dict(status='finished',finished_utc=bench.now()))
if __name__=='__main__':main()
