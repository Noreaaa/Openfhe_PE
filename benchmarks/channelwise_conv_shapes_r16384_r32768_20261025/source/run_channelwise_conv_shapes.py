#!/usr/bin/env python3
"""Fully encrypted channel-wise convolution, four ResNet shapes, two ring dimensions."""
import csv,fcntl,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
import run_channelwise_baselines as monitor
ROOT=monitor.ROOT;OUT=ROOT/'benchmarks/channelwise_conv_shapes_r16384_r32768_20261025'
LAYERS=[3,8,12,16];RINGS=[16384,32768]
def report():
 records=[]
 for ring in RINGS:
  for layer in LAYERS:
   p=OUT/f'ring{ring}/layer{layer}'
   if (p/'result.json').exists():records.append(json.loads((p/'result.json').read_text()))
 lines=['# Fully Encrypted Channel-Wise Convolution: Representative ResNet Shapes','',f'Updated UTC: {monitor.now()}','',
 '## Scope','',
 'The channel-wise convolution helper shared with the prior channelwise_prefix_bench baseline is benchmarked. It follows the historical one-ciphertext-per-channel design, with padding, folded BN and cached spatial rotations. There is no partial-encryption bypass or sparse row packing. Four main-path convolution indices in the project\'s modified ResNet are selected: 3 (64→64, 56×56), 8 (128→128, 28×28), 12 (256→256, 14×14), 16 (512→512, 7×7). All are 3×3, stride 1, padding 1, with trained weights and folded batch normalization. These represent stages 1–6, 7–10, 11–14 and 15–18 respectively; stem and downsampling convolutions are not separately characterized.','',
 'Ring dimensions: 16384 and 32768; slots=N/2 (8192 and 16384). Depth 30, 30/40-bit scaling/first moduli, HYBRID digits 6, 96 threads. Each case runs once in a fresh process. Every input feature value is encrypted. Each channel is independently packed into one ciphertext with contiguous H×W spatial slots; unused slots are zero. No channel sharing, spatial compaction or encrypted-region sweep is performed. This fresh per-layer layout differs from the retained 112-pitch grid of the earlier end-to-end prefix; it isolates the convolution shape itself. Input feature values are obtained from that plaintext prefix without residual additions, so this is a convolution-shape benchmark rather than full ResNet inference.','',
 'Every tested convolution starts independently with freshly encrypted level-0, scale-degree-1 ciphertexts. This holds the ciphertext modulus state fixed across shapes; it does not reproduce the consumed levels of an 18-layer encrypted run. Only Conv+BN is timed. No ReLU, scheme-switching keys or bootstrap keys are evaluated/loaded in these isolated processes. Only the eight required spatial rotation keys are generated. The shared convolution caches rotated inputs and accumulates ciphertext/plaintext products across all input channels; no ReLU or bootstrapping occurs.','',
 '## Measurements','',
 'Conv wall time excludes preparation, key generation, encryption and verification. Current RSS is sampled every 0.5 seconds during the operator; it includes resident CKKS keys, reference tensors and allocator retention in the same process. Whole-process peak is obtained independently from GNU time. Neither peak is a full pipeline memory measurement. Output reconstruction is checked against plaintext Conv+BN, at every output feature position. No network transfer or allocator trimming.','',
 '| Layer group | Representative layer | Ring / slots | Input C×H×W | Output C×H×W | Fully encrypted output H×W | Status | Conv+BN (s) | Operator RSS mean / max (GiB) | Whole-process peak (GiB) | Max abs error |','|---|---:|---|---|---|---|---|---:|---|---:|---:|']
 groups={3:'1–6',8:'7–10',12:'11–14',16:'15–18'}
 for ring in RINGS:
  for layer in LAYERS:
   r=next((r for r in records if r['ring']==ring and r['layer']==layer),None)
   if not r:lines.append(f'| {groups[layer]} | {layer} | {ring} / {ring//2} | — | — | — | pending / running | — | — | — | — |');continue
   x=r.get('measurement',{})
   shape=lambda prefix:'×'.join(str(x.get(prefix+k,'?')) for k in ('channels','height','width'))
   lines.append(f"| {groups[layer]} | {layer} | {ring} / {ring//2} | {shape('input_')} | {shape('output_')} | {x.get('encrypted_height','?')}×{x.get('encrypted_width','?')} | {r['status']} | {float(x.get('conv_ms',0))/1000:.3f} | {r['operator_mean_gib']:.3f} / {r['operator_peak_gib']:.3f} | {r['whole_process_peak_gib']:.3f} | {float(x.get('max_abs',0)):.6g} |")
 lines+=['','Per-case raw files: `run.log`, `measurement.csv`, `rss.csv`, `time.txt`, `result.json`, `command.json`. Binary and source snapshots are retained. Only successful complete cases are valid measured results.']
 temp=OUT/'REPORT.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'REPORT.md');monitor.save(OUT/'summary.json',records)
def run(binary,ring,layer):
 folder=OUT/f'ring{ring}/layer{layer}';folder.mkdir(parents=True,exist_ok=True)
 command=[str(binary),str(ROOT),str(layer),str(ring),str(ring//2),'30',str(folder/'measurement.csv')]
 monitor.save(folder/'command.json',dict(command=command,started_utc=monitor.now()))
 env=os.environ.copy();env.update(LD_LIBRARY_PATH='/root/local/openFHE/lib:/root/local/cnpy/lib',OMP_NUM_THREADS='96',OMP_DYNAMIC='FALSE')
 start=time.monotonic();samples=[];pressure=False;lastreport=0
 with (folder/'run.log').open('w') as log,(folder/'rss.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['utc','elapsed_s','phase','layer','rss_gib','process_peak_gib','threads']);writer.writeheader()
  process=subprocess.Popen(['/usr/bin/time','-f','wall_seconds=%e\npeak_rss_kib=%M\nexit_code=%x','-o',str(folder/'time.txt'),*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while process.poll() is None:
   try:
    children=Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split();info=monitor.rss(int(children[0])) if children else {}
    phasepath=folder/'measurement.csv.phase.json';phase=json.loads(phasepath.read_text()) if phasepath.exists() else dict(phase='startup',layer=layer)
    if 'VmRSS' in info:
     sample=dict(utc=monitor.now(),elapsed_s=time.monotonic()-start,phase=phase['phase'],layer=layer,rss_gib=info['VmRSS']/1024**2,process_peak_gib=info['VmHWM']/1024**2,threads=info['Threads']);samples.append(sample);writer.writerow(sample);f.flush()
     monitor.save(OUT/'live.json',dict(status='running',ring=ring,slots=ring//2,model_pid=int(children[0]),runner_pid=os.getpid(),**sample))
    if monitor.available()<12:
     pressure=True;os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    if time.monotonic()-lastreport>30:report();lastreport=time.monotonic()
   except (OSError,ValueError,KeyError):pass
   time.sleep(.5)
  code=process.wait()
 raw=monitor.rows(folder/'measurement.csv');measurement=raw[0] if raw else {}
 metrics=dict(s.split('=',1) for s in (folder/'time.txt').read_text().splitlines() if '=' in s)
 memory=monitor.summarize([s for s in samples if s['phase']=='server_conv'])
 good=code==0 and bool(measurement) and int(measurement['ring'])==ring and int(measurement['layer'])==layer
 result=dict(ring=ring,layer=layer,status='success' if good else 'memory_pressure' if pressure else 'failed',exit_code=code,finished_utc=monitor.now(),wall_s=float(metrics.get('wall_seconds',time.monotonic()-start)),operator_mean_gib=memory['mean_gib'],operator_peak_gib=memory['peak_gib'],whole_process_peak_gib=float(metrics.get('peak_rss_kib',0))/1024**2,measurement=measurement)
 monitor.save(folder/'result.json',result);report();print('CASE_FINISHED',ring,layer,result['status'],flush=True);return result

def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 binary=OUT/'channelwise_conv_shape_bench'
 if not binary.exists():shutil.copy2(ROOT/'build/channelwise_conv_shape_bench',binary)
 snap=OUT/'source';snap.mkdir(exist_ok=True)
 for path in ['tests/sparse_packing/channelwise_conv_shape_bench.cpp','src/cnn/conv2d.cpp','tests/sparse_packing/channelwise_bench_common.hpp','tests/sparse_packing/channelwise_prefix_bench.cpp','src/encryption/setparam.cpp','scripts/run_channelwise_conv_shapes.py']:shutil.copy2(ROOT/path,snap/Path(path).name)
 monitor.save(OUT/'configuration.json',dict(started_utc=monitor.now(),runner_pid=os.getpid(),rings=RINGS,slots='N/2',depth=30,scale_bits=30,first_bits=40,digits=6,threads=96,fully_encrypted=True,input_level=0,input_degree=1,layers=LAYERS,implementation='channel-wise shared baseline convolution; contiguous per-channel HxW; full feature encryption',repeats=1,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
 report()
 for ring in RINGS:
  for layer in LAYERS:
   folder=OUT/f'ring{ring}/layer{layer}'
   if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='success':continue
   result=run(binary,ring,layer)
   if result['status']!='success':monitor.save(OUT/'live.json',dict(status='failed',result=result));return
 monitor.save(OUT/'completion.json',dict(status='finished',finished_utc=monitor.now()));monitor.save(OUT/'live.json',dict(status='finished',finished_utc=monitor.now()));report()
if __name__=='__main__':main()
