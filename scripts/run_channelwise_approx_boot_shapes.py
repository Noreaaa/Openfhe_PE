#!/usr/bin/env python3
"""Channel-wise Conv+BN, polynomial and parallel bootstrap with full setup."""
import csv,fcntl,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
import run_channelwise_baselines as monitor
ROOT=monitor.ROOT;OUT=ROOT/'benchmarks/channelwise_approx_boot_shapes_r16384_r32768_20261025'
LAYERS=[3,8,12,16];RINGS=[16384,32768]
def report():
 records=[]
 for ring in RINGS:
  for layer in LAYERS:
   f=OUT/f'ring{ring}/layer{layer}/result.json'
   if f.exists():records.append(json.loads(f.read_text()))
 lines=['# Channel-Wise Approximation + Parallel CKKS Bootstrapping: Four Shapes','',f'Updated UTC: {monitor.now()}','',
 '## Configuration','',
 'Fully encrypted channel-wise inputs; one contiguous H×W feature map per ciphertext. Representative layers 3, 8, 12, 16 cover 64×56×56, 128×28×28, 256×14×14, 512×7×7, respectively. All are same-channel-count 3×3, stride-1, padding-1 Conv+BN operators. Ring dimensions 16384 and 32768; slots N/2; depth 30; scaling/first moduli 59/60 bits; HYBRID digits 6; 96 OpenMP threads. One independent process per case, with fresh level-0 input.','',
 'Each measured process includes CKKS context creation, bootstrap precomputation, key generation (including bootstrap and spatial rotation keys), full input encryption, Conv+BN, normalized degree-6 polynomial activation, genuine CKKS bootstrapping and numerical verification. No scheme-switch keys or FHEW operations are included, because this is the approximation method only. Inputs are prepared by the same sequential plaintext prefix without residual additions as the shape-only benchmark; these are independent shape pipelines, not an 18-layer encrypted execution.','',
 'The polynomial is p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Conv+BN includes normalization R=max(1,1.1 max|plaintext preactivation|) and activation returns R p(x/R). Bootstrap inputs are divided by B=max(1,1.1 max|plaintext polynomial output|), explicitly rescaled and depleted to two towers, bootstrapped, then restored by B. Assertions require two towers before refresh and more than two afterwards. Bounds are calibrated on this single image outside operator timers. Feature arithmetic error and approximation-versus-ReLU error are distinct; this is not a classifier accuracy evaluation.','',
 '## Concurrency and comparison scope','',
 'Polynomial evaluation and bootstrap both use the historical outer OpenMP channel-parallel policy, up to 96 simultaneous ciphertext tasks. This deliberately differs from the previous sequential-bootstrap experiment. Nested teams are disabled by the process defaults. Large temporary working sets are expected; the monitor stops a case if host MemAvailable falls below 12 GiB. It does not silently reduce threads or ring dimension.','',
 'This is an adapted channel-wise baseline with cached rotations, padding and folded BN, not an unmodified historical executable. The 59/60-bit precision was chosen after genuine bootstrap probes failed with lower precision. Therefore these timings cannot be attributed to adding bootstrap alone versus the earlier 30/40-bit shape-only runs. Every case starts in a fresh process, so allocations from previous network layers are not reproduced.','',
 '## Time and memory','',
 'All phase timings are wall seconds. Operator total = convolution + polynomial + bootstrap. Setup+compute includes key setup and encryption as well; process wall additionally includes data preparation, calibration, numerical verification and other overhead. Current RSS is sampled every 0.5 seconds for all phases. Operator RSS includes resident keys and allocator retention; whole-process peak includes setup and verification too. No network measurement or allocator trimming.','',
 '| Group | Layer | Ring / slots | Status | Key setup (s) | Encrypt (s) | Conv+BN (s) | Polynomial (s) | Bootstrap (s) | Operator total (s) | Setup+compute (s) | Process wall (s) | Whole-process peak (GiB) |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 groups={3:'1–6',8:'7–10',12:'11–14',16:'15–18'}
 for ring in RINGS:
  for layer in LAYERS:
   r=next((x for x in records if x['ring']==ring and x['layer']==layer),None)
   if not r:lines.append(f'| {groups[layer]} | {layer} | {ring} / {ring//2} | pending / running | — | — | — | — | — | — | — | — | — |');continue
   x=r.get('measurement',{});v=lambda k:float(x.get(k,0))/1000
   lines.append(f"| {groups[layer]} | {layer} | {ring} / {ring//2} | {r['status']} | {v('key_setup_ms'):.3f} | {v('encryption_ms'):.3f} | {v('conv_ms'):.3f} | {v('activation_ms'):.3f} | {v('bootstrap_ms'):.3f} | {v('operator_total_ms'):.3f} | {v('pipeline_ms'):.3f} | {r['wall_s']:.3f} | {r['whole_process_peak_gib']:.3f} |")
 lines+=['','## Phase RSS and numerical checks','', '| Layer | Ring | Key setup max (GiB) | Conv max (GiB) | Polynomial max (GiB) | Bootstrap max (GiB) | Operator mean / max (GiB) | Post-bootstrap RMSE | vs ReLU RMSE |','|---:|---:|---:|---:|---:|---:|---|---:|---:|']
 for r in records:
  mem=r.get('phase_memory',{});x=r.get('measurement',{});peak=lambda p:mem.get(p,{}).get('peak_gib',0)
  lines.append(f"| {r['layer']} | {r['ring']} | {peak('key_setup'):.3f} | {peak('server_conv'):.3f} | {peak('server_relu'):.3f} | {peak('server_bootstrap'):.3f} | {r['operator_mean_gib']:.3f} / {r['operator_peak_gib']:.3f} | {float(x.get('rmse',0)):.6g} | {float(x.get('vs_relu_rmse',0)):.6g} |")
 lines+=['','A preliminary 8-output-channel parallel-bootstrap probe is retained under `validation/probe8`; it is not included in the four-shape table. Successful cases must pass convolution, polynomial and post-bootstrap arithmetic checks. Incomplete or failed cases are not valid complete performance measurements. Raw logs, per-phase RSS, numerical metrics, binary and source snapshots are retained.']
 temp=OUT/'REPORT.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'REPORT.md');monitor.save(OUT/'summary.json',records)
def run(binary,ring,layer,output_limit=0):
 folder=OUT/'validation/probe8' if output_limit else OUT/f'ring{ring}/layer{layer}';folder.mkdir(parents=True,exist_ok=True)
 command=[str(binary),str(ROOT),str(layer),str(ring),str(ring//2),'30',str(folder/'measurement.csv'),str(output_limit)]
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
 memory=monitor.summarize([s for s in samples if s['phase'] in ('server_conv','server_relu','server_bootstrap')])
 good=code==0 and bool(measurement) and int(measurement['ring'])==ring and int(measurement['layer'])==layer
 result=dict(ring=ring,layer=layer,status='success' if good else 'memory_pressure' if pressure else 'failed',exit_code=code,finished_utc=monitor.now(),wall_s=float(metrics.get('wall_seconds',time.monotonic()-start)),operator_mean_gib=memory['mean_gib'],operator_peak_gib=memory['peak_gib'],whole_process_peak_gib=float(metrics.get('peak_rss_kib',0))/1024**2,measurement=measurement,phase_memory={phase:monitor.summarize([s for s in samples if s['phase']==phase]) for phase in ('key_setup','input_encryption','server_conv','server_relu','server_bootstrap','verification')})
 monitor.save(folder/'result.json',result);report();print('CASE_FINISHED',ring,layer,result['status'],flush=True);return result

def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 binary=OUT/'channelwise_approx_boot_shape_bench'
 if not binary.exists():shutil.copy2(ROOT/'build/channelwise_approx_boot_shape_bench',binary)
 snap=OUT/'source';snap.mkdir(exist_ok=True)
 for path in ['tests/sparse_packing/channelwise_approx_boot_shape_bench.cpp','src/cnn/conv2d.cpp','tests/sparse_packing/channelwise_bench_common.hpp','tests/sparse_packing/channelwise_prefix_bench.cpp','src/encryption/setparam.cpp','scripts/run_channelwise_approx_boot_shapes.py']:shutil.copy2(ROOT/path,snap/Path(path).name)
 monitor.save(OUT/'configuration.json',dict(started_utc=monitor.now(),runner_pid=os.getpid(),rings=RINGS,slots='N/2',depth=30,scale_bits=59,first_bits=60,bootstrap_budget=[3,3],parallel_bootstrap=True,digits=6,threads=96,fully_encrypted=True,input_level=0,input_degree=1,layers=LAYERS,implementation='channel-wise full-feature Conv+BN + normalized polynomial + parallel genuine bootstrap',repeats=1,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
 report()
 probe=OUT/'validation/probe8/result.json'
 if not probe.exists() or json.loads(probe.read_text()).get('status')!='success':
  result=run(binary,16384,3,output_limit=8)
  if result['status']!='success':monitor.save(OUT/'live.json',dict(status='validation_failed',result=result));return
 for ring in RINGS:
  for layer in LAYERS:
   folder=OUT/f'ring{ring}/layer{layer}'
   if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='success':continue
   result=run(binary,ring,layer)
   if result['status']!='success':monitor.save(OUT/'live.json',dict(status='failed',result=result));return
 monitor.save(OUT/'completion.json',dict(status='finished',finished_utc=monitor.now()));monitor.save(OUT/'live.json',dict(status='finished',finished_utc=monitor.now()));report()
if __name__=='__main__':main()
