#!/usr/bin/env python3
"""A separate approximation baseline with genuine CKKS refresh after every stage."""
import fcntl,hashlib,json,os,resource,shutil
from pathlib import Path
import run_channelwise_baselines as base
OUT=base.ROOT/'benchmarks/channelwise_approx_bootstrap_ring32768_20261025'
MODE='approximation_bootstrap'
def report():
    folder=OUT/MODE;data=base.rows(folder/'layers.csv')
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    config=json.loads((OUT/'configuration.json').read_text()) if (OUT/'configuration.json').exists() else {}
    old=base.ROOT/'benchmarks/channelwise_full_ring32768_20261025/approximation'
    oldrows=base.rows(old/'layers.csv');previous=json.loads((old/'result.json').read_text())
    conv=sum(float(x['conv_ms']) for x in data)/1000;activation=sum(float(x['activation_ms']) for x in data)/1000;boot=sum(float(x['bootstrap_ms']) for x in data)/1000
    lines=['# Channel-Wise Polynomial Approximation with CKKS Bootstrapping','',f'Updated UTC: {base.now()}','',
    '## Configuration','',
    'Same image, weights, channel-wise spatial layout, folded BN and normalized degree-6 polynomial as the completed no-bootstrap approximation baseline. Client executes original Conv1+BN+ReLU. Server features are fully encrypted; original convolutions 2–8 execute once, without residual additions or classifier. Ring dimension 32768; 16384 slots; one ciphertext per channel; 96 OpenMP threads; HYBRID numLargeDigits=6.','',
    f"Scaling/first modulus bits: {config.get('scale_bits','pending')}/{config.get('first_bits','pending')}. Multiplicative depth: {config.get('depth','pending')}; level budget {{3,3}}. OpenFHE's GetBootstrapDepth returns 20 reserved levels for UNIFORM_TERNARY; initial depth 30 reserves ten additional levels. These settings do not establish 128-bit security.",'',
    'A genuine CKKS bootstrap is performed after every activation, including the final stage (64 ciphertexts for each of stages 2–6, 128 for each of stages 7–8: 576 refreshes in total). Each ciphertext is divided by a plaintext-calibrated output bound, its scale degree is reduced to one, and the modulus chain is explicitly reduced to two towers. After EvalBootstrap it is scaled back. Assertions require two towers before bootstrap and more than two afterwards; a skipped/no-op refresh cannot silently pass. The explicit internal level-reduction API is used because the public LevelReduce is a no-op with FLEXIBLEAUTO in this installed OpenFHE version. Normalization, depletion and restoration costs are included in bootstrap time. Ciphertexts are refreshed sequentially, with OpenFHE using the thread pool internally.','',
    'The original polynomial is p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Per-layer convolution normalization follows R=max(1,1.1×max|plaintext preactivation|), and returns R×p(x/R). Bootstrapping normalization is also calibrated from the matching plaintext polynomial trajectory. Calibration is specific to this benchmark image and is performed outside server timers; it is not a dataset-wide bound.','',
    'The installed OpenFHE example uses 59/60-bit moduli for 64-bit bootstrapping. Preliminary genuine-refresh probes with 30/40, 40/50 and 40/41 failed numerical checks; their logs are retained. The selected 59/60-bit two-refresh probe passed with RMSE 8.00e-5 and 1.36e-4, restoring 11 towers each time. The early initial-depletion-check probe is invalid for performance because its public LevelReduce call did not force genuine refresh.', '', '## Measurement and validation','',
    'Conv+BN, polynomial evaluation and bootstrap wall times are separate. Per-layer cumulative time includes all three. Key generation, client compute/encryption, plaintext reference construction and verification are excluded from operator totals. RSS is sampled every 0.5 s, including server_bootstrap phases. Operator-phase RSS still includes resident keys and retained allocations from this shared process; lifetime whole-process peak is reported separately. No network measurement or allocator trimming. Each layer is checked against the plaintext polynomial path after refreshing; polynomial-versus-ReLU error is separate from HE arithmetic error and is not classifier accuracy.','',
    '## Summary','', '| Run | Status | Conv+BN (min) | Polynomial (min) | Bootstrap (min) | Total server (min) | Operator RSS mean / peak (GiB) | Lifetime peak (GiB) |','|---|---|---:|---:|---:|---:|---|---:|',
    f"| No bootstrap, depth 36 | completed | {sum(float(x['conv_ms']) for x in oldrows)/60000:.3f} | {sum(float(x['activation_ms']) for x in oldrows)/60000:.3f} | 0 | {sum(float(x['conv_ms'])+float(x['activation_ms']) for x in oldrows)/60000:.3f} | {previous['server_mean_gib']:.3f} / {previous['server_peak_gib']:.3f} | {previous['whole_process_peak_gib']:.3f} |",
    f"| Bootstrap after each stage | {result.get('status','running')} ({len(data)}/7 stages) | {conv/60:.3f} | {activation/60:.3f} | {boot/60:.3f} | {(conv+activation+boot)/60:.3f} | {result.get('server_mean_gib',0):.3f} / {result.get('server_peak_gib',0):.3f} | {result.get('whole_process_peak_gib',0):.3f} |",'',
    'Changing modulus bit sizes (30/40 to 59/60), initial depth and keeping ciphertexts refreshed changes convolution cost; this is a pipeline comparison rather than a matched-level convolution microbenchmark. Running totals cover only completed stages. Until completion, final memory summaries are pending.','',
    '## Per-layer results','', '| Layer | C×H×W | Conv+BN (s) | Polynomial (s) | Bootstrap (s) | Cumulative (s) | Level before / after refresh | Remaining towers | Operator RSS mean / peak (GiB) | Post-refresh arithmetic RMSE | vs ReLU RMSE |','|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|']
    for x in data:
        mem=result.get('layers',{}).get(x['layer'],{})
        lines.append(f"| {x['layer']} | {x['channels']}×{x['height']}×{x['width']} | {float(x['conv_ms'])/1000:.3f} | {float(x['activation_ms'])/1000:.3f} | {float(x['bootstrap_ms'])/1000:.3f} | {float(x['cumulative_ms'])/1000:.3f} | {x['pre_bootstrap_level']} / {x['post_bootstrap_level']} | {x['remaining_towers']} | {mem.get('mean_gib',0):.3f} / {mem.get('peak_gib',0):.3f} | {float(x['activation_rmse']):.6g} | {float(x['vs_relu_rmse']):.6g} |")
    lines+=['','Raw data: `approximation_bootstrap/{layers.csv,rss.csv,run.log,time.txt,result.json}`. The executable, source snapshot and hashes are retained in this directory. The two-refresh parameter probe is archived separately.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    probe=OUT/'probe_59_60/run.log'
    if probe.read_text().count('BOOTSTRAP_PROBE pass=')!=2 or 'EXPERIMENT_ERROR' in probe.read_text():raise RuntimeError('Parameter probe did not pass')
    binary=OUT/'channelwise_prefix_bench';shutil.copy2(base.ROOT/'build/channelwise_prefix_bench',binary)
    snap=OUT/'source';snap.mkdir(exist_ok=True)
    for p in ['tests/sparse_packing/channelwise_prefix_bench.cpp','scripts/run_channelwise_bootstrap.py','scripts/run_channelwise_baselines.py']:shutil.copy2(base.ROOT/p,snap/Path(p).name)
    base.save(OUT/'configuration.json',dict(started_utc=base.now(),runner_pid=os.getpid(),ring=32768,slots=16384,depth=30,bootstrap_budget=[3,3],bootstrap_reserve=20,scale_bits=59,first_bits=60,digits=6,threads=96,bootstrap_after=list(range(2,9)),refresh_count=576,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    base.OUT=OUT;base.report=report;report()
    result=base.run(MODE,binary,depth=30,last=8,scale_bits=59,first_bits=60)
    base.save(OUT/'live.json',dict(status='finished' if result['status']=='success' else 'failed',result=result))
    if result['status']=='success':base.save(OUT/'completion.json',dict(status='finished',finished_utc=base.now()))
    report()
if __name__=='__main__':main()
