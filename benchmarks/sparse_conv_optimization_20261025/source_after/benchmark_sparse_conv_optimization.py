#!/usr/bin/env python3
"""Compare frozen before/after sparse convolution executables sequentially."""
import hashlib,json,os,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmarks/sparse_conv_optimization_20261025'
rows=[]
def report():
 lines=['# Sparse convolution optimization: matched-input benchmark','',
 'Real ResNet weights and the same image are used for both executables. Ring dimension 16384, 8192 slots, depth 40, 30/40-bit moduli, HYBRID digits 6, 96 threads; direct rotation keys and no channel-concurrency cap. The original-image sensitive region is 40×40 and is propagated through the preceding plaintext Conv+BN+ReLU stages. This tests individual convolution calls, not complete encrypted prefixes. Scheme-switch keys are not resident in either process. Input levels are matched to the archived prefix (0, 12 and 24 for layers 2, 3 and 7). One observation per case; timings are not statistical estimates.','',
 'Both implementations perform the same public convolution, encrypted convolution, packing and folded BN. The optimized path reduces only occupied channel positions, uses one-level baby-step/giant-step stride-2 compaction, and performs owned accumulator additions in place. Output layout and multiplicative-level consumption must match.','',
 '| Layer | Before (s) | After (s) | Speedup | Before / after output level | Before / after whole-process peak (GiB) | Before / after max absolute error |','|---:|---:|---:|---:|---|---|---|']
 for stage in [2,3,7]:
  pair={r['version']:r for r in rows if r['layer']==stage}
  if len(pair)<2:continue
  a,b=pair['before'],pair['after'];assert a['output_level']==b['output_level'] and a['towers']==b['towers']
  lines.append(f"| {stage} | {a['conv_s']:.3f} | {b['conv_s']:.3f} | {a['conv_s']/b['conv_s']:.2f}× | {a['output_level']:g} / {b['output_level']:g} | {a['peak_gib']:.3f} / {b['peak_gib']:.3f} | {a['max_abs']:.6g} / {b['max_abs']:.6g} |")
 lines+=['','The new `CONV_PROFILE` log separates public convolution, preparation/encoding, encrypted processing, and final public-value insertion. Decryption/reference checks occur outside convolution timers. Full-process memory includes setup/verification and is not isolated cloud RSS. Archived full-prefix measurements are not overwritten.','', '## Correctness checks','', 'The sparse convolution suite passed after extending the wide-region case (31-pixel width, seven input channels, nonzero starting column), covering kernels 1/3/7, strides 1/2, channel-block tails, BN/bias, and chained convolutions. A separate 10000-case randomized index check verified the compaction map. Each real-weight case below also decrypts and compares the complete output to plaintext Conv+BN.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/root/local/openFHE/lib:/root/local/cnpy/lib',OMP_NUM_THREADS='96',OMP_DYNAMIC='FALSE')
(OUT/'binary_hashes.json').write_text(json.dumps({v:hashlib.sha256((OUT/v).read_bytes()).hexdigest() for v in ['before','after']},indent=2)+'\n')
for stage,level in [(2,0),(3,12),(7,24)]:
 for version in ['before','after']:
  folder=OUT/f'layer{stage}_{version}';folder.mkdir(exist_ok=True)
  command=[str(OUT/version),str(ROOT),str(stage),'40',str(level)]
  (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
  print('RUNNING',stage,version,flush=True)
  with (folder/'run.log').open('w') as log:
   subprocess.run(['/usr/bin/time','-v','-o',str(folder/'time.txt'),*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  text=(folder/'run.log').read_text();match=re.search(r'OPT_BENCH (.+)',text)
  if not match:raise RuntimeError('missing benchmark result')
  record={k:float(v) for k,v in (s.split('=') for s in match[1].split())};record.update(version=version,layer=stage)
  if record['max_abs']>.02:raise RuntimeError('incorrect convolution')
  rows.append(record);report();print('FINISHED',record,flush=True)
report();(OUT/'completion.json').write_text(json.dumps(dict(status='finished'))+'\n')
