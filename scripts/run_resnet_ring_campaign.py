#!/usr/bin/env python3
"""Run the 32768-ring repeats, then repeat the entire experiment at ring 16384."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/resnet_ring_comparison_slots2048_20261023'

def stamp():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def rows(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))
def report():
    lines=['# ResNet ring-dimension comparison', '', 'Configuration: 30/40 bits, depth 40, HYBRID digits 6, 2048 slots, 96 OpenMP threads. Direct rotations; unrestricted output-channel concurrency. No CKKS bootstrap. Each phase runs eight central encrypted areas (10..80 step 10) through eight sequential convolutions, one run per area (8 cases per phase). Phase 32768 finishes before phase 16384 starts.', '',
      'The first phase retries memory failures with ring 16384; fallback results are identified by actual ring and kept separate. The second phase runs all 8 cases independently at ring 16384, regardless of first-phase fallback results. Cancelled 1024-slot runs are excluded. Each case is a fresh process, with fresh keys and plaintext verification.', '',
      'The executed prefix has no residual additions/downsample branches. Layer 2 is the convolution replacing maxpool; layer 8 is layer2.0.conv2. Forward time excludes setup and plaintext observer; process wall time includes them. Layer peak RSS is cumulative process high-water RSS, not isolated working memory.', '',
      '## Progress and detailed tables', '', '[32768 phase: per-run, prefix and per-layer tables](ring32768/REPORT.md)', '', '[16384 phase: per-run, prefix and per-layer tables](ring16384/REPORT.md)', '',
      '## Single-run measurements', '', '| Primary ring | Actual ring | Area | Samples | Wall mean (s) | Forward mean (s) | Whole-process peak mean (GiB) | Whole-process peak max (GiB) |', '|---:|---:|---|---:|---:|---:|---:|---:|']
    combined=[];total_wall=0;total_forward=0;successes=0
    for primary in (32768,16384):
        phase=OUT/f'ring{primary}'
        for r in rows(phase/'run_means.csv'):
            lines.append(f"| {primary} | {r['ring']} | {r['area']}×{r['area']} | {r['samples']} | {float(r['wall_s_mean']):.3f} | {float(r['forward_s_mean']):.3f} | {float(r['peak_gib_mean']):.3f} | {float(r['peak_gib_max']):.3f} |")
        for r in rows(phase/'layer_means.csv'):combined.append(dict(primary_ring=primary,**r))
        if (phase/'totals.json').exists():
            t=json.loads((phase/'totals.json').read_text());total_wall+=t['sum_attempt_wall_s'];total_forward+=t['sum_success_forward_s'];successes+=t['successful_cases']
    lines+=['',f'Successful cases: {successes}/16. Sum of completed process wall times, including fallback attempts: {total_wall:.3f} s. Sum of successful forward times: {total_forward:.3f} s.', '',
      '## Per-layer measurements', '', '| Primary ring | Actual ring | Area | Layer | Samples | Conv+BN mean (s) | ReLU mean (s) | Total mean (s) |', '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in combined:lines.append(f"| {r['primary_ring']} | {r['ring']} | {r['area']}×{r['area']} | {r['layer']} | {r['samples']} | {float(r['conv_s_mean']):.3f} | {float(r['relu_s_mean']):.3f} | {float(r['total_s_mean']):.3f} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if combined:
        with (OUT/'layer_means.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    binary=OUT/'resnet_experiment';runner=OUT/'runner_snapshot.py'
    if not binary.exists():shutil.copy2(ROOT/'build/test',binary)
    if not runner.exists():shutil.copy2(ROOT/'scripts/run_resnet_repeated.py',runner)
    previous=json.loads((OUT/'configuration.json').read_text()) if (OUT/'configuration.json').exists() else {}
    config=dict(campaign_pid=os.getpid(),started_utc=previous.get('started_utc',stamp()),scheduler_updated_utc=stamp(),phases=[32768,16384],planned_cases=16,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    save(OUT/'configuration.json',config)
    for ring in (32768,16384):
        phase=OUT/f'ring{ring}';phase.mkdir(exist_ok=True)
        if not (phase/'REPORT.md').exists():(phase/'REPORT.md').write_text(f'# Ring {ring}\n\nQueued; no completed measurements yet.\n')
    report();start=time.monotonic()-(datetime.now(timezone.utc)-datetime.fromisoformat(config['started_utc'])).total_seconds()
    for ring in (32768,16384):
        phase=OUT/f'ring{ring}'
        env=os.environ.copy();env.update(RESNET_REPO_ROOT=str(ROOT),RESNET_RING_DIM=str(ring),RESNET_OUTPUT=str(phase),RESNET_BINARY=str(binary),RESNET_ADOPT_CASE_PATH=str(OUT/'adopt_current.json'))
        with (phase/'runner.log').open('a') as log:
            proc=subprocess.Popen(['python3','-u',str(runner)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            print('PHASE_START',ring,'pid',proc.pid,flush=True)
            while proc.poll() is None:
                save(OUT/'live.json',dict(status='running',primary_ring=ring,campaign_pid=os.getpid(),runner_pid=proc.pid,phase_status_path=str(phase/'live.json'),elapsed_campaign_s=time.monotonic()-start,updated_utc=stamp()))
                report();time.sleep(10)
            if proc.returncode:
                save(OUT/'live.json',dict(status='runner_failed',primary_ring=ring,exit_code=proc.returncode,updated_utc=stamp()))
                raise SystemExit(proc.returncode)
        print('PHASE_FINISH',ring,flush=True);report()
    result=dict(status='finished',finished_utc=stamp(),elapsed_campaign_s=time.monotonic()-start)
    save(OUT/'completion.json',result);save(OUT/'live.json',result);report()

if __name__=='__main__':main()
