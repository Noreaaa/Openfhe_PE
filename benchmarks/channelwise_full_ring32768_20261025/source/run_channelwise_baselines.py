#!/usr/bin/env python3
"""Detached fully encrypted channel-wise baselines, measured in separate processes."""
import csv,fcntl,hashlib,json,os,resource,shutil,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/channelwise_full_ring32768_20261025'
def now():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def rows(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))
def report():
    lines=['# Fully Encrypted Channel-Wise ResNet Prefix Baselines','',f'Updated UTC: {now()}','',
    '## Scope and implementation','',
    'Original convolution 1 (Conv + BN + exact ReLU) executes on the simulated client. All 64 × 112 × 112 resulting features are encrypted, using one ciphertext per channel and 16384 logical slots. The server executes original convolutions 2–8, including the convolution that replaces max pooling. This is the same sequential prefix as previous experiments; residual additions and the classifier are not included. There is no plaintext feature bypass and no 10×10–80×80 partial-encryption sweep.','',
    'The channel-wise layout is derived from the existing `Conv2d_C` in `src/cnn/conv2d.cpp`. The benchmark adapter adds correct same padding and folded batch normalization, and caches rotations across output channels. It retains spatial indices across strided convolutions rather than repacking channels. This is an adapted implementation, not an unmodified historical executable.','',
    'Parameters: ring dimension 32768, 16384 slots, 30/40-bit scaling/first moduli, HYBRID numLargeDigits=6, 96 OpenMP threads. Each activation mode runs once in a separate process. Scheme switching uses depth 40; polynomial approximation uses depth 36. Both omit CKKS bootstrapping. FHEW sign bootstrapping remains part of the scheme-switch activation. Experimental HEStd_NotSet / FHEW TOY parameters do not establish 128-bit security.','',
    'Scheme-switch ReLU follows the current encrypted-mask implementation: CKKS → FHEW → encrypted sign → CKKS → ciphertext multiplication. No sign decryption occurs in the activation. Ciphertexts are processed sequentially for scheme switching; sign evaluations and library kernels use the thread pool.','',
    'Approximation reuses the historical polynomial p(z)=0.0229645+0.5z+1.4340124z²−2.089185z⁴+1.1551725z⁶. Although the historical comment calls it degree 7, its highest nonzero degree is 6. For numerical stability, each convolution output is divided by R=max(1,1.1×max|plaintext preactivation|) via folded coefficients; the activation returns R×p(x/R). Bounds are calibrated offline on this single benchmark image and this polynomial trajectory, not computed by decrypting server intermediate results. They are not dataset-wide guarantees. Unlike the historical `Relu_appx`, this adapter does not bootstrap after every activation; it reserves sufficient initial depth instead. Constants are injected only at active feature indices.','',
    '## Measurement conventions','',
    'Operator wall times exclude plaintext reference construction, calibration and decryption-based verification. Client computation, encryption and key generation are separately marked in logs/RSS samples and excluded from server forward totals. RSS is sampled every 0.5 s; per-layer operator-phase mean/peak and lifetime GNU time peak are distinct. All phases share a process: resident keys, allocator retention and reference tensors remain included in operator-phase RSS. Memory is not measured in an isolated server process. No allocator trimming or forced peak reset is used. No network transfer is timed. Arithmetic checks compare encrypted convolution and activation outputs with the matching plaintext path. Approximation-versus-exact-ReLU error is reported separately on the current polynomial-path preactivation; it is not end-to-end classifier accuracy.','',
    '## Summary','', '| Mode | Status | Server forward (min) | Conv+BN (min) | Activation (min) | Operator RSS mean / peak (GiB) | Whole-process peak (GiB) | Process wall (min) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('approximation','scheme_switch'):
        folder=OUT/mode;r=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        data=rows(folder/'layers.csv')
        forward=sum(float(x['conv_ms'])+float(x['activation_ms']) for x in data)/60000
        lines.append(f"| {mode} | {r.get('status','pending / running')} ({len(data)}/7 layers) | {forward:.3f} | {sum(float(x['conv_ms']) for x in data)/60000:.3f} | {sum(float(x['activation_ms']) for x in data)/60000:.3f} | {r.get('server_mean_gib',0):.3f} / {r.get('server_peak_gib',0):.3f} | {r.get('whole_process_peak_gib',0):.3f} | {r.get('wall_s',0)/60:.3f} |")
    lines+=['','## Per-layer results','','| Mode | Original layer | Output C×H×W | Conv+BN (s) | Activation (s) | Cumulative (s) | Operator RSS mean / peak (GiB) | Level / towers | Calibration R | Conv max error | Activation arithmetic RMSE | vs exact ReLU RMSE |','|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for mode in ('approximation','scheme_switch'):
        folder=OUT/mode;meta=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        for x in rows(folder/'layers.csv'):
            mem=meta.get('layers',{}).get(x['layer'],{})
            lines.append(f"| {mode} | {x['layer']} | {x['channels']}×{x['height']}×{x['width']} | {float(x['conv_ms'])/1000:.3f} | {float(x['activation_ms'])/1000:.3f} | {float(x['cumulative_ms'])/1000:.3f} | {mem.get('mean_gib',0):.3f} / {mem.get('peak_gib',0):.3f} | {x['level']} / {x['remaining_towers']} | {float(x['normalization_bound']):.4f} | {float(x['conv_max_abs']):.6g} | {float(x['activation_rmse']):.6g} | {float(x['vs_relu_rmse']):.6g} |")
    lines+=['','Raw artifacts per mode: `run.log`, `layers.csv`, `rss.csv`, `time.txt`, `result.json`, `command.json`. Source/binary SHA256 values are recorded in `configuration.json`. Pending or failed cases must not be interpreted as completed performance results.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def rss(pid):
    try:
        d={}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('VmRSS','VmHWM','Threads'):d[k]=int(v.split()[0])
        return d
    except (OSError,ValueError):return {}
def available():
    for l in Path('/proc/meminfo').read_text().splitlines():
        if l.startswith('MemAvailable:'):return int(l.split()[1])/1024**2
    return 0
def summarize(samples):
    if not samples:return dict(mean_gib=0,peak_gib=0,sampled_seconds=0)
    integral=duration=0
    for a,b in zip(samples,samples[1:]):
        dt=b['elapsed_s']-a['elapsed_s']
        if 0<dt<2 and a['phase']==b['phase'] and a['layer']==b['layer']:
            duration+=dt;integral+=dt*(a['rss_gib']+b['rss_gib'])/2
    return dict(mean_gib=integral/duration if duration else sum(s['rss_gib'] for s in samples)/len(samples),peak_gib=max(s['rss_gib'] for s in samples),sampled_seconds=duration)
def run(mode,binary):
    folder=OUT/mode;folder.mkdir(exist_ok=True);depth=36 if mode=='approximation' else 40
    command=[str(binary),str(ROOT),mode,str(folder/'layers.csv'),'8',str(depth)]
    save(folder/'command.json',dict(command=command,started_utc=now()))
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/root/local/openFHE/lib:/root/local/cnpy/lib',OMP_NUM_THREADS='96',OMP_DYNAMIC='FALSE')
    start=time.monotonic();samples=[];pressure=False;lastreport=0
    with (folder/'run.log').open('w') as log,(folder/'rss.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['utc','elapsed_s','phase','layer','rss_gib','process_peak_gib','threads']);writer.writeheader()
        proc=subprocess.Popen(['/usr/bin/time','-f','wall_seconds=%e\npeak_rss_kib=%M\nexit_code=%x','-o',str(folder/'time.txt'),*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            try:
                children=Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                phase=json.loads((folder/'layers.csv.phase.json').read_text()) if (folder/'layers.csv.phase.json').exists() else dict(phase='startup',layer=0)
                info=rss(int(children[0])) if children else {}
                if 'VmRSS' in info:
                    s=dict(utc=now(),elapsed_s=time.monotonic()-start,phase=phase['phase'],layer=phase['layer'],rss_gib=info['VmRSS']/1024**2,process_peak_gib=info['VmHWM']/1024**2,threads=info['Threads'])
                    samples.append(s);writer.writerow(s);f.flush()
                    save(OUT/'live.json',dict(status='running',mode=mode,model_pid=int(children[0]),runner_pid=os.getpid(),**s))
                if available()<12:
                    pressure=True;os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if time.monotonic()-lastreport>30:report();lastreport=time.monotonic()
            except (OSError,ValueError,KeyError):pass
            time.sleep(.5)
        code=proc.wait()
    data=rows(folder/'layers.csv');metrics={}
    if (folder/'time.txt').exists():metrics=dict(x.split('=',1) for x in (folder/'time.txt').read_text().splitlines() if '=' in x)
    ops=[s for s in samples if s['phase'] in ('server_conv','server_relu')];memory=summarize(ops)
    result=dict(mode=mode,status='success' if code==0 and [int(x['layer']) for x in data]==list(range(2,9)) else 'memory_pressure' if pressure else 'failed',exit_code=code,completed_layers=len(data),finished_utc=now(),wall_s=float(metrics.get('wall_seconds',time.monotonic()-start)),whole_process_peak_gib=float(metrics.get('peak_rss_kib',0))/1024**2,server_mean_gib=memory['mean_gib'],server_peak_gib=memory['peak_gib'],layers={str(i):summarize([s for s in ops if s['layer']==i]) for i in range(2,9)})
    save(folder/'result.json',result);report();print('CASE_FINISHED',mode,result['status'],flush=True);return result

def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    binary=OUT/'channelwise_prefix_bench';shutil.copy2(ROOT/'build/channelwise_prefix_bench',binary)
    save(OUT/'configuration.json',dict(started_utc=now(),runner_pid=os.getpid(),ring=32768,slots=16384,depths=dict(approximation=36,scheme_switch=40),scale_bits=30,first_bits=40,digits=6,threads=96,repeats=1,client_layer=1,server_layers=list(range(2,9)),fully_encrypted=True,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'tests/sparse_packing/channelwise_prefix_bench.cpp').read_bytes()).hexdigest()))
    report()
    for mode in ('approximation','scheme_switch'):
        result=run(mode,binary)
        if result['status']!='success':save(OUT/'live.json',dict(status='failed',mode=mode,result=result));return
    save(OUT/'completion.json',dict(status='finished',finished_utc=now()));save(OUT/'live.json',dict(status='finished',finished_utc=now()));report()
if __name__=='__main__':main()
