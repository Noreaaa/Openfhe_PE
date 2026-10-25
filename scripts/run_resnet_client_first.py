#!/usr/bin/env python3
"""Validate then run client-first-layer / 8192-slot ResNet prefixes, once per area."""
import csv,fcntl,hashlib,json,math,os,re,resource,shutil,signal,subprocess,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/resnet_client_first_slots8192_20261024'
BASELINE=ROOT/'benchmarks/resnet_ring_comparison_slots2048_20261023/ring16384'

def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def read_csv(path):
    if not path.exists():return []
    with path.open()as f:return list(csv.DictReader(f))
def memory(pid):
    try:
        data={}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('VmRSS','VmHWM','Threads'):data[k]=int(v.split()[0])
        return data
    except (OSError,ValueError):return {}
def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])
    return 0

def report():
    cases=[json.loads(p.read_text())for p in sorted(OUT.glob('area*/result.json'))]
    if cases:
        with (OUT/'runs.csv').open('w',newline='')as f:
            fields=['area','status','server_forward_s','client_first_s','encryption_s','client_plus_forward_s','server_sampled_mean_gib','server_sampled_peak_gib','whole_process_peak_gib','uploaded_ciphertexts','payload_coefficient_bytes','wall_s','max_abs_error','max_relative_rmse','attempt_dir']
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    lines=['# Client first layer + 8192 slots', '',f'Updated UTC: {now()}', '',
      'Parameters: ring 16384, 30/40-bit moduli, depth 40, HYBRID digits 6, 8192 logical slots, 96 OpenMP threads. Direct rotations; no channel concurrency limit; no CKKS bootstrap. Original convolution 1 (Conv+BN+ReLU) runs in plaintext on the simulated client. Original convolutions 2..8 run on mixed plaintext/ciphertext features. The prefix still has no residual additions or classifier.', '',
      'Client, encryption and server timings are measured separately on the same machine. No network transfer is measured. Payload is ciphertext coefficient storage, not serialized wire size. The original sensitive rectangle is propagated by receptive-field overlap before encryption. Rows retain original full-width coordinates.', '',
      'RSS is sampled every 0.5 seconds. Server samples include only explicitly marked Conv/ReLU execution, excluding key generation, client processing and reference verification time windows. These remain same-process measurements: resident keys, allocator retention and reference tensors are not isolated into a separate server process. Reported sampled peak is distinct from lifetime whole-process peak. No allocator trimming or peak reset is applied.', '',
      'The combined client-offload + slot-count change is compared against the completed 2048-slot all-server baseline. This is not a factorial ablation: improvements cannot be attributed to just one of the two changes.', '',
      '| Area | Status | Client conv1 (s) | Encrypt (s) | Server forward (min) | Client+encrypt+forward (min) | Baseline forward (min) | Server sampled mean/max (GiB) | Whole-process peak (GiB) | Upload ciphertexts |',
      '|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in cases:
        baseline=BASELINE/f"repeat1/area{r['area']}/layers8/result.json"
        old=json.loads(baseline.read_text())if baseline.exists()else{}
        lines.append(f"| {r['area']}×{r['area']} | {r['status']} | {r.get('client_first_s',0):.3f} | {r.get('encryption_s',0):.3f} | {r.get('server_forward_s',0)/60:.3f} | {r.get('client_plus_forward_s',0)/60:.3f} | {old.get('forward_s',0)/60:.3f} | {r.get('server_sampled_mean_gib',0):.3f} / {r.get('server_sampled_peak_gib',0):.3f} | {r.get('whole_process_peak_gib',0):.3f} | {r.get('uploaded_ciphertexts',0)} |")
    lines+=['', '## Server layer measurements', '', '| Area | Original layer | Conv+BN (s) | ReLU (s) | Cumulative forward (s) | Output ciphertexts | Level |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in cases:
        for x in read_csv(Path(r['attempt_dir'])/'layers.csv'):
            lines.append(f"| {r['area']} | {x['conv_layers']} | {float(x['conv_ms'])/1000:.3f} | {float(x['relu_ms'])/1000:.3f} | {float(x['total_ms'])/1000:.3f} | {x['ciphertexts']} | {x['level']} |")
    temp=OUT/'REPORT.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'REPORT.md')

def run_case(binary,area,total_layers,case,ring=16384,slots=8192,depth=40,client_first=True,experimental_depth=False,region=None):
    case.mkdir(parents=True,exist_ok=True)
    command=[str(binary),'--model','ResNet-18','--ringDim',str(ring),'--batch',str(slots),'--layers',str(total_layers),'--depth',str(depth),'--scale-bits','30','--first-bits','40','--conv-scratch-mib','0','--encrypt-size',str(area),'--bootstrap-after','0','--verify','--data-root',str(ROOT),'--profile',str(case/'layers.csv')]
    if region is not None:
        if len(region)!=4:raise ValueError('region must be top,bottom,left,right')
        for name,value in zip(('top','bottom','left','right'),region):command.extend(['--'+name,str(value)])
    if client_first:command.append('--client-first-layer')
    if experimental_depth:command.append('--experimental-depth')
    save(case/'configuration.json',dict(command=command,started_utc=now(),original_sensitive_area=area,total_layers=total_layers,client_layers=int(client_first),server_layers=total_layers-int(client_first),explicit_region=region))
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/root/local/openFHE/lib:/root/local/cnpy/lib',OMP_NUM_THREADS='96',OMP_DYNAMIC='FALSE')
    start=time.monotonic();memory_rows=[];pressure=False;model_pid=None
    with (case/'run.log').open('w')as log,(case/'rss.csv').open('w',newline='')as samplefile:
        writer=csv.DictWriter(samplefile,fieldnames=['utc','monotonic_s','phase','layer','model_pid','rss_gib','process_peak_gib','threads']);writer.writeheader();samplefile.flush()
        proc=subprocess.Popen(['/usr/bin/time','-f','wall_seconds=%e\npeak_rss_kib=%M\nexit_code=%x','-o',str(case/'time.txt'),*command],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
        try:
            while proc.poll()is None:
                try:
                    children=Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                    if children:model_pid=int(children[0])
                    phase=json.loads((case/'layers.csv.phase.json').read_text())if (case/'layers.csv.phase.json').exists()else{'phase':'startup','layer':0}
                    info=memory(model_pid)if model_pid else{}
                    if 'VmRSS'in info:
                        r=dict(utc=now(),monotonic_s=time.monotonic(),phase=phase['phase'],layer=phase['layer'],model_pid=model_pid,rss_gib=info['VmRSS']/1024**2,process_peak_gib=info['VmHWM']/1024**2,threads=info['Threads'])
                        writer.writerow(r);samplefile.flush();memory_rows.append(r)
                        save(OUT/'live.json',dict(status='running',area=area,total_layers=total_layers,model_pid=model_pid,wrapper_pid=proc.pid,runner_pid=os.getpid(),elapsed_s=time.monotonic()-start,attempt_dir=str(case),**{k:r[k]for k in ('phase','layer','rss_gib','process_peak_gib','utc')}))
                    if mem_available()<12*1024**2:
                        pressure=True;os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
                except (OSError,ValueError,KeyError):pass
                time.sleep(.5)
        except BaseException:
            if proc.poll()is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait()
            raise
        code=proc.wait()
    layers=read_csv(case/'layers.csv');client_rows=read_csv(case/'layers.csv.client.csv');client=client_rows[0]if client_rows else{}
    last=layers[-1]if layers else{}
    text=(case/'run.log').read_text(errors='replace');metrics=dict(line.split('=',1)for line in (case/'time.txt').read_text().splitlines()if'='in line)
    checks=[]
    for m in re.finditer(r'PREFIX_ACCURACY layers=(\d+) max_abs=([\deE+.-]+) rmse=([\deE+.-]+) reference_max_abs=([\deE+.-]+)',text):
        checks.append(dict(layer=int(m[1]),max_abs=float(m[2]),rmse=float(m[3]),reference_max_abs=float(m[4])))
    max_error=max((x['max_abs']for x in checks),default=math.inf)
    relative=max((x['rmse']/max(1,x['reference_max_abs'])for x in checks),default=math.inf)
    expected=list(range(2 if client_first else 1,total_layers+1))
    structure_ok=[int(r['conv_layers'])for r in layers]==expected and all(int(r['slots'])==slots for r in layers)
    blocks=math.ceil(int(client.get('channels',64))/max(1,slots//int(client.get('width',112))))
    expected_cts=(((area+6)//2)*blocks if area%2==0 else 0) if client_first else area*blocks
    geometry_ok=bool(client)and int(client['ciphertexts'])==expected_cts
    # Default even center sizes 10..80 map to area/2+3 affected rows; all 64 channels fit one ciphertext.
    accuracy_ok=[x['layer']for x in checks]==expected and relative<.001 and all(x['max_abs']<.1*max(1,x['reference_max_abs'])for x in checks)
    good=code==0 and structure_ok and geometry_ok and accuracy_ok
    server=[r for r in memory_rows if r['phase']in ('server_conv','server_relu','server_bootstrap','server_other')]
    area_s=0;integral=0
    for a,b in zip(memory_rows,memory_rows[1:]):
        dt=b['monotonic_s']-a['monotonic_s']
        if a['phase'].startswith('server_')and a['phase']!='server_ready' and b['phase'].startswith('server_')and b['phase']!='server_ready' and 0<dt<3:
            area_s+=dt;integral+=dt*(a['rss_gib']+b['rss_gib'])/2
    peak=float(metrics.get('peak_rss_kib',0))/1024**2
    forward=float(last.get('total_ms',0))/1000;client_s=float(client.get('client_first_ms',0))/1000;encrypt_s=float(client.get('encryption_ms',0))/1000
    result=dict(area=area,status='success'if good else'memory_pressure'if pressure else'failed',exit_code=code,finished_utc=now(),attempt_dir=str(case),wall_s=float(metrics.get('wall_seconds',time.monotonic()-start)),server_forward_s=forward,client_first_s=client_s,encryption_s=encrypt_s,client_plus_forward_s=client_s+encrypt_s+forward,server_sampled_mean_gib=integral/area_s if area_s else 0,server_sampled_peak_gib=max((r['rss_gib']for r in server),default=0),server_sampled_seconds=area_s,whole_process_peak_gib=peak,uploaded_ciphertexts=int(client.get('ciphertexts',0)),payload_coefficient_bytes=int(client.get('coefficient_bytes',0)),max_abs_error=max_error,max_relative_rmse=relative,structure_ok=structure_ok,geometry_ok=geometry_ok,accuracy_ok=accuracy_ok,accuracy_checks=checks)
    save(case/'result.json',result)
    print('CASE_FINISHED',area,total_layers,result['status'],'wall_s',result['wall_s'],flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    binary=OUT/'resnet_experiment'
    if not binary.exists():shutil.copy2(ROOT/'build/test',binary)
    save(OUT/'configuration.json',dict(runner_pid=os.getpid(),started_utc=now(),ring=16384,slots=8192,depth=40,digits=6,scale_bits=30,first_bits=40,client_layers=1,total_layers=8,threads=96,areas=list(range(10,81,10)),repeats=1,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    report()
    smoke=OUT/'validation_area10_layers2'
    previous=json.loads((smoke/'result.json').read_text())if(smoke/'result.json').exists()else{}
    if previous.get('status')!='success':
        result=run_case(binary,10,2,smoke)
        if result['status']!='success':
            save(OUT/'live.json',dict(status='validation_failed',result=result));return
    for area in range(10,81,10):
        case=OUT/f'area{area:02d}'
        if(case/'result.json').exists()and json.loads((case/'result.json').read_text()).get('status')=='success':continue
        result=run_case(binary,area,8,case);report()
        if result['status']!='success':
            save(OUT/'live.json',dict(status='case_failed',area=area,result=result));return
    save(OUT/'completion.json',dict(status='finished',finished_utc=now()));save(OUT/'live.json',dict(status='finished',finished_utc=now()));report()
if __name__=='__main__':main()
