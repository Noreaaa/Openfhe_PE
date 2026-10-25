import sys,resource,shutil,hashlib,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_resnet_client_first as r
r.OUT=Path(__file__).resolve().parent
lock=(r.OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
binary=r.OUT/'resnet_experiment'
shutil.copy2(r.ROOT/'benchmarks/resnet_client_first_slots8192_20261024/resnet_experiment',binary)
r.save(r.OUT/'configuration.json',dict(ring=2048,slots=1024,depth=40,scale_bits=30,first_bits=40,digits=6,threads=96,area=10,client_layers=1,total_layers=8,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
result=r.run_case(binary,10,8,r.OUT/'area10',ring=2048,slots=1024)
r.save(r.OUT/'live.json',dict(status='finished' if result['status']=='success' else 'failed',result=result))
