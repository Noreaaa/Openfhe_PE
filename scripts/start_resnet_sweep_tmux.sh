#!/usr/bin/env bash
# Usage: ./scripts/start_resnet_sweep_tmux.sh [output_directory] [session_name] [runner options...]
set -euo pipefail
sweep_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sweep_output="${1:-$sweep_root/test_results/resnet_sweep_t16}"
if (( $# )); then shift; fi
sweep_session="${1:-resnet_sweep16}"
if (( $# )); then shift; fi
command -v tmux >/dev/null
command -v python3 >/dev/null
test -x /usr/bin/time
cmake --build "$sweep_root/build" --target test -j 4
python3 - "$sweep_root" "$sweep_output" "$sweep_session" "$@" <<'PY'
import pathlib,shlex,subprocess,sys
root=pathlib.Path(sys.argv[1]).resolve()
output=pathlib.Path(sys.argv[2]).resolve()
session=sys.argv[3]
if subprocess.run(['tmux','has-session','-t','='+session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
    raise SystemExit(f'Tmux session already exists: {session}; attach with: tmux attach -t {session}')
output.mkdir(parents=True,exist_ok=True)
runner=['python3','-u',str(root/'scripts/run_resnet_sweep.py'),'--output',str(output),*sys.argv[4:]]
command='set -o pipefail; '+shlex.join(runner)+' 2>&1 | tee -a '+shlex.quote(str(output/'runner.log'))
subprocess.run(['tmux','new-session','-d','-s',session,'-n','run','-c',str(root),'bash','-lc',command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session+':run','remain-on-exit','on'],check=True)
monitor=['watch','-n','15','-x','python3',str(root/'scripts/run_resnet_sweep.py'),'--status','--output',str(output)]
subprocess.run(['tmux','new-window','-t',session,'-n','monitor','-c',str(root),*monitor],check=True)
subprocess.run(['tmux','select-window','-t',session+':run'],check=True)
print(f'Started {session}: one model process at a time; options: {shlex.join(sys.argv[4:])}')
print(f'Attach: tmux attach -t {session}')
print(f'Summary: {output / "summary.csv"}')
print(f'Status: {output / "live.json"}')
PY
