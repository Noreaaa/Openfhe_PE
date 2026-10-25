#!/usr/bin/env python3
"""Sequential optimized ResNet sweep with configurable OpenMP and area limits.
GNU time measures whole-process wall time/peak RSS; profile CSV measures forwards.
Resume skips successful cases with an identical command and binary fingerprint.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["area_size", "layers", "ring_dim", "slots", "depth", "threads", "status",
          "wall_seconds", "inference_seconds", "conv_seconds", "relu_seconds",
          "peak_rss_mib", "peak_rss_gib", "exit_code", "started_utc", "finished_utc", "log", "profile", "error"]


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def summarize(output):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("area*/layers*/result.json"))]
    temp = output / "summary.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["area_size"], r["layers"])))
    temp.replace(output / "summary.csv")
    return rows


def live_memory(pid):
    """Read the model process, not the small GNU time wrapper."""
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        model_pid = int(children[0]) if children else pid
        data = {}
        for line in Path(f"/proc/{model_pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "Threads"):
                data[key] = int(value.split()[0])
        return {"model_pid": model_pid, "rss_mib": data.get("VmRSS", 0)/1024,
                "peak_rss_mib": data.get("VmHWM", 0)/1024, "os_threads": data.get("Threads", 0)}
    except (OSError, ValueError, IndexError):
        return {}


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT/"build/test")
    parser.add_argument("--output", type=Path, default=ROOT/"test_results/resnet_sweep_max")
    parser.add_argument("--depth", type=int, default=46, help="same depth for all cases")
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)), help="OpenMP thread and CPU affinity limit (default: all available CPUs)")
    parser.add_argument("--max-area", type=int, default=100, help="largest square side, multiple of 10")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="also verify plaintext outputs; affects whole-process time")
    parser.add_argument("--status", action="store_true", help="print current persistent monitoring state")
    args = parser.parse_args()
    output = args.output.resolve()
    planned_cases = (args.max_area//10)*5
    if args.status:
        config_path = output/"configuration.json"
        if config_path.exists():
            planned_cases = json.loads(config_path.read_text())["planned_cases"]
        path = output/"live.json"
        print(path.read_text() if path.exists() else "Waiting for runner...")
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("status") == "running":
                try:
                    with Path(state["log"]).with_name("layers.csv").open() as stream:
                        profiles = list(csv.DictReader(stream))
                    if profiles:
                        print("Latest completed layer:", profiles[-1]["conv_layers"],
                              "forward ms:", profiles[-1]["total_ms"],
                              "remaining towers:", profiles[-1]["remaining_towers"])
                except (OSError, KeyError):
                    pass
                try:
                    os.kill(state["runner_pid"], 0)
                except ProcessLookupError:
                    print("ALERT: runner exited; monitoring state is stale. Resume using the launcher.")
        if (output/"summary.csv").exists():
            with (output/"summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            print(f"Completed records: {len(rows)}/{planned_cases}; success={sum(r['status']=='success' for r in rows)}")
            for row in rows[-5:]:
                print(f"{row['area_size']}x{row['area_size']} layers={row['layers']} {row['status']} "
                      f"wall={row['wall_seconds']}s inference={row['inference_seconds']}s peak={row['peak_rss_gib']}GiB")
        return 0
    if args.depth < 1 or args.threads < 1:
        parser.error("depth and threads must be positive")
    if args.max_area < 10 or args.max_area > 100 or args.max_area % 10:
        parser.error("max-area must be 10..100 and a multiple of 10")
    binary = args.binary.resolve()
    if not args.dry_run and (not binary.is_file() or not os.access(binary, os.X_OK)):
        parser.error(f"executable missing: {binary}; build target 'test' first")
    fingerprint = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else "missing"
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_THREAD_LIMIT=str(args.threads), OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LC_ALL="C")
    # Pin the entire child process tree to the requested CPU limit as well.
    cpus = sorted(os.sched_getaffinity(0))[:args.threads] if hasattr(os, "sched_getaffinity") else []
    print(f"Sweep: {planned_cases} cases, one model at a time, depth={args.depth}, {args.threads} OpenMP threads, CPUs={cpus}", flush=True)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    lock = None
    if not args.dry_run:
        lock = (output/".runner.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another runner already owns this output directory")
        def interrupted_signal(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted_signal)
        signal.signal(signal.SIGHUP, interrupted_signal)
    if not args.dry_run:
        configuration = dict(planned_cases=planned_cases,depth=args.depth,threads=args.threads,
                             max_area=args.max_area,ring_dim=16384,slots=1024,layers=[6,7,8,9,10],
                             binary_sha256=fingerprint,cpu_affinity=cpus,updated_utc=stamp())
        write_json(output/"configuration.json",configuration)
        # Ask the executable to validate its own level budget before launching expensive cases.
        preflight = [str(binary),"--model","ResNet-18","--layers","10","--depth",str(args.depth),
                     "--ringDim","16384","--batch","1024","--act","0","--encrypt-size",str(args.max_area),
                     "--data-root",str(ROOT),"--check-config"]
        checked = subprocess.run(preflight,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/"preflight.log").write_text(checked.stdout)
        if checked.returncode:
            write_json(output/"live.json",dict(status="invalid_config",planned_cases=planned_cases,
                       depth=args.depth,threads=args.threads,max_area=args.max_area,exit_code=checked.returncode,
                       reason=checked.stdout.strip(),updated_utc=stamp()))
            with (output/"summary.csv").open("w",newline="") as stream:
                csv.DictWriter(stream,fieldnames=FIELDS).writeheader()
            print("PREFLIGHT FAILED; no benchmark cases launched.\n"+checked.stdout,flush=True)
            return checked.returncode
    interrupted = False
    for area in range(10, args.max_area+1, 10):
        for layers in range(6, 11):
            case = output/f"area{area:03d}"/f"layers{layers:02d}"
            command = [str(binary), "--model", "ResNet-18", "--layers", str(layers), "--depth", str(args.depth),
                       "--ringDim", "16384", "--batch", "1024", "--act", "0", "--encrypt-size", str(area),
                       "--data-root", str(ROOT), "--profile", str(case/"layers.csv")]
            if args.verify:
                command.append("--verify")
            signature = hashlib.sha256(json.dumps([command, fingerprint, env["OMP_NUM_THREADS"], cpus]).encode()).hexdigest()
            if args.dry_run:
                print(shlex.join(command))
                continue
            case.mkdir(parents=True, exist_ok=True)
            result_path = case/"result.json"
            if result_path.exists() and not args.rerun:
                previous = json.loads(result_path.read_text())
                if previous.get("status") == "success" and previous.get("signature") == signature:
                    print(f"SKIP area={area} layers={layers} (already successful)", flush=True)
                    continue
            # Remove stale per-run artifacts before retrying a failed/interrupted case.
            for name in ("layers.csv", "time.txt"):
                (case/name).unlink(missing_ok=True)
            row = dict(area_size=area, layers=layers, ring_dim=16384, slots=1024, depth=args.depth, threads=args.threads,
                       status="running", started_utc=stamp(), signature=signature, binary_sha256=fingerprint,
                       log=str(case/"run.log"), profile=str(case/"layers.csv"), command=command, cpu_affinity=cpus)
            write_json(result_path, row)
            print(f"START area={area} layers={layers} {row['started_utc']}", flush=True)
            start = time.monotonic()
            process = None
            try:
                with (case/"run.log").open("w") as log:
                    def affinity():
                        if cpus:
                            os.sched_setaffinity(0, cpus)
                    process = subprocess.Popen(["/usr/bin/time", "-f", "wall_seconds=%e\npeak_rss_kib=%M\nexit_code=%x",
                                                "-o", str(case/"time.txt"), *command], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                               preexec_fn=affinity if cpus else None)
                    last_print = 0
                    while process.poll() is None:
                        elapsed = time.monotonic()-start
                        live = dict(status="running", area_size=area, layers=layers, depth=args.depth,
                                    threads=args.threads, elapsed_seconds=round(elapsed, 1), runner_pid=os.getpid(),
                                    wrapper_pid=process.pid, updated_utc=stamp(), log=row["log"], **live_memory(process.pid))
                        try:
                            with (case/"layers.csv").open() as stream:
                                completed = list(csv.DictReader(stream))
                            if completed:
                                last = completed[-1]
                                live["completed_layers"] = int(last["conv_layers"])
                                live["forward_seconds_so_far"] = round(float(last["total_ms"])/1000, 3)
                                live["remaining_towers"] = int(last["remaining_towers"])
                        except (OSError, ValueError, KeyError):
                            pass
                        write_json(output/"live.json", live)
                        if elapsed-last_print >= 60:
                            print(f"MONITOR area={area} layers={layers} elapsed={elapsed:.0f}s "
                                  f"rss={live.get('rss_mib',0):.0f}MiB peak={live.get('peak_rss_mib',0):.0f}MiB", flush=True)
                            last_print = elapsed
                        time.sleep(5)
                    row["exit_code"] = process.returncode
            except KeyboardInterrupt:
                interrupted = True
                if process:
                    stop(process)
                    row["exit_code"] = process.returncode
                row["status"] = "interrupted"
            except Exception as error:
                if process:
                    stop(process)
                row.update(status="failed", error=str(error))
            row["wall_seconds"] = round(time.monotonic()-start, 3)
            stats = {}
            if (case/"time.txt").exists():
                stats = dict(line.split("=", 1) for line in (case/"time.txt").read_text().splitlines() if "=" in line)
            if "wall_seconds" in stats:
                row["wall_seconds"] = float(stats["wall_seconds"])
            if "peak_rss_kib" in stats:
                row["peak_rss_mib"] = round(float(stats["peak_rss_kib"])/1024, 3)
                row["peak_rss_gib"] = round(float(stats["peak_rss_kib"])/(1024*1024), 4)
            try:
                with (case/"layers.csv").open() as stream:
                    profiles = list(csv.DictReader(stream))
                last = profiles[-1]
                if int(last["conv_layers"]) != layers or int(last["slots"]) != 1024:
                    raise ValueError("incomplete layer profile or incorrect slot count")
                for dst, src in (("inference_seconds", "total_ms"), ("conv_seconds", "cumulative_conv_ms"),
                                 ("relu_seconds", "cumulative_relu_ms")):
                    row[dst] = round(float(last[src])/1000, 6)
                if row["status"] == "running":
                    row["status"] = "success" if row.get("exit_code") == 0 else "failed"
            except (OSError, IndexError, KeyError, ValueError) as error:
                row.setdefault("error", str(error))
                if row["status"] == "running":
                    row["status"] = "failed"
            row["finished_utc"] = stamp()
            write_json(result_path, row)
            rows = summarize(output)
            write_json(output/"live.json", dict(status=row["status"], area_size=area, layers=layers,
                       completed=len(rows), successful=sum(r["status"]=="success" for r in rows), updated_utc=stamp()))
            print(f"{row['status'].upper()} area={area} layers={layers} wall={row['wall_seconds']}s "
                  f"inference={row.get('inference_seconds','NA')}s peak={row.get('peak_rss_gib','NA')}GiB", flush=True)
            if interrupted:
                print("Stopped; rerun this command to resume.", flush=True)
                return 130
    if not args.dry_run:
        rows = summarize(output)
        successful = sum(r["status"]=="success" for r in rows)
        write_json(output/"live.json", dict(status="finished", successful=successful, records=len(rows), updated_utc=stamp()))
        print(f"FINISHED successful={successful}/{planned_cases} summary={output/'summary.csv'}", flush=True)
        return 0 if successful == planned_cases else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
