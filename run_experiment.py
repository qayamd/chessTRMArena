#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#unattended experiment orchestrator -- runs all training jobs then does arena eval
#crash-safe and restart-safe; already-completed runs are skipped on re-launch

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

# job definitions: (run_name, description, argv for train.py)
# steps calibrated for RTX 4070 Ti SUPER (~72h total), adjust as needed

def build_jobs(data: str, val_data: str | None) -> list[tuple[str, str, list[str]]]:
    #builds the ordered list of (run_name, description, argv) training jobs

    data_flags = ["--data", data]
    if val_data:
        data_flags += ["--val-data", val_data]

    common = data_flags + [
        "--num-return-buckets", "128",
        "--val-fraction", "0.001",
        "--device", "auto",
        "--ckpt-every", "10000",
        "--keep-ckpts", "3",
    ]

    jobs = []

    # ── Phase 1: matched pairs (≈31h) ────────────────────────────────────

    jobs.append(("p1_tiny_trm", "Phase 1: tiny TRM (d=64 r=8)", [
        "--model", "trm",
        "--embedding-dim", "64", "--num-heads", "8",
        "--num-layers", "2", "--n-recurrence", "8",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p1_tiny_tx", "Phase 1: tiny Transformer (d=64 L=4)", [
        "--model", "transformer",
        "--embedding-dim", "64", "--num-heads", "8",
        "--num-layers", "4",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p1_small_trm", "Phase 1: small TRM (d=128 r=8)", [
        "--model", "trm",
        "--embedding-dim", "128", "--num-heads", "8",
        "--num-layers", "2", "--n-recurrence", "8",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "5e-4", "--warmup-steps", "2000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p1_small_tx", "Phase 1: small Transformer (d=128 L=6)", [
        "--model", "transformer",
        "--embedding-dim", "128", "--num-heads", "8",
        "--num-layers", "6",
        "--batch-size", "512", "--lr", "5e-4", "--warmup-steps", "2000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    # ── Phase 2: TRM shrink (≈13h) ──────────────────────────────────────

    jobs.append(("p2_d32_r8", "Phase 2: TRM shrink d=32 r=8", [
        "--model", "trm",
        "--embedding-dim", "32", "--num-heads", "4",
        "--num-layers", "2", "--n-recurrence", "8",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p2_d32_r4", "Phase 2: TRM shrink d=32 r=4", [
        "--model", "trm",
        "--embedding-dim", "32", "--num-heads", "4",
        "--num-layers", "2", "--n-recurrence", "4",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p2_d32_r2", "Phase 2: TRM shrink d=32 r=2", [
        "--model", "trm",
        "--embedding-dim", "32", "--num-heads", "4",
        "--num-layers", "2", "--n-recurrence", "2",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p2_d16_r8", "Phase 2: TRM shrink d=16 r=8", [
        "--model", "trm",
        "--embedding-dim", "16", "--num-heads", "2",
        "--num-layers", "2", "--n-recurrence", "8",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "1e-3", "--warmup-steps", "1000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    # ── Phase 3: medium ceiling (≈30h) ──────────────────────────────────

    jobs.append(("p3_medium_tx", "Phase 3: medium Transformer (d=256 L=8)", [
        "--model", "transformer",
        "--embedding-dim", "256", "--num-heads", "8",
        "--num-layers", "8",
        "--batch-size", "256", "--lr", "3e-4", "--warmup-steps", "5000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    jobs.append(("p3_medium_trm", "Phase 3: medium TRM (d=256 r=8)", [
        "--model", "trm",
        "--embedding-dim", "256", "--num-heads", "8",
        "--num-layers", "2", "--n-recurrence", "8",
        "--deep-supervision", "--use-ema",
        "--batch-size", "512", "--lr", "3e-4", "--warmup-steps", "5000",
        "--steps", "200000",
        "--log-every", "100",
    ]))

    # Attach common flags to each job.
    return [(name, desc, argv + common + ["--run-name", name]) for name, desc, argv in jobs]


# ── Helpers ──────────────────────────────────────────────────────────────────

def final_ckpt_path(run_name: str, output_dir: str) -> Path | None:
    #returns the final checkpoint path if the run already finished
    run_dir = Path(output_dir) / run_name
    finals = sorted(run_dir.glob("step_*_final.pt"))
    return finals[-1] if finals else None


def all_ckpt_paths(output_dir: str) -> list[Path]:
    #finds all final checkpoints across all run dirs
    return sorted(Path(output_dir).rglob("step_*_final.pt"))


def log(msg: str, log_file) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def _stream_pipe(pipe, *destinations):
    #tees pipe output to multiple destinations (stdout + log file)
    for raw_line in iter(pipe.readline, ""):
        for dest in destinations:
            dest.write(raw_line)
            dest.flush()
    pipe.close()


def run_training_job(cmd: list[str], job_log_path: Path) -> int:
    #runs a training subprocess, streaming output to both terminal and job log
    job_log = open(job_log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # stream merged stdout+stderr to both terminal and the per-job log
    _stream_pipe(proc.stdout, sys.stdout, job_log)
    proc.wait()
    job_log.close()
    return proc.returncode


def write_status(output_dir: str, jobs: list, current_idx: int,
                 state: str, elapsed_h: float = 0.0,
                 completed: list[str] | None = None,
                 failed: list[str] | None = None) -> None:
    #writes status.json so you can check experiment progress externally
    completed = completed or []
    failed = failed or []
    run_name = jobs[current_idx][0] if current_idx < len(jobs) else None
    desc = jobs[current_idx][1] if current_idx < len(jobs) else None

    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(jobs),
        "current_job_index": current_idx + 1,
        "current_job": run_name,
        "current_job_desc": desc,
        "state": state,  # "running", "done", "failed", "arena", "complete"
        "elapsed_hours": round(elapsed_h, 2),
        "completed": completed,
        "failed": failed,
        "remaining": [j[0] for j in jobs[current_idx + 1:]],
    }
    status_path = os.path.join(output_dir, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


# ── Arena evaluation ─────────────────────────────────────────────────────────

def run_arena(ckpt_a: Path, ckpt_b: Path, games: int, log_file,
              device: str = "auto") -> str:
    #runs a head-to-head match between two checkpoints
    cmd = [
        PYTHON, "arena.py",
        "--model-a", str(ckpt_a),
        "--model-b", str(ckpt_b),
        "--games", str(games),
        "--opening-depth", "4",
        "--device", device,
    ]
    log(f"Arena: {ckpt_a.parent.name} vs {ckpt_b.parent.name} ({games} games)", log_file)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    output = result.stdout + result.stderr
    log(output.strip().split("\n")[-1], log_file)  # log the summary line
    return output


def run_puzzle_eval(ckpt: Path, puzzle_csv: str, num_puzzles: int,
                    log_file, device: str = "auto") -> str:
    #runs puzzle eval for a single checkpoint
    cmd = [
        PYTHON, "arena.py",
        "--model-a", str(ckpt),
        "--model-b", "random",
        "--puzzles", puzzle_csv,
        "--num-puzzles", str(num_puzzles),
        "--device", device,
    ]
    log(f"Puzzles: {ckpt.parent.name} ({num_puzzles} puzzles)", log_file)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    output = result.stdout + result.stderr
    for line in output.strip().split("\n"):
        if "accuracy" in line.lower() or "solved" in line.lower():
            log(f"  {line.strip()}", log_file)
    return output


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="3-day experiment orchestrator for TRM vs Transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Training .bag file.")
    p.add_argument("--val-data", default=None, help="Validation .bag file.")
    p.add_argument("--output-dir", default="results", help="Base output directory.")
    p.add_argument("--arena-games", type=int, default=50,
                   help="Games per arena matchup.")
    p.add_argument("--puzzles", default="puzzles.csv",
                   help="Lichess puzzle CSV for evaluation.")
    p.add_argument("--num-puzzles", type=int, default=500,
                   help="Number of puzzles to evaluate per model.")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-arena", action="store_true",
                   help="Only run training, skip arena evaluation.")
    p.add_argument("--skip-training", action="store_true",
                   help="Only run arena evaluation on existing checkpoints.")
    args = p.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "experiment_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    log("=" * 60, log_file)
    log("EXPERIMENT START", log_file)
    log("=" * 60, log_file)

    jobs = build_jobs(args.data, args.val_data)
    total_jobs = len(jobs)

    # ── Training phase ───────────────────────────────────────────────────
    completed_runs: list[str] = []
    failed_runs: list[str] = []

    if not args.skip_training:
        log(f"Training: {total_jobs} jobs queued", log_file)

        for i, (run_name, desc, argv) in enumerate(jobs, 1):
            # skip if already done
            existing = final_ckpt_path(run_name, output_dir)
            if existing:
                log(f"[{i}/{total_jobs}] SKIP {desc} (already done: {existing.name})",
                    log_file)
                completed_runs.append(run_name)
                continue

            log(f"[{i}/{total_jobs}] START {desc}", log_file)
            t0 = time.time()

            # make sure run dir exists for the job log
            run_dir = Path(output_dir) / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            job_log_path = run_dir / "orchestrator.log"

            cmd = [PYTHON, "train.py", "--output-dir", output_dir] + argv
            try:
                write_status(output_dir, jobs, i - 1, "running",
                             completed=completed_runs, failed=failed_runs)

                returncode = run_training_job(cmd, job_log_path)
                elapsed = time.time() - t0

                if returncode != 0:
                    log(f"[{i}/{total_jobs}] FAILED {run_name} "
                        f"(exit {returncode}, {elapsed/3600:.1f}h)", log_file)
                    failed_runs.append(run_name)
                    # log last 20 lines for diagnosis
                    if job_log_path.exists():
                        tail = job_log_path.read_text(encoding="utf-8", errors="replace")
                        for line in tail.strip().split("\n")[-20:]:
                            log(f"  {line}", log_file)
                    write_status(output_dir, jobs, i - 1, "failed",
                                 elapsed_h=elapsed / 3600,
                                 completed=completed_runs, failed=failed_runs)
                    continue

                log(f"[{i}/{total_jobs}] DONE  {run_name} ({elapsed/3600:.1f}h)", log_file)
                completed_runs.append(run_name)
                write_status(output_dir, jobs, i - 1, "done",
                             elapsed_h=elapsed / 3600,
                             completed=completed_runs, failed=failed_runs)

            except Exception as e:
                log(f"[{i}/{total_jobs}] EXCEPTION {run_name}: {e}", log_file)
                failed_runs.append(run_name)
                continue

        log("Training phase complete.", log_file)

    # ── Arena phase ──────────────────────────────────────────────────────
    if not args.skip_arena:
        log("", log_file)
        log("=" * 60, log_file)
        log("ARENA EVALUATION", log_file)
        log("=" * 60, log_file)
        write_status(output_dir, jobs, total_jobs - 1, "arena",
                     completed=completed_runs, failed=failed_runs)

        ckpts = all_ckpt_paths(output_dir)
        # filter out test/resume runs, only keep experiment runs
        ckpts = [c for c in ckpts if c.parent.name.startswith("p")]
        log(f"Found {len(ckpts)} final checkpoints.", log_file)

        if len(ckpts) < 2 and not args.puzzles:
            log("Not enough checkpoints for arena matches.", log_file)
        else:
            # round-robin: every pair plays once
            arena_results = []
            pairs = list(itertools.combinations(ckpts, 2))
            log(f"Round-robin: {len(pairs)} matchups, {args.arena_games} games each",
                log_file)

            for j, (a, b) in enumerate(pairs, 1):
                log(f"  Match {j}/{len(pairs)}", log_file)
                try:
                    output = run_arena(a, b, args.arena_games, log_file,
                                       device=args.device)
                    arena_results.append({
                        "a": a.parent.name, "b": b.parent.name, "output": output,
                    })
                except Exception as e:
                    log(f"  Arena error: {e}", log_file)

            # save structured results
            arena_path = os.path.join(output_dir, "arena_results.json")
            with open(arena_path, "w") as f:
                json.dump(arena_results, f, indent=2)
            log(f"Arena results saved to {arena_path}", log_file)

        # Puzzle evaluation for each checkpoint.
        if args.puzzles and os.path.exists(args.puzzles):
            log("", log_file)
            log("PUZZLE EVALUATION", log_file)
            puzzle_results = []
            for ckpt in ckpts:
                try:
                    output = run_puzzle_eval(
                        ckpt, args.puzzles, args.num_puzzles, log_file,
                        device=args.device,
                    )
                    puzzle_results.append({
                        "model": ckpt.parent.name, "output": output,
                    })
                except Exception as e:
                    log(f"  Puzzle error for {ckpt.parent.name}: {e}", log_file)

            puzzle_path = os.path.join(output_dir, "puzzle_results.json")
            with open(puzzle_path, "w") as f:
                json.dump(puzzle_results, f, indent=2)
            log(f"Puzzle results saved to {puzzle_path}", log_file)

    log("", log_file)
    log("=" * 60, log_file)
    log("EXPERIMENT COMPLETE", log_file)
    log("=" * 60, log_file)
    write_status(output_dir, jobs, total_jobs - 1, "complete",
                 completed=completed_runs, failed=failed_runs)
    log_file.close()


if __name__ == "__main__":
    main()
