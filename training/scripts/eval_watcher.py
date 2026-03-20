#!/usr/bin/env python3
"""
Epoch-level evaluation watcher for training runs.

Watches training run directories for completed epoch checkpoints (epoch_NNN.pt).
Since the training script saves every epoch checkpoint, no snapshotting needed.

For each new epoch checkpoint found:
  - Runs a monitoring eval (10K samples, ~20 min) for quick feedback
  - When all runs finish training, runs FULL eval (all samples)

Usage:
    nohup python training/scripts/eval_watcher.py > runs/sparse_voxel_det/eval_watcher.log 2>&1 &

NOTE: You may need to adjust RUNS configuration below (GPU indices, run directories,
batch sizes) to match your system. The default configuration expects v83 training runs.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================================
# Configuration
# ============================================================================
RUNS = [
    {
        "name": "v83",
        "run_dir": PROJECT_ROOT / "runs" / "sparse_voxel_det" / "v83_seed42",
        "eval_cuda": "6",  # Physical GPU 2 (RTX 3090 24GB)
        "data_dir": "data/datasets/fred_paper_parity_v82/sparse",
        "label_dir": "data/datasets/fred_paper_parity/labels",
        "split": "canonical_test",
        "batch_size": 2,  # Native 1280x720 needs more VRAM per sample
    },
    {
        "name": "v83-640",
        "run_dir": PROJECT_ROOT / "runs" / "sparse_voxel_det" / "v83_640_seed42",
        "eval_cuda": "6",  # Physical GPU 2 (RTX 3090 24GB) — sequential
        "data_dir": "data/datasets/fred_paper_parity_v82_640/sparse",
        "label_dir": "data/datasets/fred_paper_parity/labels",
        "split": "canonical_test",
        "batch_size": 4,  # 640x640 fits comfortably
    },
    {
        "name": "v83-seed123",
        "run_dir": PROJECT_ROOT / "runs" / "sparse_voxel_det" / "v83_seed123",
        "eval_cuda": "6",  # Physical GPU 2 (RTX 3090 24GB) — sequential
        "data_dir": "data/datasets/fred_paper_parity_v82/sparse",
        "label_dir": "data/datasets/fred_paper_parity/labels",
        "split": "canonical_test",
        "batch_size": 2,  # Native 1280x720 needs more VRAM per sample
    },
]

EVAL_SCRIPT = PROJECT_ROOT / "training" / "scripts" / "evaluate_sparse_voxel_det.py"
PYTHON = sys.executable  # Use the current Python interpreter
POLL_INTERVAL = 120          # Check every 2 minutes
MONITOR_MAX_SAMPLES = 10000  # Fast eval for monitoring (~20 min)
MONITOR_TIMEOUT = 3600       # 1 hour max for monitoring eval
FULL_EVAL_TIMEOUT = 21600    # 6 hours max for full eval (119K samples)
TOTAL_EPOCHS = 50            # Total epochs per run (v83: 50 epochs)
FINAL_EPOCH = TOTAL_EPOCHS - 1  # 0-indexed


def find_epoch_checkpoints(run_dir: Path) -> dict:
    """Find all epoch_NNN.pt checkpoints and return {epoch: path}."""
    checkpoints = {}
    for pt_file in run_dir.glob("epoch_*.pt"):
        try:
            ep = int(pt_file.stem.replace("epoch_", ""))
            checkpoints[ep] = pt_file
        except ValueError:
            pass
    return checkpoints


def get_evaluated_epochs(run_dir: Path, eval_prefix: str = "manual_eval") -> set:
    """Check which epochs already have completed eval results."""
    evals_dir = run_dir / "evals"
    if not evals_dir.exists():
        return set()
    evaluated = set()
    for d in evals_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{eval_prefix}_ep"):
            try:
                ep = int(d.name.replace(f"{eval_prefix}_ep", ""))
                metrics_file = d / "fullval_metrics.json"
                if metrics_file.exists():
                    evaluated.add(ep)
            except ValueError:
                pass
    return evaluated


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the entire process group (handles sudo unshare child processes)."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"    WARNING: Failed to kill process tree: {e}")
        try:
            proc.kill()
        except Exception:
            pass


def run_evaluation(run_info: dict, epoch: int, checkpoint: Path, full: bool = False) -> bool:
    """Run evaluation for a specific epoch checkpoint.

    Args:
        run_info: Run configuration dict
        epoch: Epoch number (0-indexed)
        checkpoint: Path to the epoch checkpoint file
        full: If True, run full eval (all samples). If False, use MONITOR_MAX_SAMPLES.

    Returns True if successful.
    """
    name = run_info["name"]
    cuda_idx = run_info["eval_cuda"]

    if not checkpoint.exists():
        print(f"  [{name}] Checkpoint {checkpoint} does not exist, skipping")
        return False

    eval_tag = "full_eval" if full else "manual_eval"
    eval_dir = run_info["run_dir"] / "evals" / f"{eval_tag}_ep{epoch}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_info["run_dir"] / "evals" / f"{eval_tag}_ep{epoch}.log"

    mode_str = "FULL" if full else f"MONITOR ({MONITOR_MAX_SAMPLES} samples)"
    timeout = FULL_EVAL_TIMEOUT if full else MONITOR_TIMEOUT

    max_samples_arg = "" if full else f"--max_samples {MONITOR_MAX_SAMPLES} "

    print(f"  [{name}] {mode_str} eval ep{epoch} -> {checkpoint.name} on CUDA={cuda_idx} (timeout={timeout}s)...")
    sys.stdout.flush()

    cmd = (
        f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        f"{PYTHON} {EVAL_SCRIPT} "
        f"--checkpoint {checkpoint} "
        f"--data_dir {run_info['data_dir']} "
        f"--label_dir {run_info['label_dir']} "
        f"--split {run_info['split']} "
        f"--outdir {eval_dir} "
        f"--batch_size {run_info['batch_size']} "
        f"--num_workers 4 "
        f"{max_samples_arg}"
    )

    proc = None
    try:
        with open(log_file, "w") as lf:
            lf.write(f"# {mode_str} eval for {name} ep{epoch}\n")
            lf.write(f"# Checkpoint: {checkpoint}\n")
            lf.write(f"# Started: {datetime.now().isoformat()}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = proc.wait(timeout=timeout)

        if returncode == 0:
            metrics_file = eval_dir / "fullval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                map50 = metrics.get("mAP@50", metrics.get("mAP_50", 0)) * 100
                map5095 = metrics.get("mAP@50:95", metrics.get("mAP_50_95", 0)) * 100
                prec = metrics.get("precision", 0)
                rec = metrics.get("recall", 0)
                n = metrics.get("n_samples", "?")
                print(f"  [{name}] Ep{epoch} ({mode_str}): "
                      f"mAP@50={map50:.2f}% mAP@50:95={map5095:.2f}% "
                      f"P={prec:.3f} R={rec:.3f} (n={n})")
                sys.stdout.flush()
                return True
            else:
                print(f"  [{name}] Ep{epoch}: eval completed but no metrics file")
                sys.stdout.flush()
                return False
        else:
            print(f"  [{name}] Ep{epoch}: eval FAILED (exit code {returncode})")
            sys.stdout.flush()
            return False

    except subprocess.TimeoutExpired:
        print(f"  [{name}] Ep{epoch}: eval TIMED OUT after {timeout}s, killing process tree...")
        sys.stdout.flush()
        if proc is not None:
            _kill_process_tree(proc)
        return False
    except Exception as e:
        print(f"  [{name}] Ep{epoch}: eval ERROR: {e}")
        sys.stdout.flush()
        if proc is not None:
            _kill_process_tree(proc)
        return False


def main():
    print("=" * 70)
    print("  Evaluation Watcher")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Monitoring {len(RUNS)} runs, polling every {POLL_INTERVAL}s")
    print(f"  Monitor evals: {MONITOR_MAX_SAMPLES} samples, timeout={MONITOR_TIMEOUT}s")
    print(f"  Full evals: all samples, timeout={FULL_EVAL_TIMEOUT}s (final epoch only)")
    print(f"  Eval GPUs: see RUNS configuration above")
    print(f"  Training saves every epoch checkpoint -> no snapshot timing issues")
    print("=" * 70)
    sys.stdout.flush()

    while True:
        found_new = False

        for run_info in RUNS:
            name = run_info["name"]
            run_dir = run_info["run_dir"]

            # Find available epoch checkpoints saved by training
            checkpoints = find_epoch_checkpoints(run_dir)
            if not checkpoints:
                continue

            # Find already-evaluated epochs
            evaluated = get_evaluated_epochs(run_dir, "manual_eval")

            # Run monitoring eval for each un-evaluated epoch
            for ep in sorted(checkpoints.keys()):
                if ep not in evaluated:
                    found_new = True
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{ts}] {name}: found epoch_{ep:03d}.pt (unevaluated)")
                    sys.stdout.flush()

                    success = run_evaluation(run_info, ep, checkpoints[ep], full=False)
                    if success:
                        evaluated.add(ep)
                    sys.stdout.flush()

        # Check if all runs finished (have epoch 19 checkpoint)
        all_done = True
        for run_info in RUNS:
            checkpoints = find_epoch_checkpoints(run_info["run_dir"])
            if FINAL_EPOCH not in checkpoints:
                all_done = False
                break

        if all_done:
            print("\n" + "=" * 70)
            print("  All 3 runs completed training! Running FULL evaluations...")
            print("=" * 70)
            sys.stdout.flush()

            for run_info in RUNS:
                name = run_info["name"]
                run_dir = run_info["run_dir"]
                checkpoints = find_epoch_checkpoints(run_dir)

                # Full eval on the final epoch
                full_evaluated = get_evaluated_epochs(run_dir, "full_eval")
                if FINAL_EPOCH not in full_evaluated:
                    run_evaluation(run_info, FINAL_EPOCH, checkpoints[FINAL_EPOCH], full=True)

                # Also full eval on best.pt
                best_pt = run_dir / "best.pt"
                if best_pt.exists():
                    best_eval_dir = run_dir / "evals" / "full_eval_best"
                    if not (best_eval_dir / "fullval_metrics.json").exists():
                        print(f"\n  [{name}] Running FULL eval on best.pt...")
                        sys.stdout.flush()
                        best_eval_dir.mkdir(parents=True, exist_ok=True)
                        log_file = run_dir / "evals" / "full_eval_best.log"
                        cmd = (
                            f"CUDA_VISIBLE_DEVICES={run_info['eval_cuda']} "
                            f"{PYTHON} {EVAL_SCRIPT} "
                            f"--checkpoint {best_pt} "
                            f"--data_dir {run_info['data_dir']} "
                            f"--label_dir {run_info['label_dir']} "
                            f"--split {run_info['split']} "
                            f"--outdir {best_eval_dir} "
                            f"--batch_size {run_info['batch_size']} "
                            f"--num_workers 4"
                        )
                        with open(log_file, "w") as lf:
                            proc = subprocess.Popen(
                                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            try:
                                proc.wait(timeout=FULL_EVAL_TIMEOUT)
                            except subprocess.TimeoutExpired:
                                _kill_process_tree(proc)
                                print(f"  [{name}] best.pt eval TIMED OUT")
                                sys.stdout.flush()

            # Final summary
            print("\n" + "=" * 70)
            print("  FINAL RESULTS SUMMARY")
            print("=" * 70)
            for run_info in RUNS:
                name = run_info["name"]
                evals_dir = run_info["run_dir"] / "evals"
                print(f"\n  {name}:")

                # Show monitoring evals
                for ep in range(TOTAL_EPOCHS):
                    metrics_file = evals_dir / f"manual_eval_ep{ep}" / "fullval_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            m = json.load(f)
                        map50 = m.get("mAP@50", m.get("mAP_50", 0)) * 100
                        n = m.get("n_samples", "?")
                        print(f"    Ep{ep:2d} (monitor, n={n}): mAP@50={map50:.2f}%")

                # Show full evals
                for tag in [f"full_eval_ep{FINAL_EPOCH}", "full_eval_best"]:
                    metrics_file = evals_dir / tag / "fullval_metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            m = json.load(f)
                        map50 = m.get("mAP@50", m.get("mAP_50", 0)) * 100
                        map5095 = m.get("mAP@50:95", m.get("mAP_50_95", 0)) * 100
                        prec = m.get("precision", 0)
                        rec = m.get("recall", 0)
                        n = m.get("n_samples", "?")
                        print(f"    {tag} (n={n}): mAP@50={map50:.2f}%  mAP@50:95={map5095:.2f}%  P={prec:.3f}  R={rec:.3f}")

            print("\n" + "=" * 70)
            print("  Done! All evaluations complete.")
            print("=" * 70)
            sys.stdout.flush()
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
