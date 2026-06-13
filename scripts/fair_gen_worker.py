#!/usr/bin/env python3
"""FAIR generation worker: runs DramaBox inference for a batch of tasks.

Called by fair_eval_worker.py as a subprocess:
    python fair_gen_worker.py <tasks_json_path> <num_gpus>

Tasks JSON format: list of dicts with keys:
    prompt, prompt_id, ref, ref_name, output, lora
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRAMABOX_DIR = "/home/deployer/laion/DramaBox"
INFERENCE_SCRIPT = os.path.join(DRAMABOX_DIR, "src", "inference.py")
CHECKPOINT = os.path.join(DRAMABOX_DIR, "models", "ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
FULL_CHECKPOINT = os.path.join(DRAMABOX_DIR, "models", "ltx-2.3-22b-dev.safetensors")
GEMMA_ROOT = "/home/deployer/.cache/dramabox/models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots/826e729dbaeea4ecb143738eed2bcf3539ebf7bf"
PYTHON = "/home/deployer/miniconda3/envs/ml-general/bin/python"
LORA_RANK = 128


def run_one(task, gpu_id):
    """Run a single inference task on a specific GPU."""
    os.makedirs(os.path.dirname(task["output"]) or ".", exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("LD_LIBRARY_PATH", None)

    # Support full model checkpoints (no LoRA) via "checkpoint" key
    base_ckpt = task.get("checkpoint", CHECKPOINT)

    cmd = [
        PYTHON, INFERENCE_SCRIPT,
        "--prompt", task["prompt"],
        "--output", task["output"],
        "--checkpoint", base_ckpt,
        "--full-checkpoint", FULL_CHECKPOINT,
        "--gemma-root", GEMMA_ROOT,
        "--seed", "42",
        "--no-watermark",
    ]

    # Optional explicit duration
    if task.get("gen_duration"):
        cmd.extend(["--gen-duration", str(task["gen_duration"])])

    if task.get("lora"):
        cmd.extend(["--lora", task["lora"], "--lora-rank", str(LORA_RANK)])

    if task.get("ref"):
        cmd.extend(["--voice-sample", task["ref"]])
    else:
        cmd.append("--no-ref")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env=env, cwd=DRAMABOX_DIR,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0 and os.path.exists(task["output"])
        if not success:
            stderr_tail = result.stderr[-300:] if result.stderr else ""
            print(f"  FAIL gpu={gpu_id} {Path(task['output']).name}: {stderr_tail}", file=sys.stderr)
        else:
            print(f"  OK   gpu={gpu_id} {Path(task['output']).name} ({elapsed:.1f}s)")
        return success
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT gpu={gpu_id} {Path(task['output']).name}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ERROR gpu={gpu_id} {Path(task['output']).name}: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <tasks.json> <num_gpus>", file=sys.stderr)
        sys.exit(1)

    tasks_file = sys.argv[1]
    num_gpus = int(sys.argv[2])

    with open(tasks_file) as f:
        tasks = json.load(f)

    if not tasks:
        print("No tasks to generate.")
        return

    print(f"Generating {len(tasks)} samples on {num_gpus} GPUs...")

    # Resolve physical GPU IDs from parent's CUDA_VISIBLE_DEVICES
    # Parent sets e.g. CUDA_VISIBLE_DEVICES=6,7, so we use those physical IDs
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_visible:
        gpu_ids = [int(x) for x in parent_visible.split(",")][:num_gpus]
    else:
        gpu_ids = list(range(num_gpus))
    print(f"Using physical GPU IDs: {gpu_ids}")

    ok = 0
    fail = 0

    # Run tasks in parallel across GPUs
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            gpu = gpu_ids[i % len(gpu_ids)]
            fut = pool.submit(run_one, task, gpu)
            futures[fut] = task

        for fut in as_completed(futures):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"Done: {ok} ok, {fail} failed out of {len(tasks)} total")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
