#!/usr/bin/env python3
"""Run the benchmark suite against one backend and write results.

    # local K3 build (loads 350 GB, keep it resident across all tasks)
    uv run eval/run_eval.py --backend mlx:/path/to/build \\
        --src ~/…/Kimi-K3-src --limit-belebele 200

    # ollama comparators
    uv run eval/run_eval.py --backend ollama:gemma4:31b --limit-belebele 200

One backend per invocation, deliberately: a 350 GB MLX model and an ollama
model cannot both be resident on this machine, and interleaving them would make
timings meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from datetime import date

import backends as B
import monitor as M
import tasks as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="ollama:<model> or mlx:<path>")
    ap.add_argument("--src", default=None, help="Kimi-K3-src (mlx backend only)")
    ap.add_argument("--toolchain", default=str(pathlib.Path.home() / "kimi-k3-mlx"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--tasks", default="belebele_hrv,belebele_eng,humaneval")
    ap.add_argument("--limit-belebele", type=int, default=200)
    ap.add_argument("--limit-humaneval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--monitor-interval", type=float, default=5.0,
                    help="seconds between memory samples (0 disables)")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out_dir) if a.out_dir else pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = {}
    if a.backend.startswith("mlx:"):
        if not a.src:
            raise SystemExit("--src is required for the mlx backend")
        kw = dict(src=a.src, toolchain=a.toolchain, label=a.label, thinking=a.thinking)
    label = a.label or a.backend.split(":", 1)[1]

    # Start sampling BEFORE the backend is constructed — for the MLX backend
    # that call loads 350 GB, and the load itself is one of the most
    # interesting things to measure.
    mon = M.attach(out_dir, label, pid=os.getpid(), interval=a.monitor_interval)
    mon.label("load")
    try:
        backend = B.make_backend(a.backend, **kw)
    except Exception:
        mon.stop()
        raise
    label = a.label or backend.name
    mon.proc_pattern = getattr(backend, "proc_pattern", None)

    run = {
        "label": label,
        "backend": a.backend,
        "date": date.today().isoformat(),
        "seed": a.seed,
        "decoding": "greedy (temperature 0)",
        "tasks": {},
    }

    wanted = [t.strip() for t in a.tasks.split(",")]
    t_start = time.time()

    for task in wanted:
        print(f"\n[eval] {label} :: {task}", flush=True)
        mon.label(task)
        t0 = time.time()

        safe_label = label.replace(":", "-").replace("/", "-")
        ckpt = T.Checkpoint(out_dir / "checkpoints" / f"{safe_label}-{task}.jsonl")

        if task.startswith("belebele_"):
            lang = {"hrv": "hrv_Latn", "eng": "eng_Latn"}[task.split("_")[1]]
            ds = T.load_belebele(lang, a.limit_belebele, a.seed)
            res = T.score_belebele(backend, ds, ckpt=ckpt)
            print(f"    -> accuracy {res['accuracy']:.1%}  parse {res['parse_rate']:.1%}  n={res['n']}")
        elif task == "humaneval":
            ds = T.load_humaneval(a.limit_humaneval, a.seed)
            res = T.score_humaneval(backend, ds, ckpt=ckpt)
            print(f"    -> pass@1 {res['pass@1']:.1%}  n={res['n']}")
        else:
            raise SystemExit(f"unknown task {task!r}")

        res["wall_s"] = round(time.time() - t0, 1)
        run["tasks"][task] = res

    # The headline metric. Parallel items mean this isolates Croatian ability
    # from general capability -- a big model beating a small one on raw hrv
    # score says nothing on its own.
    hrv = run["tasks"].get("belebele_hrv", {}).get("accuracy")
    eng = run["tasks"].get("belebele_eng", {}).get("accuracy")
    if hrv is not None and eng is not None:
        run["croatian_gap"] = round(eng - hrv, 4)
        print(f"\n[eval] English-Croatian gap: {100*(eng-hrv):+.1f} pts "
              f"(eng {eng:.1%} / hrv {hrv:.1%})")

    mon.stop()
    run["resources"] = mon.summary()
    r = run["resources"]
    if r:
        print(f"\n[eval] peak wired {r['peak_wired_gb']:.0f} GB | "
              f"peak rss {r['peak_rss_gb']:.0f} GB | "
              f"peak compressed {r['peak_compressed_gb']:.0f} GB")
        sus = r.get("pagein_gb_per_min_sustained", 0.0)
        print(f"[eval] page-ins: {r['pagein_gb_load']:.1f} GB load, "
              f"{r['pagein_gb_inference']:.2f} GB early-inference (lazy mmap), "
              f"{sus:.3f} GB/min sustained "
              f"({'resident' if sus < 0.5 else 'THRASHING'})")

    run["total_wall_s"] = round(time.time() - t_start, 1)
    safe = label.replace(":", "-").replace("/", "-")
    path = out_dir / f"eval-{safe}-{run['date']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"\n[eval] wrote {path}  ({run['total_wall_s']/60:.1f} min)")

    backend.close()


if __name__ == "__main__":
    main()
