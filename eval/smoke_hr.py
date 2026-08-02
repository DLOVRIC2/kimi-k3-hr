#!/usr/bin/env python3
"""Croatian-focused smoke test — the thing this build exists for.

The toolchain's own smoke.py hardcodes Python / French-capital / Chinese, so it
never touches Croatian. This runs the same shape of test against prompts that
actually exercise what the hr+code calibration was meant to preserve.

Chinese is included deliberately as a NEGATIVE control. The calibration corpus
contained no Chinese, so Chinese experts scored low and were pruned. If Chinese
loops while Croatian stays fluent, that is the targeting working — not a defect.

    uv run eval/smoke_hr.py --path <build> --src <Kimi-K3-src>
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

STOP_TOKENS = {163584, 163585, 163586, 163588}

PROMPTS = [
    ("hr-prose", "croatian",
     "Objasni ukratko razliku između svršenih i nesvršenih glagola u hrvatskom jeziku."),
    ("hr-business", "croatian",
     "Napiši kratku obavijest zaposlenicima trgovine o promjeni radnog vremena "
     "tijekom blagdana. Najviše 80 riječi."),
    ("hr-code", "croatian+code",
     "Napiši Python funkciju koja provjerava je li OIB (hrvatski osobni "
     "identifikacijski broj) ispravan, uključujući kontrolnu znamenku."),
    ("code", "control",
     "Write a Python function `merge_intervals(intervals)` that merges "
     "overlapping closed intervals given as (start, end) tuples."),
    ("zh", "negative control — pruned away on purpose",
     "简要解释机器学习的基本原理。"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--toolchain", default=str(pathlib.Path(
        os.environ.get("K3_MLX", pathlib.Path.home() / "kimi-k3-mlx"))),
        help="checkout of PipeNetwork/kimi-k3-mlx; overrides $K3_MLX")
    ap.add_argument("--raw", dest="chat", action="store_false",
                    help="skip the chat template (base-model style completion)")
    ap.add_argument("--thinking", action="store_true",
                    help="open the reasoning channel (very slow at ~5 tok/s)")
    a = ap.parse_args()

    tc = pathlib.Path(a.toolchain)
    sys.path.insert(0, str(tc / "scripts"))
    sys.path.insert(0, str(tc))

    import mlx.core as mx
    import mlxmem
    from mlx_lm.utils import load_model
    from reap_calibrate import build_tokenizer

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import k3_chat

    # Before load_model, always. An unwired model faults weights from SSD on
    # every decode step and reports a fraction of its real speed.
    mlxmem.wire()

    tok_src = a.path if os.path.exists(os.path.join(a.path, "tiktoken.model")) else a.src
    enc = build_tokenizer(tok_src)

    t0 = time.time()
    print(f"[hr] loading {a.path} ...", flush=True)
    model, _ = load_model(pathlib.Path(a.path), lazy=False)
    mx.eval(model.parameters())
    peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
    print(f"[hr] loaded in {time.time()-t0:.0f}s | peak {peak:.0f} GB\n", flush=True)

    rates = []
    for pid, role, prompt in PROMPTS:
        # Template it. Raw text makes an instruction-tuned model continue the
        # prompt instead of answering it.
        ids = (k3_chat.encode(enc, a.src, k3_chat.user(prompt), thinking=a.thinking)
               if a.chat else enc.encode_ordinary(prompt))
        x = mx.array([ids])
        cache = model.make_cache()

        t0 = time.time()
        logits = model(x, cache=cache)
        mx.eval(logits)
        prefill_s = time.time() - t0

        print(f"{'='*70}\n[{pid}]  ({role})\n{'='*70}")
        print(f"PROMPT: {prompt.strip()[:120]}")
        print(f"prefill {len(ids)} tok in {prefill_s:.1f}s\n")

        out, tok = [], int(mx.argmax(logits[0, -1]))
        t0 = time.time()
        for _ in range(a.max_tokens):
            if tok in STOP_TOKENS:
                break
            out.append(tok)
            logits = model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1]))
        gen_s = time.time() - t0
        rate = len(out) / max(gen_s, 1e-9)
        rates.append(rate)

        print(enc.decode(out))
        print(f"\n--> {len(out)} tok in {gen_s:.1f}s = {rate:.2f} tok/s\n", flush=True)

    print(f"{'='*70}")
    print(f"mean decode: {sum(rates)/len(rates):.2f} tok/s")
    print("[hr] HR-SMOKE-DONE")


if __name__ == "__main__":
    main()
