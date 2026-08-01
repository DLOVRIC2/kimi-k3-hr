"""One generation interface over MLX (local K3 builds) and Ollama.

Both backends must be driven IDENTICALLY or the comparison is worthless: same
prompts, same greedy decoding, same token budget, same stop handling. Anything
that differs between them becomes a confound indistinguishable from a real
capability difference.

Greedy everywhere (temperature 0). Sampling would make pass@1 a lottery and
Belebele accuracy noisy across reruns, and neither benchmark is measuring
diversity.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    thinking_chars: int = 0
    truncated: bool = False

    @property
    def decode_tok_s(self) -> float:
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0


class Backend:
    name: str

    # Tokens to allow for a task whose answer is a single digit.
    #
    # This is a per-backend property because it is not about the answer at all —
    # it is about what the model emits BEFORE the answer. A model with a hidden
    # reasoning channel spends the budget thinking first and returns empty
    # content if capped too low; a model without one starts writing the answer
    # immediately and burns any budget you give it on prose.
    #
    # Getting this wrong is expensive rather than incorrect: measured here, a
    # 512-token budget on a 5.2 tok/s local model meant ~98 seconds per item —
    # 5.5 hours for a 200-item task that should take under an hour.
    answer_budget: int = 512

    # Tokens for a task whose answer is a function body.
    #
    # Deliberately NOT equal across backends. Fairness here means every model
    # gets enough budget to FINISH — not that every model gets the same number.
    # A model that reasons before writing code needs room for both; capping it
    # to the non-reasoning model's budget measures truncation, not coding
    # ability. Measured: gpt-oss:120b scored 20% on HumanEval at 512 tokens
    # while scoring 100% on both Belebele languages — it was being cut off
    # mid-function, not failing to code.
    code_budget: int = 2048

    # Process name to attribute memory to, when the model runs out-of-process.
    proc_pattern: str | None = None

    def generate(self, prompt: str, max_tokens: int = 512) -> Generation:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OllamaBackend(Backend):
    """Ollama's /api/chat. Greedy, no context carried between calls.

    /api/chat, not /api/generate: these are instruction-tuned models and the
    chat endpoint applies their template. Same lesson as K3 -- raw prompting an
    instruct model measures the wrong thing.

    Reasoning is the subtle trap. Several of these emit a separate `thinking`
    channel that does NOT appear in `content`, and it consumes the token budget
    first. With a small num_predict the whole budget goes to thinking and
    `content` comes back EMPTY -- which scores as a wrong answer and looks like
    a model that cannot read.

    `think: false` is honoured by gemma4 but ignored by gpt-oss, which reasons
    regardless. That asymmetry cannot be removed, so it is measured instead:
    thinking_chars is recorded per call and reported, rather than pretending all
    models were compared under identical conditions.
    """

    # Reasoning channel consumes the budget before content appears — measured
    # at 125+ tokens on gpt-oss even with think=false.
    answer_budget = 512
    code_budget = 2048
    proc_pattern = "llama-server"

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 think: bool = False):
        self.name = model
        self.model = model
        self.host = host
        self.think = think

    def generate(self, prompt: str, max_tokens: int = 512) -> Generation:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "num_predict": max_tokens,
                "seed": 20260801,
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        wall = time.time() - t0

        msg = d.get("message", {}) or {}
        prefill_s = d.get("prompt_eval_duration", 0) / 1e9
        decode_s = d.get("eval_duration", 0) / 1e9 or wall
        return Generation(
            text=msg.get("content", "") or "",
            prompt_tokens=d.get("prompt_eval_count", 0),
            gen_tokens=d.get("eval_count", 0),
            prefill_s=prefill_s,
            decode_s=decode_s,
            thinking_chars=len(msg.get("thinking", "") or ""),
            truncated=d.get("done_reason") == "length",
        )


class MLXBackend(Backend):
    """A converted local K3 build, driven through its own chat template.

    Loads once and stays resident — a 350 GB reload per benchmark would dominate
    the run. wire() must precede load_model or decode collapses to a fraction of
    real speed (see mlxmem.py: 0.20 vs 5.42 tok/s).
    """

    def __init__(self, path: str, src: str, toolchain: str, label: str | None = None,
                 thinking: bool = False):
        # No hidden reasoning channel unless thinking is on, so the answer digit
        # is the first thing emitted. 24 tokens covers a digit plus any short
        # preamble; 512 would be spent writing an essay at 5 tok/s.
        self.answer_budget = 512 if thinking else 24
        self.code_budget = 2048 if thinking else 640
        self.proc_pattern = None  # runs in-process; track our own pid
        self.name = label or pathlib.Path(path).name
        self.src = src
        self.thinking = thinking

        tc = pathlib.Path(toolchain).expanduser()
        sys.path.insert(0, str(tc / "scripts"))
        sys.path.insert(0, str(tc))
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

        import mlx.core as mx
        import mlxmem
        from mlx_lm.utils import load_model
        from reap_calibrate import build_tokenizer
        import k3_chat

        self.mx = mx
        self.k3_chat = k3_chat

        mlxmem.wire()
        tok_src = path if (pathlib.Path(path) / "tiktoken.model").exists() else src
        self.enc = build_tokenizer(str(tok_src))

        t0 = time.time()
        print(f"[mlx] loading {path} ...", flush=True)
        self.model, _ = load_model(pathlib.Path(path).expanduser(), lazy=False)
        mx.eval(self.model.parameters())
        peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
        print(f"[mlx] loaded in {time.time()-t0:.0f}s | peak {peak:.0f} GB", flush=True)

    def generate(self, prompt: str, max_tokens: int = 512) -> Generation:
        mx = self.mx
        ids = self.k3_chat.encode(self.enc, self.src, self.k3_chat.user(prompt),
                                  thinking=self.thinking)
        x = mx.array([ids])
        cache = self.model.make_cache()

        t0 = time.time()
        logits = self.model(x, cache=cache)
        mx.eval(logits)
        prefill_s = time.time() - t0

        out: list[int] = []
        tok = int(mx.argmax(logits[0, -1]))
        t0 = time.time()
        for _ in range(max_tokens):
            if tok in self.k3_chat.STOP_TOKENS:
                break
            out.append(tok)
            logits = self.model(mx.array([[tok]]), cache=cache)
            mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1]))
        decode_s = time.time() - t0

        return Generation(
            text=self.enc.decode(out),
            prompt_tokens=len(ids),
            gen_tokens=len(out),
            prefill_s=prefill_s,
            decode_s=decode_s,
        )


def make_backend(spec: str, **kw) -> Backend:
    """`ollama:gemma4:31b` or `mlx:/path/to/build`."""
    if spec.startswith("ollama:"):
        return OllamaBackend(spec.split(":", 1)[1])
    if spec.startswith("mlx:"):
        return MLXBackend(spec.split(":", 1)[1], **kw)
    raise SystemExit(f"unknown backend spec: {spec!r}")
