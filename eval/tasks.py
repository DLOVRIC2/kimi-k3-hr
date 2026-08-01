"""Belebele and HumanEval, scored objectively.

Belebele is the load-bearing benchmark here because it is FULLY PARALLEL: the
same 900 items exist in every language. So `score(eng) - score(hrv)` isolates
Croatian ability from general capability, which raw scores cannot. A 601B model
beating a 31B model on Croatian proves nothing on its own; a 601B model with a
SMALLER English-Croatian gap than the 31B model is evidence the Croatian
targeting worked.

Unparseable answers are counted as wrong AND reported separately. Silently
dropping them would flatter a model that refuses hard questions, and the refusal
rate is itself a signal about pruning damage.
"""

from __future__ import annotations

import json
import multiprocessing
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass


class Checkpoint:
    """Append-only per-item log, so a long run survives being interrupted.

    Results were previously written once, at the end. A four-hour sweep that died
    at the three-hour mark therefore wrote nothing -- which happened twice, when a
    tool timeout took a backgrounded job with it. Each item is now flushed as it
    completes and replayed on restart.

    Keyed on the task's own item id (Belebele question_number, HumanEval task_id)
    rather than loop position, so resuming is correct even if the shuffle or the
    limit changes between attempts.
    """

    def __init__(self, path: pathlib.Path | str | None):
        self.path = pathlib.Path(path) if path else None
        self.done: dict[str, dict] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves one truncated final line.
                    # Dropping it is correct: that item simply gets redone.
                    continue
                self.done[str(rec["id"])] = rec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.done)

    def append(self, rec: dict) -> None:
        self.done[str(rec["id"])] = rec
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    def get(self, item_id) -> dict | None:
        return self.done.get(str(item_id))

BELEBELE_PROMPT = """Passage:
{passage}

Question: {question}

1. {a1}
2. {a2}
3. {a3}
4. {a4}

Answer with the number of the correct option only (1, 2, 3, or 4). Reply with a single digit and nothing else."""


@dataclass
class ItemResult:
    id: str
    correct: bool
    parsed: bool
    raw: str
    gen_tokens: int = 0
    decode_tok_s: float = 0.0
    thinking_chars: int = 0
    truncated: bool = False
    echoed: bool = False


def load_belebele(config: str, limit: int | None, seed: int):
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", config, split="test")
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


_DIGIT = re.compile(r"[1-4]")


def _is_echo(text: str, prompt: str) -> bool:
    """Did the model repeat the prompt back instead of answering?

    A distinct failure mode from a wrong answer: the model never engaged with
    the question at all. Scoring it as merely "incorrect" hides that, so it is
    counted separately. Observed at ~10% on K3 and gpt-oss:20b.
    """
    head = text.strip()[:60]
    return bool(head) and (head.startswith("Passage:") or head[:40] in prompt[:200])


def score_belebele(backend, ds, log=print, ckpt: Checkpoint | None = None) -> dict:
    # `ckpt or Checkpoint(None)` is WRONG here: Checkpoint defines __len__, so an
    # empty one is falsy and would be silently replaced by a no-op -- disabling
    # checkpointing on precisely the first run, the one it exists to protect.
    if ckpt is None:
        ckpt = Checkpoint(None)
    results: list[ItemResult] = []
    if len(ckpt):
        log(f"    resuming: {len(ckpt)} items already done")
    for i, row in enumerate(ds):
        prior = ckpt.get(row["question_number"])
        if prior is not None:
            results.append(ItemResult(**prior))
            continue
        prompt = BELEBELE_PROMPT.format(
            passage=row["flores_passage"], question=row["question"],
            a1=row["mc_answer1"], a2=row["mc_answer2"],
            a3=row["mc_answer3"], a4=row["mc_answer4"],
        )
        # The backend declares its own budget — see Backend.answer_budget.
        # A fixed value is wrong in both directions: too low truncates a
        # reasoning model's thinking and returns empty content; too high makes
        # a non-reasoning model write prose for 98 seconds per item.
        g = backend.generate(prompt, max_tokens=getattr(backend, "answer_budget", 512))
        echoed = _is_echo(g.text, prompt)
        # An echo never contains a real answer; any digit in it is incidental
        # text from the restated passage, so don't let it score by luck.
        m = None if echoed else _DIGIT.search(g.text)
        parsed = m is not None
        correct = parsed and int(m.group()) == int(row["correct_answer_num"])
        r = ItemResult(
            id=str(row["question_number"]), correct=correct, parsed=parsed,
            raw=g.text.strip()[:40], gen_tokens=g.gen_tokens,
            decode_tok_s=g.decode_tok_s, thinking_chars=g.thinking_chars,
            truncated=g.truncated, echoed=echoed,
        )
        results.append(r)
        ckpt.append(r.__dict__)
        if (i + 1) % 25 == 0:
            acc = sum(r.correct for r in results) / len(results)
            log(f"    {i+1}/{len(ds)}  acc {acc:.1%}")

    n = len(results)
    return {
        "n": n,
        "accuracy": sum(r.correct for r in results) / n if n else 0.0,
        "parse_rate": sum(r.parsed for r in results) / n if n else 0.0,
        "mean_tok_s": sum(r.decode_tok_s for r in results) / n if n else 0.0,
        "mean_thinking_chars": sum(r.thinking_chars for r in results) / n if n else 0.0,
        "truncated_rate": sum(r.truncated for r in results) / n if n else 0.0,
        "echo_rate": sum(r.echoed for r in results) / n if n else 0.0,
        "items": [r.__dict__ for r in results],
    }


# ---------------------------------------------------------------- HumanEval

def load_humaneval(limit: int | None, seed: int):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


HUMANEVAL_PROMPT = """Complete this Python function. Reply with the complete function only, inside a single ```python code block. No explanation.

```python
{prompt}```"""

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    """Pull runnable code out of a chat response.

    Models answer HumanEval in three shapes: a fenced block containing the whole
    function, a fenced block containing only the body, or a bare continuation of
    the signature. Handle all three, otherwise the score measures formatting
    compliance rather than coding ability.
    """
    m = _FENCE.search(text)
    body = m.group(1) if m else text

    if f"def {entry_point}" in body:
        return body
    # Body-only: re-attach the original signature.
    if body.strip() and not body.lstrip().startswith("def "):
        indented = textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
        return prompt + "\n" + indented
    return prompt + body


_RUNNER = """
import sys, json
{code}

{test}

try:
    check({entry_point})
    print("__PASS__")
except BaseException as e:
    print("__FAIL__", type(e).__name__)
"""


def _run_one(code: str, test: str, entry_point: str, timeout: int = 15) -> tuple[bool, str]:
    """Execute generated code in a separate short-lived process.

    Model-generated code is untrusted: it runs in its own interpreter with a hard
    timeout so an infinite loop or a crash takes down only that subprocess. Not a
    security sandbox — do not point this at adversarial input.
    """
    src = _RUNNER.format(code=code, test=test, entry_point=entry_point)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        out = p.stdout
        if "__PASS__" in out:
            return True, "pass"
        reason = out.split("__FAIL__")[-1].strip() if "__FAIL__" in out else (
            p.stderr.strip().splitlines()[-1][:80] if p.stderr.strip() else "no output")
        return False, reason
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"harness:{type(e).__name__}"


def score_humaneval(backend, ds, log=print, ckpt: Checkpoint | None = None) -> dict:
    # `ckpt or Checkpoint(None)` is WRONG here: Checkpoint defines __len__, so an
    # empty one is falsy and would be silently replaced by a no-op -- disabling
    # checkpointing on precisely the first run, the one it exists to protect.
    if ckpt is None:
        ckpt = Checkpoint(None)
    results = []
    if len(ckpt):
        log(f"    resuming: {len(ckpt)} items already done")
    for i, row in enumerate(ds):
        prior = ckpt.get(row["task_id"])
        if prior is not None:
            results.append(prior)
            continue
        g = backend.generate(HUMANEVAL_PROMPT.format(prompt=row["prompt"]),
                             max_tokens=getattr(backend, "code_budget", 2048))
        code = extract_code(g.text, row["prompt"], row["entry_point"])
        ok, reason = _run_one(code, row["test"], row["entry_point"])
        rec = {
            "id": row["task_id"], "correct": ok,
            # A truncated generation is a budget failure, not a coding failure —
            # label it so it can't be silently read as "the model can't code".
            "reason": "truncated" if (g.truncated and not ok) else reason,
            "gen_tokens": g.gen_tokens, "decode_tok_s": g.decode_tok_s,
            "truncated": g.truncated,
        }
        results.append(rec)
        ckpt.append(rec)
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{len(ds)}  pass@1 {sum(r['correct'] for r in results)/len(results):.1%}")

    n = len(results)
    return {
        "n": n,
        "pass@1": sum(r["correct"] for r in results) / n if n else 0.0,
        "truncated_rate": sum(r.get("truncated", False) for r in results) / n if n else 0.0,
        "mean_tok_s": sum(r["decode_tok_s"] for r in results) / n if n else 0.0,
        "items": results,
    }
