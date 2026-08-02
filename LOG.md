# Lab log

Chronological record of building a Croatian-targeted expert prune of Kimi K3 on one
Mac Studio. Written for the writeup, so it keeps the mistakes rather than the tidy
version — the failures are where the transferable content is. Every number here was
measured on this machine unless marked otherwise.

Format per entry: **symptom → cause → fix → lesson**. The symptom is what it looked
like at the time, which is usually nothing like the cause.

---

## Machine constants

| | |
|---|---|
| Machine | Mac Studio, M3 Ultra, 512 GB unified memory |
| Memory bandwidth, spec | 819 GB/s |
| Memory bandwidth, **measured** | **651 GB/s** (`tools/bwtest.py`) — 79% of spec |
| Decode efficiency η, measured | ≈0.50 |
| Decode model | `tok/s = η × bandwidth / (active_params × bpw / 8)` |

The measured-vs-spec gap matters: predictions built on the sticker number are ~25%
optimistic before anything else goes wrong.

## The target

| | |
|---|---|
| Kimi K3 | 2.8 T total / 104 B active |
| Architecture | 92 MoE layers × 896 experts, top-16 routing, 2 shared experts |
| Smallest published quant | 594 GB |
| Our RAM | 512 GB |

**K3 does not fit at any quantisation.** That single fact is what forces the whole
project: the only remaining lever is deleting experts.

### Quantisation vs pruning (the distinction the project rests on)

- **Quantisation** reduces bits per parameter. Every weight survives at lower
  precision — uniform degradation, like JPEG compression.
- **Pruning** deletes whole experts. Some capabilities survive intact, others vanish
  entirely — selective amputation, like cropping.

And the consequence people get backwards:

- **Total** params decide whether a model *fits*.
- **Active** params decide how *fast* it runs.

Pruning shrinks total only. Our 601 B build has the same 104 B active as the 2.8 T
original, so **pruning buys fit, not speed.** A prune that halves the file size does
not halve the latency.

---

## Phase 1 — Corpus design

The premise: REAP scores every expert against a calibration corpus and deletes the low
scorers, so **the calibration corpus decides what the model can still do.** The
published builds used ~40% code, ~30% English, ~15% Chinese, ~15% across
ja/ru/ko/de/fr/es/ar. Croatian appears nowhere.

### 1. Claimed a tool was a cheap pre-check without reading it

**Symptom:** proposed running `reap_overlap.py` first, as a fast way to answer the
question before committing to anything expensive.
**Cause:** I inferred its behaviour from its name. It consumes a *tagged calibration
run* — the expensive thing it was supposed to let us avoid.
**Fix:** read the source before proposing it.
**Lesson:** in a pipeline where each stage costs hours, guessing at a tool's inputs
costs a whole stage. This recurred often enough to be the theme of the project.

### 2. Three corpora would have destroyed the comparison

**Symptom:** initial design had three recipes → three calibration runs → three builds.
Felt clean.
**Cause:** three calibration runs differ from each other by run-to-run variance as well
as by corpus. Those two effects are **not separable after the fact**, so the central
comparison would have been uninterpretable.
**Fix:** one tagged corpus (`corpus/recipes/survey.yaml`, 8 sources), one calibration
run, and `reap_subset.py` to derive each arm by summing whichever tag buckets it needs.
**Lesson:** from `reap_subset.py`'s own docs — subsetting one tagged run is *"strictly
better than re-calibrating on a filtered corpus — same tokens, same layer states, same
conditions."* Calibrate once, subset many. Worth reading a tool's docstring for the
experimental design advice, not just the flags.

### 3. Wrong manifest format — would have silently produced one bucket

**Symptom:** corpus builder wrote `<corpus>.tags.jsonl`.
**Cause:** `reap_calibrate.py` reads `<corpus>.txt.sources.json`, shaped
`{"order": [...], "chars": [...]}`. Nothing would have errored.
**Fix:** matched the real format, and added a hard pre-flight assertion:

```python
if pos != len(text):
    raise SystemExit("MANIFEST DESYNC: ... Refusing to write.")
```

**Lesson:** a wrong manifest doesn't crash — it yields **single-bucket saliency**,
which makes every per-language analysis in this repo impossible, discovered only after
a multi-hour run. Any contract that can fail silently before a long job gets an
explicit check that refuses to start.

### 4. The interleaver emitted runs — first quarter was 100% Croatian

**Symptom:** corpus composition looked correct. Overall ratios were exactly on target.
**Cause:** weighted round-robin emitted each source in contiguous runs. **Calibration
only reads the first `seqs × seqlen` tokens** — so the corpus that would actually be
scored was pure Croatian, and every other source was in the unread tail.
**Fix:** largest-deficit scheduling, so *every prefix* approximates the target mix:

```python
for n in range(total_chunks):
    for s in sources:
        deficit = s.ratio * (n + 1) - cursor[s.tag]   # pick the largest
```

Plus the builder now reports achieved composition **of the first quarter**, not just of
the whole file.
**Lesson:** this is the exact failure the project studies — a corpus that omits a
language — reintroduced by accident, in the tooling built to study it. When a consumer
reads a prefix, validate the prefix.

### 5. Dataset that can't load

**Symptom:** `codeparrot/github-code-clean` failed under `datasets>=3`.
**Cause:** script-based datasets are no longer supported.
**Fix:** `code-search-net/code_search_net`, config `python`, field `whole_func_string`.

### Gotcha: pooled multilingual datasets lie

C4's pooled multilingual config measured **97% Latin script over 200 sampled
documents**. Every recipe here uses per-language *named* configs instead. Sample your
corpus before trusting its label.

---

## Phase 2 — Memory wiring

### 6. Told the user a reboot was needed — twice, and wrong both times

**Symptom:** asserted `sudo sysctl iogpu.wired_limit_mb` needs a reboot to take effect.
**Cause:** guessed.
**Fix:** it applies **immediately**. And a reboot *clears* it — the opposite of what I
said. It must be re-applied after every boot.
**Lesson:** the user asked twice because the answer sounded wrong. It was.

### Gotcha: `mlxmem.wire()` must precede `load_model`

Not wiring the model first collapses decode **27×** — 0.20 vs 5.42 tok/s. The model
still loads and still answers, just at a speed you'd misread as "the prune broke it".
An ordering bug that presents as a capability result.

---

## Phase 3 — First measurements

Calibration ran over 262k tokens across 8 tagged sources. `top-242 of 896` experts
retained per layer (chance overlap baseline: **27.0%**).

### Expert overlap between languages

```
                  code     de     en     fr     hr     ru     sr     zh
  hr             29.2%  62.3%  57.0%  61.6%    —   73.9%  68.0%  47.9%
  code             —   34.1%  28.8%  34.3%  29.2%  28.2%  37.4%  23.8%

  off-diagonal: min 23.8%  mean 50.4%  max 73.9%   (1.87x chance)
```

Two results worth publishing on their own:

- **hr↔ru = 73.9%** is the strongest pair in the entire matrix. Croatian's experts are
  substantially Russian's experts — and Russian *is* in the reference mix. So Croatian
  partially survives on a neighbour's coattails rather than dying outright. Nobody had
  measured this.
- **code↔zh = 23.8% is *below* the 27% chance baseline.** Code and Chinese actively
  avoid each other's experts. Code is the most distinct thing in the corpus — it
  overlaps *everything* less than languages overlap each other.

### 7. Summing raw over-weighted Russian and flattered the answer

**Symptom:** the counterfactual "what would a reference-mix plan retain for Croatian?"
initially looked benign.
**Cause:** our survey corpus is **12.4% Russian**; the reference mix is **~2.1%**. Since
Russian is Croatian's nearest neighbour at 73.9% overlap, summing our raw buckets let
Russian carry Croatian and overstate its survival.
**Fix:** rescale each source to its reference share before summing
(`analysis/reference_mix_retention.py`).
**Lesson:** when simulating a counterfactual corpus out of a different corpus, reweight
to the target distribution. The nearest neighbour is exactly the source that will hide
the effect.

### Retention result

| | Croatian |
|---|---|
| chance | 27.0% |
| under a reference-mix plan | **54.2%** |
| own-distribution ceiling | 82.5% |
| gap | **+28.3 pts** |
| mean gap for sources *in* the reference mix | +18.4 pts |

**Croatian is partially degraded, not dead.** It retains real mass but measurably less
than languages that are in the reference mix. Retention tracks corpus share on a
notably flat curve — 54% → 66% across 0% → 43.8% share — so Croatian at 0% lands about
where German does at 2.3%. A targeted build helps, but the headroom is ~28 points, not
the total-collapse story we expected going in.

---

## Phase 4 — Building it

### 8. `--profile mxfp4` left non-experts at bf16 — 4.5× slower than it should be

**Symptom:** first build was **402 GB** and decoded at **1.14 tok/s**. Nearly 5× slower
than the published REAP80's 5.54 tok/s at a similar size. Looked like our prune had
damaged the model.
**Cause:** `--profile mxfp4` quantises *experts* and leaves everything else at bf16 —
114 GB of attention and shared-expert weights, re-read from memory **on every single
token**. Purely a bandwidth problem; nothing was wrong with the prune.
**Fix:** rebuild with `--nonexpert-bits 8`.

| | before | after |
|---|---|---|
| size | 402 GB | **350 GB** |
| decode | 1.14 tok/s | **5.19 tok/s** |

**Lesson:** the single highest-leverage flag in the project, and it was a default. In a
bandwidth-bound regime the *non*-expert weights dominate, because unlike experts they
are read every token regardless of routing. Check what a "profile" leaves untouched.

### 9. Bash-tool timeout killed a backgrounded job at layer 86/92

**Symptom:** a `nohup`'d conversion died silently after ~90 minutes.
**Cause:** the tool call that launched it timed out, taking the child with it.
**Fix:** the launching command must return *immediately* — `nohup … & disown`, then poll
separately. Happened twice before it stuck.

---

## Phase 5 — The chat template

### 10. Five ordinary tokens that looked like a broken model

**Symptom:** the finished build echoed prompts back and degenerated into repetition.
Textbook over-pruned model. I was ready to call the prune a failure.
**Cause:** the toolchain's `build_tokenizer` registers K3's high vocab range as generic
`<|reserved_special_token_N|>` placeholders. tiktoken has therefore **never heard of
`<|open|>`** and cheerfully encoded it as five *ordinary* text tokens. The model was
being shown its own chat markup as literal prose, so it did the only sensible thing and
continued the text.
**Fix:** read the true mapping from `tokenizer_config.json`'s `added_tokens_decoder`,
and refuse rather than fall back:

```python
tid = specials.get(seg.text)
if tid is None:
    raise ValueError(f"control segment {seg.text!r} is not in added_tokens_decoder ...")
```

**Lesson:** **the best mistake in the project.** A tokenizer bug and a capability
collapse are indistinguishable from the output alone. Before concluding anything about
a model's ability, verify it is receiving its own chat format — a silent fallback to
ordinary encoding turns a control token into text and produces exactly the symptoms of
brain damage. Also: K3 is instruction-tuned, so raw-prompting it makes it behave like a
base model. That is not a capability failure, and mistaking one for the other is how
you "discover" that a build can't follow instructions.

### Selective damage, demonstrated

Once the template was right, on the *same build in the same run*:

- **Croatian** — fluent, grammatical prose
- **Python** — correct
- **Chinese** — **zero tokens**

The thesis, visible in one run. Also the cleanest evidence that pruning is selective
amputation rather than uniform degradation.

---

## Phase 6 — Building the eval harness

Moved from hand-scored 0–5 probes to objective scoring: **Belebele** (reading
comprehension) and **HumanEval** (code).

Belebele is load-bearing because it is **fully parallel** — the same 900 items exist in
every language — so `score(eng) − score(hrv)` isolates Croatian ability from general
capability. A big model beating a small one on raw Croatian proves nothing; a smaller
*English−Croatian gap* is real evidence.

### 11. Left a 4-hour eval running and forgot about it

**Symptom:** launched a run, moved on to other work, never checked back.
**Cause:** no discipline around long jobs.
**Fix:** killed and discarded it. This is also where the user's standing rule came from:
**confirm before starting anything that occupies the machine for hours.**

### 12. A fixed 512-token budget made Belebele 5.5 hours per language

**Symptom:** K3 taking ~98 s for a question whose answer is one digit.
**Cause:** a single `max_tokens=512` for all backends. K3 has no hidden reasoning
channel with thinking off, so it starts answering immediately and then **writes prose
until it hits the cap.**
**Fix:** per-backend `answer_budget` — 24 for non-thinking MLX, 512 for models with a
reasoning channel.
**Lesson:** the budget isn't about the answer, it's about what the model emits *before*
the answer. One number cannot be right for both architectures.

### 13. The same fix, applied too aggressively, produced a fake result

**Symptom:** gpt-oss:120b scored **20% on HumanEval** while scoring 90% on *both*
Belebele languages. I nearly wrote that up as a finding.
**Cause:** the 512-token cap truncated it mid-function. It was being cut off, not
failing to code.
**Fix:** separate `code_budget` — 2048 for reasoning models. Result: **20% → 100%.**
**Lesson:** **the benchmark was measuring my configuration, not the model.** Fairness
means every model gets enough budget to *finish*, not that every model gets the same
number. And the tell was already visible: a model acing reading comprehension and
failing at code is a harness bug, not a capability profile. Truncation is now labelled
in the results so it can never be silently read as inability.

### 14. Reported 0 GB for a 20 GB model

**Symptom:** memory monitor showed ollama models using nothing.
**Cause:** ollama runs the model in a **separate `llama-server` process**; we were
sampling our own PID.
**Fix:** `proc_pattern` — attribute RSS to the largest matching process.

### 15. Called a healthy run "thrashing"

**Symptom:** monitor reported "faulted 80.8 GB from disk" on a run that was completely
fine.
**Cause:** two conflations. First, **loading a 350 GB model necessarily reads 350 GB
from SSD** — that's the load, not a fault. Second, llama.cpp **mmaps the GGUF and pages
in lazily on first touch**, so ollama reports a model as loaded before its pages are
resident and the reads land in the first ~30 s of "inference".
**Fix:** split page-ins by phase, then replace the total with a **sustained tail rate**:

```python
def _sustained_gb_per_min(rows):
    tail = rows[len(rows)//2:]          # real thrashing is SUSTAINED
```

Measured on gpt-oss:120b: 15.23 GB paged in during the first 31 s, then **0.015 GB/min**
for the rest. Healthy.
**Lesson:** cumulative counters answer "how much", never "is this pathological". Rate
over a tail window does.

### Gotcha: `free` is not a health signal on macOS

Free RAM sits near zero on a healthy Mac, because unused RAM is wasted RAM. Read
**compressed** and **swap** instead. The monitor reports `free` but deliberately doesn't
treat it as a warning.

### Gotcha: echoing is a distinct failure mode from a wrong answer

Some models repeat the prompt instead of answering. Scoring that as merely "incorrect"
hides it, and any digit inside the echoed passage can score **correct by luck**. Now
detected, excluded from parsing, and reported as its own rate. Observed at ~10% on K3
and gpt-oss:20b.

### 16. The checkpoint disabled itself on the only run that mattered

**Symptom:** added per-item checkpointing so an unattended sweep could survive a crash.
Unit tests passed. End-to-end, **no checkpoint file was ever written** — and the second
run silently redid all the work instead of resuming.
**Cause:** `Checkpoint` defines `__len__`, which makes an **empty instance falsy**. The
defensive idiom `ckpt = ckpt or Checkpoint(None)` therefore threw away the real
checkpoint and substituted a no-op — on exactly the first run, the one it exists to
protect. It only ever worked once a checkpoint was already non-empty, which is the case
that needs it least.
**Fix:** `if ckpt is None:` — identity, not truthiness.
**Lesson:** `x or default` is not a null check on any object that defines `__len__` or
`__bool__`. Empty collections, zero-length checkpoints, `0`, and `""` all take the
default branch. The unit test missed it because it tested `Checkpoint` in isolation; the
bug lived in the *caller*.

### 17. The repo was never self-contained — the eval ran from another venv

**Symptom:** launched the full sweep. Both K3 arms died in under a second with
`ModuleNotFoundError: No module named 'mlx'`.
**Cause:** `mlx`, `mlx-lm` and `tiktoken` were never in this project's dependencies. The
pilot had been run from `local-models/.venv`, which happened to have them. The repo
looked self-contained and wasn't.
**Fix:** pinned `mlx==0.32.0`, `mlx-lm==0.31.3`, `tiktoken==0.13.0` — the versions the
build was actually validated against. Plus a second, non-obvious step: the toolchain's
`scripts/install_model.sh` **copies `kimi_k3.py` into the target interpreter's
`mlx_lm/models/`**, because upstream mlx-lm has no K3 model class. `uv sync` alone
cannot produce a working environment.
**Lesson:** a project is only reproducible from the machine that has never run it. The
failure surfaced at launch time, which is the good case — it could as easily have been
discovered after the ollama arms had run and the K3 comparison was due.

### 18. The checkpoint key collapsed 200 items into 2

**Symptom:** first real 200-item arm finished suspiciously fast and reported
`accuracy 53.0% parse 53.0% n=200` — for a model that scored 70% with 100% parse in the
pilot. Both languages returned **accuracy exactly equal to parse rate**.
**Cause:** the checkpoint keyed on `row["question_number"]`. That is the question's index
*within its passage*, so across Belebele's 900-item split **it takes exactly two values**:

```
rows: 900 | distinct question_number: 2   →   most common: [(1, 482), (2, 418)]
```

Items 3..200 were therefore treated as already-done and **replayed the first two
results**. The benchmark made 2 real model calls and reported n=200. The aggregate was
just those two records duplicated 100×, which is exactly why accuracy and parse rate
were identical.
**Fix:** `(link, question_number)` — verified unique across all 900 rows — plus a
pre-flight that refuses to run if ids collide:

```python
if len(set(ids)) != len(ids):
    raise SystemExit(f"belebele ids are not unique ({len(set(ids))} of {len(ids)}); refusing")
```

**Lesson:** **the worst bug in the project.** It did not crash, did not warn, ran fast,
and produced a plausible-looking number that was fabricated. A field named `*_number` is
not an id. Anything used as a dedup key needs its uniqueness asserted against the actual
data, not assumed from its name — the same lesson as the manifest desync (#3), which is
why both now refuse rather than proceed.

### 19. The pilot's 100% parse rate was luck

**Symptom:** once #18 was fixed, gpt-oss:20b still showed ~47% of Belebele items
unparsed. Inspecting them: `raw=''`, `gen_tokens=512`, `truncated=True`,
`thinking_chars=2246`.
**Cause:** `answer_budget=512` for ollama. gpt-oss reasons regardless of `think: false`,
and on harder items the chain consumed the entire budget before any content was emitted
— so `content` came back **empty** and scored as wrong.
**Fix:** `answer_budget` 512 → 2048. Verified: **53% → 91.7%**, zero truncations, zero
empty responses.
**Lesson:** the n=10 pilot showed 100% parse purely because none of its ten items needed
a long chain. **Budget failures are heavy-tailed, so a small pilot is the wrong
instrument for finding them** — it validates that the harness runs, not that it is
correctly parameterised. This is the third distinct time a token budget has produced a
fake capability result (#12, #13, #19); the pattern is now the single most productive
thing to check when a score looks wrong.

**Also confirmed here:** the suspect peak-RSS from the pilot. With the other models
stopped, gpt-oss:20b measures **17 GB** rather than the 66 GB it reported when
gpt-oss:120b was still resident.

---

## Phase 7 — The Docker incident

**Symptom:** macOS "system has run out of application memory" popup mid-calibration.
Obvious suspect: the calibration.
**Cause:** calibration was innocent at **47.7 GB**. Docker's VM was holding **78.9 GB**
across ~100 containers (about 10 Supabase stacks), up 11–12 days.
**Fix:** quit Docker. Free RAM **0.1 → 130.6 GB**; compressed **114 → 39 GB**.
**Lesson:** on a machine used for both development and large-model work, the largest
memory consumer is often something that has been idle for a fortnight. Check the whole
machine before blaming the job you're watching.

---

## Pilot results (n=10 per task, 2026-08-01)

Ten items per task across four models, to validate the harness before committing hours.

| model | on disk | hrv | eng | gap | HumanEval | peak RSS | tok/s |
|---|---|---|---|---|---|---|---|
| **K3-REAP80-hr-code-q8** | 326 GB | 50% | 40% | −10 | 40% | 325 GB | 3.3 |
| gpt-oss:120b | 65 GB | 90% | 90% | +0 | 100% | 66 GB | 77.3 |
| gemma4:31b | 19 GB | 80% | 90% | +10 | 100% | 54 GB | 27.8 |
| gpt-oss:20b | 13 GB | 70% | 80% | +10 | 100% | 66 GB* | 107.0 |

\* **Suspect.** `_rss_gb_by_pattern` takes the *largest* `llama-server` process, so a
model left resident from a previous arm is attributed to the current one. 66 GB for a
13 GB model is almost certainly gpt-oss:120b still loaded. The full sweep issues
`ollama stop` between arms to remove this contamination. Peak-RSS figures from the
pilot should not be quoted for the ollama models until that rerun lands.

At n=10 the 95% CI is **±31 points**, so this is directional only. But two things are
already worth stating:

1. **The pilot paid for itself.** It caught three harness bugs (12, 13, 14) in twenty
   minutes. Two of them would have produced *publishable-looking but false* numbers.
2. **The 350 GB build loses to a 54 GB one on every axis measured.** If that holds at
   n=200, the interesting finding isn't "targeted pruning works" — it's that
   **REAP pruning appears to preserve fluency while destroying structured reasoning.**
   K3 wrote fluent Croatian prose in the smoke test and scored near the 25% chance
   floor on Croatian reading comprehension. A model that *sounds* completely intact and
   can't actually think is a better paper than the one we set out to write.

### Measured timings (from the pilot, per item)

| model | belebele | humaneval | full run (200/200/164) |
|---|---|---|---|
| K3-REAP80 | 6.9 s / 4.1 s | 35.1 s | **2.2 h** |
| gpt-oss:120b | 5.3 s / 2.6 s | 9.6 s | 53 min |
| gemma4:31b | 1.3 s / 1.1 s | 11.8 s | 40 min |
| gpt-oss:20b | 3.7 s / 2.2 s | 7.5 s | 40 min |

**K3 loads 350 GB in 45 seconds** (~7.8 GB/s off the SSD) — so keeping it resident
across tasks is not worth designing around. Full sweep ≈ **4.6 h serial**, of which
HumanEval is 2.9 h and 55% of that is K3 alone.

---

## Phase 8 — The sweep found four more before it could produce a number

Launching the full run surfaced #17–#19 above, and then three more that only appear at
scale or in thinking mode. Recorded here because the pattern is the point: **every
attempt to run the benchmark found a way the benchmark was lying.**

### 20. Thinking mode scored the chain of thought, not the answer

**Symptom:** K3 with `--thinking` scored **30% on Croatian** — chance is 25%. Half the
outputs read `<|reserved_special_token_3|>response...`, the rest were raw reasoning
("We need answer only 1/2/3/4. Need identi...").
**Cause:** with thinking on, K3 emits its chain of thought and *then* opens a second
channel, `<|open|>response<|sep|>`, before answering. Neither `OPEN` nor `SEP` is a stop
token, so the whole thing came back as one blob and the multiple-choice regex matched
whatever digit the model reasoned about **first** — uncorrelated with its conclusion.
The visible markup is the encode-side bug (#10) **mirrored on decode**: `enc.decode`
renders control ids as the generic placeholders the toolchain registered.
**Fix:** `split_channels()` — response is whatever follows the last separator, control
ids stripped.
**Lesson:** the same tokenizer defect produced two completely different-looking failures
a week apart, one on encode and one on decode. Once you find a control-token bug in one
direction, immediately check the other.

### 21. K3's answer budget was manufacturing its failures

**Symptom:** 11.7% echo rate and only 84% parsed on the non-thinking arm.
**Cause:** `answer_budget = 24` for MLX. **Median generation for a multiple-choice
answer is 1 token** — the model emits the digit and stops — so 24 looked generous. But
it binds on exactly the items where K3 opens with a preamble or restates the passage:
10 of 103 items hit the cap, and 12 of the 18 unparsed answers were cut off before
reaching a digit. Meanwhile every ollama model had 2048.
**Fix:** 24 → 256. Costs wall-clock only on that ~10%, since the median is unchanged.
**Lesson:** **the fourth budget-induced fake result** (#12, #13, #19, #21). A median
tells you nothing about whether a cap is safe — the tail is the whole question. Worse,
this one was self-inflicted: I fixed the identical bug for ollama in the same session
and left the local model capped, which quietly biased the comparison *against* the model
the project is about.

### 22. MLXBackend never reported truncation

**Symptom:** K3's `truncated_rate` was 0% in every result, including runs where items
demonstrably hit the cap.
**Cause:** `MLXBackend.generate` never set the field, so it took the dataclass default.
**Fix:** `truncated=len(out) >= max_tokens`.
**Lesson:** the field that would have exposed #21 immediately was hardcoded to "fine".
A health metric that cannot report bad news is worse than no metric — it actively
launders the failure.

### Resolved: the thinking-mode confound is not real

Final probe, n=20: **35.0%** with thinking on, against **40.5%** at n=200 with it off.
Reasoning mode does not rescue K3, so its weak scores are a property of the build rather
than of how we invoked it. The main comparison stands.

Worth recording honestly: the *broken* probe scored 30% and the fixed one scored 35%. The
channel-splitting fix (#20) was necessary for the number to mean anything, but it did not
change the conclusion — I would have reached the same answer with a harness that was
measuring the wrong text. Being right for the wrong reason is not a validation of the
method, and the only reason I know that is that I fixed it and re-ran.

K3's channel usage turned out to be erratic rather than absent: some items emit
`<|open|>response<|sep|>` and answer in 4 tokens, others emit **no channel markers at
all** and ramble for 400+ tokens before stopping naturally. One item produced a genuine
21,796-character chain. That inconsistency is itself a symptom — see Finding 1 in
RESULTS.md.

---

## Phase 9 — The sweep that finished

Ran unattended overnight, 2026-08-01 23:35 → 2026-08-02 03:44. All four arms exit 0.
2,276 scored items. Results and findings in `RESULTS.md`; this section records how the
run behaved and what was learned *during analysis*, which is the part that does not show
up in a results table.

### Arm timings (measured, for planning future runs)

| arm | wall | notes |
|---|---|---|
| k3-thinking-probe (n=20) | 16 min | 5.43 tok/s |
| k3-full | **133 min** | hrv 31m, eng 38m, HumanEval 64m |
| gpt-oss:20b full | 39 min | |
| gemma4:31b full | 32 min | |
| gpt-oss:120b full | 44 min | |
| **total** | **~4 h 9 min** | |

K3 loads 326 GB in **40–48 s** (~7 GB/s off SSD), so keeping a model resident across
arms is not worth designing around. Earlier estimates assumed a load cost that does not
exist.

### Unattended-run health checklist

Everything below was verified before leaving it overnight, and all of it held:

- **Sleep blocked** — `caffeinate -i` holds `PreventUserIdleSystemSleep`. Check with
  `pmset -g assertions`. Note `caffeinate -i` prevents *idle* sleep only.
- **Survives the session** — the launcher shows **ppid 1**, i.e. reparented to launchd,
  so closing the terminal or ending the agent session cannot kill it. Verify with
  `ps -o pid=,ppid=`. This is what `nohup … & disown` buys, and it is worth checking
  rather than assuming — a tool timeout killed backgrounded children twice (#9).
- **No leak** — over 745 samples / 52 min: RSS flat at 325 GB, wired flat at 469 GB,
  compressed flat at 3.4 GB, swap *decreasing*. Compressed memory is the signal; `free`
  is meaningless on macOS.
- **A failing arm does not stall the queue** — `sweep.sh` uses `set -u` but deliberately
  not `set -e`, so an arm that dies logs its exit code and the sweep continues.

### 23. The language comparison was never paired

**Symptom:** went to run McNemar on the Croatian-English gap — the right test, since
Belebele is fully parallel — and found only **48 of 200 items in common** between the
two languages.
**Cause:** `ds.shuffle(seed).select(range(200))` is applied to each language's dataset
*independently*. Same seed, but the source row order differs between the `hrv_Latn` and
`eng_Latn` configs, so the same seed selects different items.
**Fix:** not yet applied. Reported as limitation #1 in RESULTS.md and the gap is
evaluated with an unpaired two-proportion z-test instead (still p=0.00064).
**Lesson:** **the single reason to choose Belebele over any other benchmark is that it
is parallel** — the same 900 items in every language, which is what makes
`score(eng) − score(hrv)` isolate language ability from general capability. The harness
docstring says exactly this. And then the sampling threw that property away, silently,
because `shuffle(seed)` *looks* deterministic across configs. A design property has to
be enforced in code, not stated in a comment: the fix is to select item ids once and
intersect across languages.

### 24. Half of K3's HumanEval answers were empty, and I nearly reported it as a coding score

**Symptom:** K3 tracking 26% pass@1, with `gen_tokens` **median 0**.
**Cause:** genuine model behaviour, not a harness bug — on roughly half of prompts the
first token K3 predicts is a stop token, so it emits nothing at all. With an empty body
the harness reassembles the bare signature, the function returns `None`, and the test
fails with `AssertionError` (60 of them).
**Verification:** given four budget-related false results already (#12, #13, #19, #21),
I checked the pilot rather than trusting the interpretation — `gen_tokens` there was
`[640, 0, 0, 192, 0, 311, 287, 0, 83, 197]`, i.e. **4 of 10 already empty**. Pre-existing
and invisible at n=10, not a regression from the channel-splitting change.
**Lesson:** this is the finding, not a defect — see Finding 1. But the process matters
more than the result: the reflex to check a suspicious number against a prior run before
interpreting it is what separated "K3 cannot code" (wrong) from "K3 codes at 51% and is
silent half the time" (right). Every aggregate score in this project turned out to be
hiding a bimodal distribution.

---

## Phase 10 — Writing the analysis script found three more

Before starting the writeup I built `eval/analysis.py` to regenerate every published
table from the per-item checkpoints. The intent was reproducibility for a reader. What
it actually did first was audit me.

### 25. Two published numbers were wrong, because nothing could recompute them

**Symptom:** `RESULTS.md` claimed "50 of 99 sampled prompts produced zero tokens and
scored 0; the other 49 produced code and passed 53%." The completed task has 164 items,
not 99.
**Cause:** that sentence was computed in a shell against a checkpoint file **while the
run was still going**, then typed into markdown and never revisited. The true figures
are 83 zero-token and 81 passing 50.6%. The conditional rate in the table beside it
(50.6%) was correct, so the document disagreed with itself and nobody noticed.
**Fix:** the checkpoints are now the single source of truth and every table is emitted
by a script. Crucially `analysis.py` reads the *per-item* records, not the summary
JSONs — the summaries are themselves derived, so re-deriving from them would faithfully
preserve any error they contain.
**Lesson:** a number computed by hand in a terminal has no provenance and cannot be
re-checked. It is not that the arithmetic was hard; it is that a mid-run snapshot and a
completed run are different things, and prose cannot tell you which one you captured.
**If a figure in a writeup cannot be regenerated by a command, treat it as an
unverified claim** — including, especially, your own.

### 26. The throughput figure counted silent items as running at zero tokens per second

**Symptom:** K3 published at 4.3 tok/s, but every individual task measured ~5.4.
**Cause:** the summary averaged per-item `decode_tok_s`. An item that generates nothing
records a rate of 0.0, and K3 generated nothing on 83 of 164 coding prompts, so a third
of the "rate" samples were not rates at all. The same average also weighted a Belebele
item whose median generation is **one token** equally with a 200-token coding item —
and a one-token generation's "rate" is one forward pass over a very short interval,
which is mostly noise.
**Fix:** total tokens over total decode seconds, excluding items that decoded nothing.
K3 is a flat 5.4 tok/s across all three tasks once measured this way.
**Consequence:** the correction cuts both ways and is *not* flattering in aggregate.
K3 rises 4.3 → 5.4, but gemma4:31b falls 44.5 → 28.0 (its published figure was inflated
by those noisy single-token Belebele items). So "10× slower" becomes **5× slower**. The
conclusion holds comfortably — 13× the memory, 17× the disk, 5× slower, worse on every
task — but the number in the first draft was wrong in the direction that favoured the
story I was telling.
**Lesson:** an average over per-item rates is not a rate. Aggregate the numerator and
the denominator separately. And when a correction happens to strengthen your argument,
that is exactly when to check whether you have made the same class of error elsewhere.

### 27. My own significance test printed p = 0

**Symptom:** the first run of `analysis.py` reported `p = 0` for the K3-vs-gemma
comparisons.
**Cause:** computing the two-sided tail as `2 * (1 - normal_cdf(|z|))`. Past |z| ≈ 8 the
CDF rounds to exactly 1.0 in float64, the subtraction cancels to zero, and the result is
a hard 0 rather than a small number.
**Fix:** `math.erfc(|z| / sqrt(2))`, which computes the tail directly and stays accurate
well past z=30. The corrected values reproduce the originals exactly (6.06e-18, 3.3e-30,
2.94e-42), which also confirms the rewrite agrees with the earlier computation.
**Lesson:** `p = 0` is never a result, it is a rendering artifact — and one that reads as
*stronger* evidence than the truth. A reviewer seeing "p = 0" in a paper concludes the
author does not know what a p-value is. Catastrophic cancellation shows up wherever you
subtract a nearly-equal pair; the numerically stable form usually exists already.

### 28. The echo detector was throwing away correct answers

**Symptom:** while checking the captured example outputs, two items classified as
"echoed the passage" turned out to have generated a **single token**. One of them was
`'2'`, and the correct answer was 2.
**Cause:** `_is_echo` tested `head[:40] in prompt[:200]` with no minimum length. The
expected response to a Belebele item is one digit, and most passages contain that digit
somewhere in their first 200 characters. So a bare correct answer matched the
containment test, was labelled an echo, and — because the scorer deliberately refuses to
extract a digit from an echo, so that a restated passage cannot score by luck — was
discarded as unparsed and marked wrong.
**Scale:** 140 items across the paired sweep, 253 including the original English run.
For **all three comparators it was 100% of their reported echoes** — every single one a
bare correct answer. Those models never echoed a passage even once; the ~10% echo rate
attributed to them was entirely manufactured by the detector.
**Fix:** a 20-character floor before the containment test. An echo is a restatement and
cannot be shorter than a few words. `startswith("Passage:")` still fires at any length.
**Rescoring cost nothing.** The checkpoint stores `raw` as exactly `text.strip()[:40]`
and the rule reads `head[:40]`, so the stored prefix is precisely the rule's input: the
decision reproduces exactly rather than approximately, and 230 correct answers were
recovered from disk without loading a model. That was luck rather than design — had the
truncation been 30 characters, or the rule looked at 60, none of this would have been
recoverable and the whole sweep would have needed rerunning. **Store the exact input
your classifier consumes, not a convenient prefix of it.**
**Direction:** the correction raised the comparators 8–11 points and K3 only 3–6, so it
*widened* the gap the report is about. A bug whose fix makes your own result look worse
is the easy case; the hard case is #26, where the first error happened to favour the
story being told.
**Lesson:** a heuristic classifier that can veto a valid parse needs a bound on what it
is allowed to match. And the bug was invisible in every aggregate — echo rates of ~10%
looked entirely plausible for all four models. It only surfaced because something
unrelated (capturing example outputs for a blog post) put a single raw generation next
to its own classification. **Read your data at the level of individual records, not just
distributions.**

### Bugs found before the writeup vs during the run

Entries #25–#28 were all found *after* the sweep finished, by building tooling to
reproduce and illustrate numbers that already existed. Not one required rerunning a
model. Three were in the *reporting* layer rather than the measurement layer — the
experiment was fine, the transcription was not — and that layer had no tests, no
asserts and no second reader, which made it the least-scrutinised code in the project
and the only part whose output anyone would actually read.

#28 is the exception and the more troubling one: a genuine scoring bug, in code that had
been running since the pilot, invisible in every aggregate for the whole project. It
surfaced only because capturing example generations for a writeup happened to place one
raw response beside its own classification. **The work done to explain a result is not
separate from the work of getting it right.**

---

## State of the repo (2026-08-02)

**Complete and pushed.** `git status` clean, `origin/main` in sync.

| | |
|---|---|
| `LOG.md` | this file — 28 entries, the build narrative |
| `RESULTS.md` | the full sweep, findings, significance tests, limitations |
| `README.md` | thesis, arms, setup, licence |
| `results/full/` | the sweep as run — 13 per-item checkpoints (2,276 items), 5 memtraces, 5 result JSONs, arm logs |
| `results/paired/` | English re-scored on the Croatian item set (#23), 53 min |
| `results/rescored/` | corrected echo rule applied to both (#28), no GPU time |
| `results/examples/` | full generations for 14 stratified items, K3 and gemma4:31b on identical prompts |
| `results/pilot/` | the n=10 pilot that caught three harness bugs |
| `results/invalid/` | quarantined runs — the capped-budget K3 arm and the pre-channel-fix thinking probe, kept because they are evidence for #20 and #21 |
| `results/overlap-*.txt`, `retention-*.txt` | the expert-overlap matrix and the reference-mix counterfactual |
| `eval/analysis.py` | regenerates every table in RESULTS.md from the per-item records |

Three generations of the data are kept side by side rather than overwritten. Each
correction is a result in its own right, and the only way to show what unmatched
sampling or a loose classifier actually costs is to keep the before.

**Not in this repo:** `$MODELS/` holds the conceptual layer
(`docs/01-concepts.md`, `02-math.md`, `03-protocol.md`, `04-choosing-experts.md`,
`lab-notes.html`, `tools/fitcalc.py`, `tools/bwtest.py`) plus the 1.4 TB source weights
and the 326 GB build. It has a git dir with **zero commits and no remote** — the build
narrative is safe here, but that teaching material is not backed up anywhere.

**Open work** is enumerated as limitations 1–6 in RESULTS.md. Paired resampling is now
done; the control arm is the top remaining item, followed by the `hr-heavy`/`hr-only`
ablations, thinking mode at proper n, variance across prune plans, and a logit probe to
explain the response-initiation failure rather than merely measure it.

---

## Open questions (2026-08-02)

Item 1 below was the blocking one and is now **resolved**: thinking mode scores 35.0%
against 40.5% non-thinking, so K3's weak results are a property of the build, not of how
we invoked it. The rest stand, and are enumerated with cost estimates as limitations 1–5
in `RESULTS.md`.

1. ~~Thinking-mode confound~~ — **resolved**, see Phase 8.
2. ~~The language comparison is unpaired~~ (#23) — **resolved**, and it changed the
   answer. The matched rerun cost 53 minutes and cut the Croatian accuracy advantage
   from −16.0 to −7.0, at p=0.13. What survived is a compliance effect, not a
   comprehension one. See Finding 2 in RESULTS.md.
3. **No control arm.** `reap_subset --keep-sources code,en,zh,de,ru,fr` reproduces the
   reference mix from the *same* calibration run. Without it, Finding 2 shows this build
   favours Croatian but cannot attribute that to the corpus rather than to pruning
   damage generally. **This is the gap between "a model" and "a claim."**
4. **`hr-heavy` and `hr-only` arms unbuilt**, so language+code vs language-only is
   untested on a Slavic language.
5. **One build, one seed** — no variance estimate across prune plans.

---

## Transferable lessons

Ranked by how much time they would have saved.

1. **A tokenizer bug is indistinguishable from a broken model.** Verify the chat template
   before concluding anything about capability — and once you find a control-token bug in
   one direction, immediately check the other (#10 encode, #20 decode).
2. **The benchmark measures your configuration until proven otherwise.** Four separate
   fake results came from token budgets (#12, #13, #19, #21). A score that contradicts a
   related score is a harness bug, not a capability profile.
3. **Aggregate scores hide bimodal distributions.** Every headline number in this project
   was concealing a "did it respond at all" split. 25% pass@1 meant 51% correct on half
   the prompts and silence on the rest. Always report the conditional alongside the mean.
4. **If a published figure cannot be regenerated by a command, it is an unverified
   claim** (#25). Two numbers in the first draft of RESULTS.md were wrong — both computed
   by hand in a shell, one against a mid-run snapshot. The reporting layer had no tests
   and no second reader, which made it the least-scrutinised code in the project and the
   only part anyone would actually read. Write the analysis script *before* the writeup;
   it audits you.
5. **Read individual records, not just distributions** (#28). A scoring bug that
   discarded correct answers survived the entire project because a ~10% echo rate looked
   plausible for all four models. It died the moment one raw generation was placed next
   to its own classification. Aggregates cannot show you a misclassification whose rate
   is believable.
6. **An average of rates is not a rate** (#26). Aggregate the numerator and denominator
   separately. Averaging per-item tok/s counted 83 silent items as "zero tokens per
   second" and weighted one-token generations equally with 200-token ones. The correction
   moved two models in opposite directions — and note which direction the *error* had
   favoured.
7. **Assert the properties you depend on, in code.** The manifest contract (#3), the
   checkpoint key (#18) and Belebele's parallelism (#23) were all documented in comments
   and all silently violated. Two now refuse to run; the third should.
8. **Quantisation profiles leave things untouched.** In a bandwidth-bound regime the
   *non*-expert weights dominate, because they are read every token regardless of
   routing. One flag: 4.5×.
9. **Check a suspicious number against a prior run before interpreting it** (#24). That
   reflex is the only thing that separated "K3 cannot code" from "K3 codes at 51% and is
   silent half the time."
10. **Calibrate once, subset many.** Separate calibration runs confound run-to-run
   variance with the effect being measured.
11. **When a consumer reads only a prefix, validate the prefix**, not the whole file.
12. **Cumulative counters cannot diagnose pathology.** Use a rate over a tail window.
13. **Pilot before committing hours — but do not trust a pilot to find tail failures.**
    n=10 caught three harness bugs in twenty minutes (#12, #13, #14) and completely
    missed two others (#19, #24) because heavy-tailed failures need scale to appear.
14. **A metric that cannot report bad news is worse than no metric** (#22). K3's
    truncation rate read 0% because the field was never set.
15. **Check the whole machine before blaming the job you are watching** — the memory
    exhaustion was a Docker VM idle for twelve days, not the calibration.
