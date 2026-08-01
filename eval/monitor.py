"""Sample system memory and process footprint while a benchmark runs.

The interesting comparison in this project is not "which model scores higher" —
a 601B model beating a 31B model is unsurprising. It is what each one COSTS to
run on one machine. A model that scores 10% better while occupying 18x the RAM
and running 6x slower is a different proposition from one that scores 10% better
for free, and no published benchmark reports that.

Sampled without sudo, so this runs unattended:

  wired       what the model actually pinned in RAM (the honest footprint)
  compressed  macOS compressing pages under pressure — the early warning
  swap        pressure that got bad enough to hit disk
  page_ins    faulting from SSD, which is what an unwired model does on every
              decode step (the 27x failure mode documented in mlxmem.py)
  rss         the backend process itself

`free` is deliberately reported but not treated as a health signal — it sits
near zero on a healthy macOS box because unused RAM is wasted RAM. Read
`compressed` and `swap` instead.

Power draw would need `sudo powermetrics`, so it is left out; run that
separately if you want joules-per-token.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict

PAGE_RE = re.compile(r"^(.*?):\s+(\d+)", re.M)


def _vm_stat() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page_size = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page_size = int(m.group(1))
    stats: dict[str, int] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().rstrip(".")
        if val.isdigit():
            stats[key.strip()] = int(val)
    return {"_page_size": page_size, **stats}


def _swap_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def _rss_gb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        return int(out) / 1048576 if out else 0.0
    except Exception:
        return 0.0


def _rss_gb_by_pattern(pattern: str) -> float:
    """Largest RSS among processes matching `pattern`.

    Needed because ollama runs the model in a separate `llama-server` process —
    tracking our own pid there reports ~0 GB and makes a 20 GB model look free.
    """
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
        pids = [p for p in out.split() if p.isdigit()]
        return max((_rss_gb(int(p)) for p in pids), default=0.0)
    except Exception:
        return 0.0


@dataclass
class Sample:
    t: float
    free_gb: float
    active_gb: float
    wired_gb: float
    compressed_gb: float
    swap_mb: float
    pageins_total: int
    rss_gb: float
    label: str = ""


@dataclass
class Monitor:
    """Background sampler. Start before loading a model, stop after the run."""

    interval: float = 5.0
    pid: int | None = None
    proc_pattern: str | None = None
    out_path: pathlib.Path | None = None
    samples: list[Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _label: str = ""
    _t0: float = 0.0

    def label(self, text: str) -> None:
        """Tag subsequent samples, so phases are separable after the fact."""
        self._label = text

    def _sample(self) -> Sample:
        v = _vm_stat()
        ps = v.get("_page_size", 4096)
        gb = lambda pages: pages * ps / 1073741824  # noqa: E731
        return Sample(
            t=round(time.time() - self._t0, 1),
            free_gb=round(gb(v.get("Pages free", 0)), 2),
            active_gb=round(gb(v.get("Pages active", 0)), 2),
            wired_gb=round(gb(v.get("Pages wired down", 0)), 2),
            compressed_gb=round(gb(v.get("Pages occupied by compressor", 0)), 2),
            swap_mb=round(_swap_mb(), 1),
            pageins_total=v.get("Pageins", 0),
            rss_gb=round(_rss_gb_by_pattern(self.proc_pattern), 2) if self.proc_pattern
            else (round(_rss_gb(self.pid), 2) if self.pid else 0.0),
            label=self._label,
        )

    def _loop(self) -> None:
        f = open(self.out_path, "a") if self.out_path else None
        try:
            while not self._stop.is_set():
                s = self._sample()
                self.samples.append(s)
                if f:
                    f.write(json.dumps(asdict(s)) + "\n")
                    f.flush()
                self._stop.wait(self.interval)
        finally:
            if f:
                f.close()

    def start(self) -> Monitor:
        self._t0 = time.time()
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)

    def summary(self) -> dict:
        """Peaks and deltas — what goes in the results table."""
        if not self.samples:
            return {}
        pageins = [s.pageins_total for s in self.samples]
        page_size = 4096

        # Split page-ins by phase. Loading a 350 GB model necessarily reads
        # 350 GB off SSD — that is not a fault, it is the load. Faulting DURING
        # inference is the pathology (an unwired model re-reading weights every
        # decode step). Reporting one combined number makes a healthy run look
        # broken: measured 80.8 GB total, of which 80.5 GB was the load and
        # 0.3 GB was 420 s of inference.
        load = [s for s in self.samples if s.label == "load"]
        infer = [s for s in self.samples if s.label != "load"]

        def _delta_gb(rows) -> float:
            if len(rows) < 2:
                return 0.0
            vals = [r.pageins_total for r in rows]
            return round((max(vals) - min(vals)) * page_size / 1073741824, 2)

        # Total page-ins during "inference" is NOT evidence of thrashing, and
        # reading it that way is wrong. Both backends front-load: MLX reads the
        # weights in an explicit load phase, while llama.cpp mmaps the GGUF and
        # pages it in lazily on first touch — so ollama reports a model as
        # loaded before its pages are resident, and the reads land in the first
        # ~30s of inference. Measured: gpt-oss:120b paged 15.2 GB in 31s, then
        # 0.02 GB across the remaining 140s.
        #
        # Real thrashing is SUSTAINED, so measure the tail instead of the total.
        def _sustained_gb_per_min(rows) -> float:
            tail = rows[len(rows) // 2:]
            if len(tail) < 2:
                return 0.0
            span_min = (tail[-1].t - tail[0].t) / 60
            if span_min <= 0:
                return 0.0
            gb = (tail[-1].pageins_total - tail[0].pageins_total) * page_size / 1073741824
            return round(gb / span_min, 3)

        return {
            "pagein_gb_load": _delta_gb(load),
            "pagein_gb_inference": _delta_gb(infer),
            "pagein_gb_per_min_sustained": _sustained_gb_per_min(infer),
            "duration_s": round(self.samples[-1].t, 1),
            "n_samples": len(self.samples),
            "peak_wired_gb": max(s.wired_gb for s in self.samples),
            "peak_rss_gb": max(s.rss_gb for s in self.samples),
            "peak_active_gb": max(s.active_gb for s in self.samples),
            "peak_compressed_gb": max(s.compressed_gb for s in self.samples),
            "peak_swap_mb": max(s.swap_mb for s in self.samples),
            "min_free_gb": min(s.free_gb for s in self.samples),
            "pagein_gb_total": round((max(pageins) - min(pageins)) * page_size / 1073741824, 2),
        }


def attach(out_dir: pathlib.Path, label: str, pid: int | None = None,
           proc_pattern: str | None = None, interval: float = 5.0) -> Monitor:
    safe = label.replace(":", "-").replace("/", "-")
    return Monitor(
        interval=interval,
        pid=pid,
        proc_pattern=proc_pattern,
        out_path=pathlib.Path(out_dir) / f"memtrace-{safe}.jsonl",
    ).start()
