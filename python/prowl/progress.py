"""Progress and time estimates for `prowl run` (stderr), learned from previous runs on this machine.

Calculation.analyze() reports phases and projection steps to a ``progress`` object (any object with
``phase(name, **info)``, ``step(done, total)`` and ``finish(timings)``; a plain callable ``f(k)`` is still accepted
and receives the last finished k index). The projection dominates large runs and its cost per k-point is nearly
constant, so its remaining time is extrapolated from the measured rate after the first chunk of k-points. Before
that, and for the total, the estimate uses the median rates of earlier runs stored in
``$XDG_CACHE_HOME/prowl/timings.jsonl`` (``~/.cache/...``; disable with PROWL_TIMING_HISTORY=0).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def history_path() -> Path | None:
    if os.environ.get("PROWL_TIMING_HISTORY", "1") == "0":
        return None
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "prowl" / "timings.jsonl"


def load_history(backend: str, threads: int, limit: int = 50) -> list[dict]:
    p = history_path()
    if p is None or not p.exists():
        return []
    out = []
    try:
        for line in p.read_text().splitlines()[-500:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("backend") == backend and r.get("threads") == threads:
                out.append(r)
    except OSError:
        return []
    return out[-limit:]


def append_history(rec: dict) -> None:
    p = history_path()
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _median(v: list[float]) -> float | None:
    v = sorted(x for x in v if x and x > 0)
    return v[len(v) // 2] if v else None


def estimate(work: dict, backend: str, threads: int) -> dict:
    """Projection and total time from earlier runs: median s/GFLOP of the projection and median share of the
    projection in the total (other phases). Empty dict without history."""
    hist = load_history(backend, threads)
    rate = _median([r["projection_s"] / r["gflop"] for r in hist if r.get("gflop")])
    share = _median([r["projection_s"] / r["total_s"] for r in hist if r.get("total_s")])
    if rate is None:
        return {}
    proj = rate * work["gflop"]
    return {"projection_s": proj, "total_s": proj / share if share else None, "runs": len(hist)}


def fmt_s(s: float | None) -> str:
    if s is None:
        return "?"
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d} min" if m < 60 else f"{m // 60}:{m % 60:02d}:{sec:02d} h"


class ProgressReporter:
    """Human-readable progress on a stream (default stderr): one line per phase, a live projection line with
    percentage and remaining time, and the estimated total. Non-terminal streams get a line every 10 %."""

    def __init__(self, stream=None, backend: str = "cpu", threads: int = 1):
        self.out = stream or sys.stderr
        self.tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.backend, self.threads = backend, threads
        self.t0 = time.time()
        self.t_phase = self.t0
        self.phase_name = None
        self.work: dict = {}
        self.proj_t0 = None
        self.last_print = 0.0
        self.last_pct = -10
        self.live = False

    def _write(self, text: str, live: bool = False) -> None:
        """live: a line that the next write replaces (terminals only; elsewhere every line is kept)."""
        live = live and self.tty
        if self.live:
            self.out.write("\r\033[K")
        self.out.write(text if live else text + "\n")
        self.out.flush()
        self.live = live

    def _close_phase(self) -> None:
        if self.phase_name:
            self._write(f"  {self.phase_name:<20s} {fmt_s(time.time() - self.t_phase)}")

    def phase(self, name: str, **info) -> None:
        self._close_phase()
        self.phase_name, self.t_phase = name, time.time()
        if name == "projection":
            self.work = info
            self.backend = info.get("backend", self.backend)
            self.proj_t0 = time.time()
            est = estimate(info, self.backend, self.threads)
            msg = (f"  projection ({self.backend}): {info['nk']} k-points x {info['nspin']} spin, {info['nbands']} bands, "
                   f"{info['norb']} orbitals, {info['gflop']:.1f} GFLOP")
            if est:
                msg += (f"; estimate {fmt_s(est['projection_s'])}" +
                        (f", total ~{fmt_s(est['total_s'])}" if est.get("total_s") else "") +
                        f" (from {est['runs']} earlier runs)")
            else:
                msg += "; estimate after the first k-points"
            self._write(msg)

    def step(self, done: int, total: int) -> None:
        now = time.time()
        el = now - (self.proj_t0 or now)
        frac = done / max(total, 1)
        rem = el / frac * (1.0 - frac) if frac > 0 else None
        pct = int(100 * frac)
        if self.tty:
            if now - self.last_print < 0.5 and done < total:
                return
        elif pct < self.last_pct + 10 and done < total:
            return
        self.last_print, self.last_pct = now, pct
        self._write(f"  projection {done}/{total} k ({pct:3d} %)  elapsed {fmt_s(el)}  remaining ~{fmt_s(rem)}",
                    live=done < total)

    def __call__(self, k: int) -> None:  # plain-callable compatibility
        pass

    def finish(self, timings: dict) -> None:
        self._close_phase()
        self.phase_name = None
        total = timings.get("total", time.time() - self.t0)
        self._write(f"  {'total':<20s} {fmt_s(total)}")
        if self.work.get("gflop") and timings.get("projection"):
            append_history({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": self.backend,
                            "threads": self.threads, "gflop": self.work["gflop"], "nk": self.work["nk"],
                            "projection_s": timings["projection"], "total_s": total})
