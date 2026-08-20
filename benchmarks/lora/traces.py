#!/usr/bin/env python3
"""trace_to_adapter_arrivals.py

Load and transform public traces into an *adapter arrival stream* suitable for evaluating
LLM-adapter placement algorithms.

Supported sources
-----------------
1) Microsoft Azure LLM inference traces (AzurePublicDataset)
   - CSV with columns: TIMESTAMP, ContextTokens, GeneratedTokens
   - TIMESTAMP is a datetime string (e.g., '2023-11-16 18:15:46.6805900')

2) Alibaba GenAI Serving Top-Down Dataset 2026 (GenTD26) / cluster-trace-v2026-GenAI
   - Request-level file: lora_request_trace.csv
   - Key columns include: gmt_create (datetime), prompt_length, num_images_per_prompt,
     num_inference_steps, checkpoint_model_version_id, num_lora

This script produces a unified output CSV/Parquet with (at least):
  - t_sec: float seconds from trace start (monotonic, starts at 0)
  - adapter_id: int adapter index in [0, num_adapters)
  - request_id: unique id within the output
  - source: 'azure' or 'gentd26'
  - plus source-specific fields

Adapter mapping
---------------
By default, requests are mapped to adapters via a categorical distribution over adapters.
For GenTD26, you can also use the trace's `num_lora` to emit multiple adapter arrivals
per request (multi-adapter mode).

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Time Utilities
# ---------------------------------------------------------------------

def parse_dt_utc_to_naive(series: pd.Series) -> pd.Series:
    """Parse timestamps as UTC, then convert to timezone-naive datetime."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Found {bad} unparsable timestamps.")
    return dt.dt.tz_convert(None)


def to_seconds_from_start(series: pd.Series) -> pd.Series:
    """Convert datetime-like series to seconds since its minimum."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Found {bad} unparsable timestamps.")
    t0 = dt.min()
    return (dt - t0).dt.total_seconds()


# ---------------------------------------------------------------------
# Adapter Weight Helpers
# ---------------------------------------------------------------------

def make_weights(num_adapters: int, spec: str, seed: int) -> np.ndarray:
    """Create a probability vector over adapters.

    spec formats:
      - "uniform"
      - "zipf:<alpha>"  (alpha>0; larger => heavier head)
      - "dirichlet:<concentration>" (smaller => spikier)
      - "file:<path.json>" (JSON list of weights or dict {id:weight})

    Returns normalized weights (sum=1).
    """
    rng = np.random.default_rng(seed)

    if spec == "uniform":
        w = np.ones(num_adapters, dtype=np.float64)
        return w / w.sum()

    if spec.startswith("zipf:"):
        alpha = float(spec.split(":", 1)[1])
        ranks = np.arange(1, num_adapters + 1, dtype=np.float64)
        w = 1.0 / np.power(ranks, alpha)
        return w / w.sum()

    if spec.startswith("dirichlet:"):
        conc = float(spec.split(":", 1)[1])
        w = rng.dirichlet(alpha=np.full(num_adapters, conc, dtype=np.float64))
        return w / w.sum()

    if spec.startswith("file:"):
        p = Path(spec.split(":", 1)[1]).expanduser()
        obj = json.loads(p.read_text())

        w = np.zeros(num_adapters, dtype=np.float64)
        if isinstance(obj, list):
            if len(obj) != num_adapters:
                raise ValueError(f"Weight list length {len(obj)} != num_adapters {num_adapters}")
            w[:] = np.asarray(obj, dtype=np.float64)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                w[int(k)] = float(v)
        else:
            raise ValueError("Weight file must be JSON list or dict")

        if (w < 0).any():
            raise ValueError("Weights must be non-negative")
        if w.sum() == 0:
            raise ValueError("All weights are zero")
        return w / w.sum()

    raise ValueError(f"Unknown weights spec: {spec}")


def sample_adapters(
        n: int,
        weights: np.ndarray,
        rng: np.random.Generator,
        replace: bool = True,
) -> np.ndarray:
    """Sample adapter ids."""
    return rng.choice(len(weights), size=n, replace=replace, p=weights)


def sample_adapters_multinomial_with_min1(
        n: int,
        weights: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    K = len(weights)
    if n < K:
        raise ValueError("n must be >= number of adapters")

    # reserve 1 per adapter
    base = np.ones(K, dtype=int)
    remaining = n - K

    # renormalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    extra = rng.multinomial(remaining, weights)
    counts = base + extra

    adapter_ids = np.repeat(np.arange(K), counts)
    rng.shuffle(adapter_ids)

    return adapter_ids


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_azure_trace(path: Path) -> pd.DataFrame:
    """Load Azure LLM inference trace CSV.

    Expected columns: TIMESTAMP, ContextTokens, GeneratedTokens
    """
    df = pd.read_csv(path)

    # Normalize column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    required = ["timestamp", "contexttokens", "generatedtokens"]
    for r in required:
        if r not in cols:
            raise ValueError(
                f"Azure trace missing column '{r}'. Found columns: {list(df.columns)}"
            )

    df = df.rename(
        columns={
            cols["timestamp"]: "timestamp",
            cols["contexttokens"]: "context_tokens",
            cols["generatedtokens"]: "generated_tokens",
        }
    )

    df["dt"] = parse_dt_utc_to_naive(df["timestamp"])
    df["t_sec"] = to_seconds_from_start(df["timestamp"])

    # Ensure numeric token counts
    df["context_tokens"] = pd.to_numeric(df["context_tokens"], errors="coerce")
    df["generated_tokens"] = pd.to_numeric(df["generated_tokens"], errors="coerce")
    if df[["context_tokens", "generated_tokens"]].isna().any().any():
        raise ValueError("Found non-numeric token counts in Azure trace")

    df = df.sort_values("t_sec").reset_index(drop=True)
    df["source"] = "azure"
    return df[["dt", "t_sec", "context_tokens", "generated_tokens", "source"]]


def load_gentd26_lora_requests(input_path: Path) -> pd.DataFrame:
    """Load GenTD26 request-level data.

    Accepts:
      - an extracted CSV named like lora_request_trace.csv
    """
    path = input_path
    tmpdir = None

    df = pd.read_csv(path)

    if "gmt_create" not in df.columns:
        raise ValueError(
            f"GenTD26 lora_request_trace missing 'gmt_create'. Columns: {list(df.columns)}"
        )

    df["t_sec"] = to_seconds_from_start(df["gmt_create"])

    # Normalize a few common numeric fields (if present)
    numeric_cols = [
        "exec_time_seconds",
        "prompt_length",
        "negative_prompt_length",
        "num_images_per_prompt",
        "num_inference_steps",
        "num_lora",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("t_sec").reset_index(drop=True)
    df["source"] = "gentd26"

    keep = [
        "t_sec",
        "gmt_create",
        "predict_type",
        "predict_status",
        "exec_time_seconds",
        "groupId",
        "prompt_length",
        "negative_prompt_length",
        "num_images_per_prompt",
        "num_inference_steps",
        "checkpoint_model_version_id",
        "num_lora",
        "source",
    ]
    keep = [c for c in keep if c in df.columns]

    if tmpdir is not None:
        # Best-effort cleanup is left to the OS; we expose tmpdir via local var only (same behavior).
        pass

    return df[keep]


# ---------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------

def to_adapter_arrivals(
        df: pd.DataFrame,
        num_adapters: int,
        weights_spec: str,
        seed: int,
        source: str,
) -> pd.DataFrame:
    """Map requests to adapters, producing a row per (request, adapter) arrival."""
    if "t_sec" not in df.columns:
        raise ValueError("Input df must have 't_sec'")

    rng = np.random.default_rng(seed)
    weights = make_weights(num_adapters, weights_spec, seed=seed)
    req_ids = np.arange(len(df), dtype=np.int64)

    if source == "azure":
        adapter_ids = sample_adapters_multinomial_with_min1(len(df), weights, rng)
        out = df.copy()
        out["request_id"] = req_ids
        out["adapter_id"] = adapter_ids.astype(np.int32)
        return out

    raise ValueError(f"Unknown source: {source}")


def multiply_arrivals(
        df: pd.DataFrame,
        k: int,
        seed: int,
        jitter_eps: float = 0.0,
) -> pd.DataFrame:
    """Replicate each row k times to increase arrival volume before adapter assignment.

	Parameters
	----------
	df : DataFrame with at least 't_sec'
	k : int replication factor (>=1)
	seed : used only if jitter_eps > 0
	jitter_eps : if >0, adds uniform noise in [-eps, eps] seconds per replicated row

	Notes
	-----
	- When jitter_eps == 0.0, timestamps are duplicated exactly.
	- We keep row order stable (original order repeated k times).
	"""
    if k <= 1:
        return df

    if "t_sec" not in df.columns:
        raise ValueError("Input df must have 't_sec' to multiply arrivals")

    # Repeat rows: [row0..rowN-1, row0..rowN-1, ...] k times
    out = df.loc[df.index.repeat(k)].copy()
    out.reset_index(drop=True, inplace=True)

    if jitter_eps and jitter_eps > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-float(jitter_eps), float(jitter_eps), size=len(out))
        out["t_sec"] = out["t_sec"].to_numpy(dtype=np.float64) + noise
        # keep times non-negative
        out["t_sec"] = out["t_sec"].clip(lower=0.0)
        # maintain monotonic-ish ordering for downstream plotting/logic
        out = out.sort_values("t_sec").reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Cutting utilities
# ---------------------------------------------------------------------

def cut_by_date_range(df: pd.DataFrame, start: str, end: str, rebase_t: bool = True) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    out = df[(df["dt"] >= start_dt) & (df["dt"] < end_dt)].copy()
    if rebase_t and not out.empty:
        out["t_sec"] = (out["dt"] - out["dt"].min()).dt.total_seconds()
    return out


def cut_by_day(df: pd.DataFrame, day: str, rebase_t: bool = True) -> pd.DataFrame:
    day0 = pd.to_datetime(day)
    day1 = day0 + pd.Timedelta(days=1)
    return cut_by_date_range(df, str(day0), str(day1), rebase_t=rebase_t)


def cut_by_hours(df: pd.DataFrame, start: str, hours: float = 2.0, rebase_t: bool = True) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = start_dt + pd.Timedelta(hours=float(hours))
    return cut_by_date_range(df, str(start_dt), str(end_dt), rebase_t=rebase_t)


def cut_trace(
        df: pd.DataFrame,
        outdir: Path,
        cut_per_day: str,
        cut_per_hour_start: str,
        cut_per_hour_end: float,
        bin_seconds: int,
) -> pd.DataFrame:
    plot_arrivals(df, "Complete Azure arrivals", outdir / "azure_range.png", bin_seconds)

    df_day = cut_by_day(df, cut_per_day, rebase_t=True)
    plot_arrivals(df_day, f"Azure arrivals (day {cut_per_day})", outdir / "azure_day.png", bin_seconds)

    # Preserve original behavior: mutate cut_per_hour_start in-place.
    cut_per_hour_start = combine_day_time(cut_per_day, cut_per_hour_start)
    df_2h = cut_by_hours(df_day, cut_per_hour_start, hours=cut_per_hour_end, rebase_t=True)

    '''plot_arrivals(
        df_2h,
        f"Azure arrivals ({cut_per_hour_start} + {cut_per_hour_end}h)",
        outdir / "azure_2h.png",
        max(1, bin_seconds // 6),
    )'''

    def span_seconds(x: pd.DataFrame) -> float:
        return float(x["t_sec"].max() - x["t_sec"].min()) if not x.empty else 0.0

    print("Spans (seconds):")
    print(f"  range: {span_seconds(df):.1f}")
    print(f"  day:   {span_seconds(df_day):.1f}")
    print(f"  2h:    {span_seconds(df_2h):.1f}")

    return df_2h


def subsample_adapters(
        df: pd.DataFrame,
        rng: np.random.Generator,
        adapters_subset_num: int,
) -> pd.DataFrame:
    unique_adapters = df["adapter_id"].unique()
    selected_adapters = rng.choice(unique_adapters, size=adapters_subset_num, replace=False)
    filtered_df = df[df["adapter_id"].isin(selected_adapters)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_arrivals(df: pd.DataFrame, title: str, out_png: Optional[Path] = None, bin_seconds: int = 60) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("Nothing to plot: dataframe is empty after cutting.")

    t = df["t_sec"].to_numpy()
    t_max = float(np.max(t))
    nbins = max(1, int(np.ceil(t_max / bin_seconds)))
    bins = np.linspace(0.0, nbins * bin_seconds, nbins + 1)

    counts, edges = np.histogram(t, bins=bins)
    x = edges[:-1] / 3600.0

    plt.figure(figsize=(10, 3.5))
    plt.step(x, counts, where="post")
    plt.xlabel("Time since window start (hours)")
    plt.ylabel(f"Arrivals per {bin_seconds}s")
    plt.title(title)
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        print(f"Saved plot: {out_png}")
    else:
        plt.show()
    plt.close()


def combine_day_time(day: str, maybe_time: str) -> str:
    """If user passes only HH:MM[:SS], prefix with day."""
    if len(maybe_time) <= 8 and ":" in maybe_time and "-" not in maybe_time:
        return f"{day} {maybe_time}"
    return maybe_time


def plot_all_arrivals_for_sampled_adapters(
        arrivals: pd.DataFrame,
        sampled_adapters: np.ndarray,
        out_png: Path,
        bin_seconds: int = 60,
        title: Optional[str] = None,
) -> None:
    """Plot arrivals over time for the specified adapter IDs (all rows, binned)."""
    import matplotlib.pyplot as plt

    if arrivals.empty:
        raise ValueError("Nothing to plot: arrivals dataframe is empty.")
    if "t_sec" not in arrivals.columns or "adapter_id" not in arrivals.columns:
        raise ValueError("arrivals must contain 't_sec' and 'adapter_id' columns.")

    bin_seconds = max(1, int(bin_seconds))
    sampled_set = set(int(x) for x in sampled_adapters.tolist())

    sub = arrivals[arrivals["adapter_id"].isin(sampled_set)].copy()
    if sub.empty:
        raise ValueError("No arrivals found for the sampled adapters (unexpected unless filtered).")

    t = sub["t_sec"].to_numpy(dtype=np.float64)
    t_max = float(np.max(t))
    nbins = max(1, int(np.ceil(t_max / bin_seconds)))
    edges = np.linspace(0.0, nbins * bin_seconds, nbins + 1)
    x_hours = edges[:-1] / 3600.0

    plt.figure(figsize=(12, 5))

    # For each adapter, compute histogram counts over time bins
    for aid in sorted(sampled_set):
        t_a = sub.loc[sub["adapter_id"] == aid, "t_sec"].to_numpy(dtype=np.float64)
        counts, _ = np.histogram(t_a, bins=edges)
        plt.step(x_hours, counts, where="post", linewidth=1.5, label=f"adapter {aid}")

    plt.xlabel("Time since window start (hours)")
    plt.ylabel(f"Arrivals per {bin_seconds}s")
    if title is None:
        title = f"Arrivals over time for sampled adapters (n={len(sampled_set)})"
    plt.title(title)
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Saved sampled-adapter time-series plot: {out_png}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert public traces into adapter arrivals")

    p.add_argument("--source", choices=["azure", "gentd26"], required=True, help="Trace source to load")
    p.add_argument("--input", type=str, required=True, help="Path to trace file (Azure CSV / GenTD26 CSV or tar.gz)")
    p.add_argument("--num-adapters", type=int, required=True, help="Total adapters in your placement problem")
    p.add_argument("--num-adapters-subset", type=int, required=True, help="Subsample adapters")
    p.add_argument(
        "--weights",
        type=str,
        default="uniform",
        help="Adapter popularity distribution: uniform | zipf:alpha | dirichlet:conc | file:path.json",
    )
    p.add_argument(
        "--arrival-multiplier",
        type=int,
        default=1,
        help="Replicate each request K times BEFORE adapter assignment to increase arrival volume.",
    )
    p.add_argument(
        "--arrival-jitter-eps",
        type=float,
        default=0.0,
        help="Optional +/- seconds jitter applied after multiplying to break timestamp ties (0 disables).",
    )
    p.add_argument("--seed", type=int, default=0)

    # GenTD26-only
    p.add_argument(
        "--gentd26-adapter-mode",
        choices=["single", "multi"],
        default="single",
        help="How to translate GenTD26 num_lora into adapter arrivals",
    )
    p.add_argument(
        "--multi-replace",
        action="store_true",
        help="In GenTD26 multi mode, sample adapters with replacement",
    )

    p.add_argument("--output", type=str, required=True, help="Output path (.csv or .parquet)")

    # Azure only for time cutting and plotting
    p.add_argument("--cut-per-day", type=str, required=True, help="e.g. 2024-10-18")
    p.add_argument("--cut-per-hour-start", type=str, required=True, help="e.g. 09:00:00")
    p.add_argument("--cut-per-hour-end", type=float, default=2.0)
    p.add_argument("--bin-seconds", type=int, default=60)

    return p.parse_args()


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def reorder_standard_columns(arrivals: pd.DataFrame) -> pd.DataFrame:
    front = ["request_id", "t_sec", "adapter_id", "source"]
    cols = front + [c for c in arrivals.columns if c not in front]
    return arrivals[cols]


def write_arrivals_csv(arrivals: pd.DataFrame, output_path: Path) -> Path:
    """Preserve original behavior: treat --output as a directory and write arrivals.csv inside it."""
    os.makedirs(output_path, exist_ok=True)  # output_path may be a Path; os.makedirs accepts it.
    out_file = os.path.join(output_path, "arrivals.csv")  # keep original behavior/type (str)
    arrivals.to_csv(out_file, index=False)
    return Path(out_file)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    in_path: str,
    out_path: str,
    source: str,
    num_adapters: int,
    num_adapters_subset: int,
    weights: str,
    arrival_multiplier: int,
    arrival_jitter_eps: float,
    gentd26_adapter_mode: str,
    multi_replace: bool,
    cut_per_day: str,
    cut_per_hour_start: str,
    cut_per_hour_end: float,
    bin_seconds: int,
    seed: int,
) -> pd.DataFrame:
    in_path = Path(in_path).expanduser()
    out_path = Path(out_path).expanduser()
    if source == "azure":
        df = load_azure_trace(in_path)
        df = cut_trace(
            df=df,
            outdir=out_path,
            cut_per_day=cut_per_day,
            cut_per_hour_start=cut_per_hour_start,
            cut_per_hour_end=cut_per_hour_end,
            bin_seconds=bin_seconds,
        )
        plot_arrivals(df, "Cut trace", out_path / "cut_trace.png", bin_seconds // 6)
        df = multiply_arrivals(df, arrival_multiplier, seed=seed, jitter_eps=arrival_jitter_eps)
        plot_arrivals(df, "Increased arrivals", out_path / "increased_arrivals.png", bin_seconds // 6)
        arrivals = to_adapter_arrivals(
            df,
            num_adapters=num_adapters,
            weights_spec=weights,
            seed=seed,
            source="azure",
        )
        arrivals = subsample_adapters(
            df=arrivals,
            rng=np.random.default_rng(seed),
            adapters_subset_num=num_adapters_subset,
        )
        plot_arrivals(arrivals, "Subsampled trace", out_path / "subsampled_trace.png", bin_seconds // 6)
    else:
        raise NotImplemented("Trace not available in the Alibaba repo at the moment")
        df = load_gentd26_lora_requests(in_path)
        df = multiply_arrivals(df, arrival_multiplier, seed=seed, jitter_eps=arrival_jitter_eps)
        plot_arrivals(df, "Increased arrivals", out_path / "increased_arrivals.png", bin_seconds // 6)
        arrivals = to_adapter_arrivals(
            df,
            num_adapters=num_adapters,
            weights_spec=weights,
            seed=seed,
            source="gentd26",
            gentd26_adapter_mode=gentd26_adapter_mode,
            multi_replace=multi_replace,
        )

    arrivals = reorder_standard_columns(arrivals)

    unique_adapters = arrivals["adapter_id"].unique()
    sampled = np.random.default_rng(seed).choice(unique_adapters, size=20, replace=False)
    plot_all_arrivals_for_sampled_adapters(
        arrivals=arrivals,
        sampled_adapters=sampled,
        out_png=out_path / "sample_adapters_arrivals.png",
        bin_seconds=bin_seconds,
        title=(
            f"Arrivals over time for sampled adapters "
            f"(n={len(sampled)}, bin={bin_seconds}s, "
            f"uniform sample)"
        ),
    )

    return arrivals


if __name__ == "__main__":
    args = parse_args()
    arrivals = main(
        in_path=args.input,
        out_path=args.output,
        source=args.source,
        num_adapters=args.num_adapters,
        num_adapters_subset=args.num_adapters_subset,
        weights=args.weights,
        arrival_multiplier=args.arrival_multiplier,
        arrival_jitter_eps=args.arrival_jitter_eps,
        gentd26_adapter_mode=args.gentd26_adapter_mode,
        multi_replace=args.multi_replace,
        cut_per_day=args.cut_per_day,
        cut_per_hour_start=args.cut_per_hour_start,
        cut_per_hour_end=args.cut_per_hour_end,
        bin_seconds=args.bin_seconds,
        seed=args.seed,
    )
    out_file = write_arrivals_csv(arrivals, args.output)
    print(f"Wrote {len(arrivals):,} arrival rows to {out_file}")
    print(arrivals.head(5).to_string(index=False))
