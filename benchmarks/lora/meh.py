from traces import plot_arrivals
import pandas as pd
from pathlib import Path
import os

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd



def plot_all_arrivals_for_sampled_adapters(
        arrivals: pd.DataFrame,
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

    sub = df["adapter_id"].unique()

    t = df["t_sec"].to_numpy(dtype=np.float64)
    t_max = float(np.max(t))
    nbins = max(1, int(np.ceil(t_max / bin_seconds)))
    edges = np.linspace(0.0, nbins * bin_seconds, nbins + 1)
    x_hours = edges[:-1] / 3600.0

    plt.figure(figsize=(12, 5))

    # For each adapter, compute histogram counts over time bins
    for aid in sorted(sub):
        t_a = df.loc[df["adapter_id"] == aid, "t_sec"].to_numpy(dtype=np.float64)
        counts, _ = np.histogram(t_a, bins=edges)
        plt.step(x_hours, counts, where="post", linewidth=1.5, label=f"adapter {aid}")

    plt.xlabel("Time since window start (hours)")
    plt.ylabel(f"Arrivals per {bin_seconds}s")
    if title is None:
        title = f"Arrivals over time for sampled adapters (n={np.inf})"
    plt.title(title)
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Saved sampled-adapter time-series plot: {out_png}")

PATH = "/home/ferran/Documents/repositories/vLLMAdapterServingScaling/benchmarks/lora/definitive_results/single_gpu_behaviour_azure/test/__256_2_proposal-starvation-2"
df = pd.read_csv(os.path.join(PATH, "arrivals.csv"))
df = df.rename(columns={
    "adapter": "adapter_id",
    "arrival_time": "t_sec"
})
df["t_sec"] -= df["t_sec"][0]
df["t_sec"] += 3600
plot_arrivals(df, "Cut trace", Path(os.path.join(PATH, "REAL_cut_trace.png")), 60 // 6)
plot_all_arrivals_for_sampled_adapters(
    arrivals=df,
    out_png=Path(os.path.join(PATH, "REAL_cut_trace_per_adapter.png")),
    bin_seconds=60,
)