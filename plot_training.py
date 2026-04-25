"""Plot training metrics from CSV log files.

Usage examples:
  python plot_training.py
  python plot_training.py --logs-dir logs --pattern "*_missiles.csv" --show
  python plot_training.py --output-dir plots
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List, Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required. Install with: pip install pandas matplotlib"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_METRICS = [
    "TotalReward",
    "MovingAvgReward",
    "CurrentSuccessRate",
    "AgentEpsilon",
    "RewardIntercept",
    "RewardSpeedBonus",
    "RewardEarlyDetection",
    "RewardPredictionAccuracy",
    "RewardFuelEfficiency",
    "RewardDistanceBonus",
]


def _find_csv_files(logs_dir: str, pattern: str) -> List[str]:
    search = os.path.join(logs_dir, pattern)
    return sorted(glob.glob(search))


def _choose_episode_key(columns: Iterable[str]) -> Optional[str]:
    for key in ("EpisodeID", "AgentEpisodeCount"):
        if key in columns:
            return key
    return None


def _aggregate_per_episode(df: pd.DataFrame) -> pd.DataFrame:
    key = _choose_episode_key(df.columns)
    if key is None:
        return df

    df = df.copy()
    df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna(subset=[key]).sort_values(key)
    # Keep the last row for each episode to avoid multiple entries per episode.
    aggregated = df.groupby(key, as_index=False).tail(1)
    return aggregated


def _plot_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: str,
    title: str,
    smoothing: int = 0,
):
    key = _choose_episode_key(df.columns)
    if key is None:
        print(f"Skipping {title}: no episode key column found.")
        return

    x = pd.to_numeric(df[key], errors="coerce")
    df = df.assign(_episode=x).dropna(subset=["_episode"]).sort_values("_episode")

    available = [m for m in metrics if m in df.columns]
    if not available:
        print(f"Skipping {title}: no known metrics found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in available:
        y = pd.to_numeric(df[metric], errors="coerce")
        if smoothing and smoothing > 1:
            y = y.rolling(window=smoothing, min_periods=1).mean()
        ax.plot(df["_episode"], y, label=metric)

    ax.set_title(title)
    ax.set_xlabel(key)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV logs.")
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing CSV logs (default: logs)",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write plots (default: plots)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to plot (default: common training metrics)",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=0,
        help="Rolling mean window for smoothing (default: 0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving",
    )

    args = parser.parse_args()

    csv_files = _find_csv_files(args.logs_dir, args.pattern)
    if not csv_files:
        print(f"No CSV files found in {args.logs_dir} matching {args.pattern}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {csv_path}...")

        df = pd.read_csv(csv_path)
        df = _aggregate_per_episode(df)

        out_path = os.path.join(args.output_dir, f"{name}_training.png")
        _plot_metrics(
            df,
            metrics=args.metrics,
            out_path=out_path,
            title=f"Training Metrics: {name}",
            smoothing=args.smoothing,
        )

        if args.show:
            img = plt.imread(out_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Training Metrics: {name}")
            plt.show()

    print(f"Saved plots to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())