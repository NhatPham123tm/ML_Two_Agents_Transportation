# utils/plot.py
"""
Plot helpers for PD-World logs.

Functions:
  - aggregate_episodes(step_csv) -> DataFrame with per-episode metrics
  - plot_learning_curves(ep_df, out_png)
  - plot_coordination(ep_df, out_png)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_pair(s):
    """Parse '(r, c)' or already-a-tuple -> (int, int)."""
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    a, b = s.split(",")
    return int(a), int(b)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def aggregate_episodes(step_csv: str) -> pd.DataFrame:
    df = pd.read_csv(step_csv)

    # Robust parsing of tuple-like columns
    pf = df["pos_F"].apply(_parse_pair)
    pm = df["pos_M"].apply(_parse_pair)
    df["manhattan_FM"] = [ _manhattan(a, b) for a, b in zip(pf, pm) ]

    # Conflicts = times agents are on the same cell in the episode
    df["conflict"] = [ int(a == b) for a, b in zip(pf, pm) ]

    # Ensure numeric dtypes
    df["episode_idx"] = pd.to_numeric(df["episode_idx"], errors="coerce").fillna(0).astype(int)
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")

    g = df.groupby("episode_idx", as_index=False).agg(
        return_sum=("reward", "sum"),
        steps=("reward", "count"),
        conflicts=("conflict", "sum"),
        avg_manhattan_FM=("manhattan_FM", "mean"),
    )

    # Guarantee plain NumPy dtypes (helps plotting)
    for col in ["episode_idx", "return_sum", "steps", "conflicts", "avg_manhattan_FM"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")

    return g


def plot_learning_curves(ep_df: pd.DataFrame, out_png: str):
    # Handle empty DF gracefully
    if ep_df is None or len(ep_df) == 0:
        # write an empty placeholder figure
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Learning Curve (no episodes)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (sum of rewards)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

    x = ep_df["episode_idx"].to_numpy()
    y = ep_df["return_sum"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (sum of rewards)")
    ax.set_title("Learning Curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_coordination(ep_df: pd.DataFrame, out_png: str):
    if ep_df is None or len(ep_df) == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Coordination Metrics (no episodes)")
        ax.set_xlabel("Episode")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

    x = ep_df["episode_idx"].to_numpy()
    y_dist = ep_df["avg_manhattan_FM"].to_numpy()
    y_steps = ep_df["steps"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y_dist, label="Avg Manhattan(F,M)")
    ax2 = ax.twinx()
    ax2.plot(x, y_steps, linestyle="--", label="Steps/episode")
    ax.set_xlabel("Episode")
    ax.set_title("Coordination Metrics")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
