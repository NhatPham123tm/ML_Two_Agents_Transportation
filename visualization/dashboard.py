# learning curves & coordination metrics
"""Episode-level plots: learning curves and coordination metrics."""
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, grp in df.groupby('run_id'):
        ax.plot(grp['episode_idx'], grp['return_sum'], label=f'run {key}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (bank account)')
    ax.set_title('Learning Curves')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_coordination(df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['episode_idx'], df['avg_manhattan_FM'], label='Avg Manhattan(F,M)')
    ax2 = ax.twinx()
    ax2.plot(df['episode_idx'], df['conflicts'], label='Conflicts/episode', linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_title('Coordination Metrics')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
