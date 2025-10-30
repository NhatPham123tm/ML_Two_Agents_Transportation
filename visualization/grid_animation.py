# step‑by‑step animation (agents, blocks, moves)

"""Step/episode animation writer (MP4) using Matplotlib FuncAnimation."""
from typing import Iterable, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd

def animate(log_df: pd.DataFrame,grid_shape: Tuple[int, int],
            obstacles: List[Tuple[int, int]], outpath: str,
            fps: int = 10) -> None:
    # Create an MP4 animation from a step log DataFrame.
    # Required columns in log_df: ['global_step','episode_idx','agent','pos_F','pos_M','action']
    H, W = grid_shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.25))

    f_dot = ax.plot([], [], 'o', markersize=10, label='F')[0]
    m_dot = ax.plot([], [], 'o', markersize=10, label='M')[0]
    ax.legend()

    steps = log_df.sort_values('global_step').reset_index(drop=True)

    def init():
        f_dot.set_data([], [])
        m_dot.set_data([], [])
        return f_dot, m_dot

    def update(i):
        row = steps.iloc[i]
        rF, cF = row['pos_F']
        rM, cM = row['pos_M']
        f_dot.set_data([cF], [rF])
        m_dot.set_data([cM], [rM])
        ax.set_title(f"ep={row['episode_idx']} step={row['global_step']} a={row['agent']} act={row['action']}")
        return f_dot, m_dot

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(steps), interval=max(1, int(1000 // fps)), blit=True)
    writer = FFMpegWriter(fps=fps)
    anim.save(outpath, writer=writer)
    plt.close(fig)
