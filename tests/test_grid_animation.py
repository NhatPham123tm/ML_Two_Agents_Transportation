import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import grid_animation

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def run():
    os.makedirs(OUTDIR, exist_ok=True)
    # build a simple step log: two agents moving to the right/down
    rows = []
    posF = [0,0]; posM = [4,0]
    for t in range(20):
        agent = 'F' if t % 2 == 0 else 'M'
        if agent == 'F':
            # move F down 4 steps, then right each step
            if t//2 < 4:
                posF = [posF[0] + 1, posF[1]]
                action = 'DOWN'
            else:
                posF = [posF[0], posF[1] + 1]
                action = 'RIGHT'
        else:
            posM = [posM[0], posM[1] + 1]
            action = 'RIGHT'
        rows.append({
            'global_step': t,
            'episode_idx': 0,
            'agent': agent,
            'pos_F': tuple(posF),
            'pos_M': tuple(posM),
            'action': action
        })
    df = pd.DataFrame(rows)
    grid_animation.animate(df, (5,5), obstacles=[(2,2)], outpath=os.path.join(OUTDIR, "traj.mp4"), fps=20)
    print("grid_anim: wrote traj.mp4")

if __name__ == "__main__":
    run()
