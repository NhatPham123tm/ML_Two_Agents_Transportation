import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import dashboard

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def run():
    os.makedirs(OUTDIR, exist_ok=True)
    # Two runs with simple synthetic curves
    rows = []
    for run_id in [1,2]:
        returns = np.linspace(-10, 50, 20) + (run_id-1)*5
        conflicts = np.linspace(10, 2, 20) + (run_id-1)
        dist = np.linspace(4, 8, 20) - (run_id-1)*0.5
        for ep in range(20):
            rows.append({
                'run_id': run_id,
                'episode_idx': ep,
                'return_sum': float(returns[ep]),
                'conflicts': float(conflicts[ep]),
                'avg_manhattan_FM': float(dist[ep]),
            })
    df = pd.DataFrame(rows)
    dashboard.plot_learning_curves(df, os.path.join(OUTDIR, "learning_curves.png"))
    dashboard.plot_coordination(df[df['run_id']==1], os.path.join(OUTDIR, "coordination_run1.png"))
    print("dashboards: wrote learning_curves.png and coordination_run1.png")

if __name__ == "__main__":
    run()
