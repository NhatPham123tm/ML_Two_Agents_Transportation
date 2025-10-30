import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import qmap

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def slice_fn_factory():
    # Canonical slice: other-agent fixed at (0,0), no carry, task_mask=0
    def slice_fn(r, c):
        # (self_pos_idx, other_pos_idx, carry_self, carry_other, task_mask)
        W = 5
        self_idx = r*W + c
        other_idx = 0
        return (self_idx, other_idx, 0, 0, 0)
    return slice_fn

def make_fake_Q():
    Q = {}
    W = 5
    def s_of(r,c): return (r*W+c, 0, 0, 0, 0)
    # Encourage moving RIGHT as c increases, DOWN as r increases
    for r in range(5):
        for c in range(5):
            s = s_of(r,c)
            Q[(s,'RIGHT')] = c
            Q[(s,'LEFT')]  = -c
            Q[(s,'UP')]    = -r
            Q[(s,'DOWN')]  = r
    return Q

def run():
    os.makedirs(OUTDIR, exist_ok=True)
    grid_shape = (5,5)
    Q = make_fake_Q()
    slice_fn = slice_fn_factory()
    qmap.q_heatmap(Q, grid_shape, slice_fn, os.path.join(OUTDIR, "q_heatmap.png"))
    qmap.policy_quiver(Q, grid_shape, slice_fn, os.path.join(OUTDIR, "policy_quiver.png"))
    print("qmaps: wrote q_heatmap.png and policy_quiver.png")

if __name__ == "__main__":
    run()
