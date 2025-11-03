import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import path

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

class EnvStub:
    def __init__(self, H=5, W=5, obstacles=None):
        self.H, self.W = H, W
        self.obstacles = set(obstacles or [])
        self.agent_pos = (0,0)
        self.goal = (4,4)
    def inside(self, r,c):
        return 0 <= r < self.H and 0 <= c < self.W and (r,c) not in self.obstacles
    def reset_to(self, start_cell, goal_cell):
        self.agent_pos = tuple(start_cell)
        self.goal = tuple(goal_cell)
        return None
    def applicable_actions(self, agent_id):
        r,c = self.agent_pos
        acts = []
        if self.inside(r-1,c): acts.append('UP')
        if self.inside(r+1,c): acts.append('DOWN')
        if self.inside(r,c-1): acts.append('LEFT')
        if self.inside(r,c+1): acts.append('RIGHT')
        return acts
    def step(self, agent_id, action):
        r,c = self.agent_pos
        if action=='UP': r -= 1
        elif action=='DOWN': r += 1
        elif action=='LEFT': c -= 1
        elif action=='RIGHT': c += 1
        if not self.inside(r,c):
            r,c = self.agent_pos
        self.agent_pos = (r,c)
        info = {'pos_agent': (r,c)}
        return None, 0.0, info

def slice_fn_factory(H=5,W=5):
    def slice_fn(r,c):
        self_idx = r*W + c
        other_idx = 0 # fixed
        return (self_idx, other_idx, 0, 0, 0)
    return slice_fn

def bfs_path(start, goal, H=5, W=5, obstacles=None):
    from collections import deque
    obstacles = set(obstacles or [])
    def inside(r,c): return 0 <= r < H and 0 <= c < W and (r,c) not in obstacles
    q = deque([start]); par = {start: None}
    while q:
        u = q.popleft()
        if u == goal: break
        r,c = u
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            v = (r+dr, c+dc)
            if inside(*v) and v not in par:
                par[v] = u; q.append(v)
    if goal not in par: return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = par[cur]
    path.reverse()
    return path

def make_fake_Q():
    Q = {}
    W = 5
    def s_of(r,c): return (r*W+c, 0, 0, 0, 0)
    for r in range(5):
        for c in range(5):
            s = s_of(r,c)
            # bias towards moving right & down (towards (4,4))
            Q[(s,'RIGHT')] = c + 0.5
            Q[(s,'DOWN')]  = r + 0.5
            Q[(s,'LEFT')]  = -c
            Q[(s,'UP')]    = -r
    return Q

def run():
    os.makedirs(OUTDIR, exist_ok=True)
    env = EnvStub()
    slice_fn = slice_fn_factory()
    Q = make_fake_Q()
    start = (0,0); goal = (4,4)
    path_q = path.greedy_rollout(env, Q, 'F', slice_fn, start, goal, max_steps=64)
    path_b = bfs_path(start, goal)
    path.draw_overlay((5,5), obstacles=[], pickups=[start], drops=[goal],
                       path_q=path_q, path_bfs=path_b,
                       outpath=os.path.join(OUTDIR, "paths_overlay.png"))
    print("paths: wrote paths_overlay.png; len(path_q) =", len(path_q))

if __name__ == "__main__":
    run()
