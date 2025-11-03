# attractive path extraction & overlays
"""Attractive path extraction (greedy from Q) and overlay vs BFS baseline."""
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from visual_helper import parse_state_action_key, MOVE_SET

MOVE_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # tie-break order
_MOVE_DIR = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}

def _own_index_from_state(state):
    """Extract own_idx from either (own,other,cs,co,mask) or ((own,...), ...)."""
    s0 = state[0]
    tup = s0 if isinstance(s0, tuple) else state
    return tup[0]

def _best_action_collapsed(Q: dict, own_idx: int, allowed) -> tuple[Optional[str], Optional[float]]:
    best_a, best_q = None, None
    allowed = set(a for a in allowed if a in MOVE_SET)
    for raw_key, q in Q.items():
        parsed = parse_state_action_key(raw_key)
        if not parsed:
            continue
        state, action = parsed
        if action not in allowed:
            continue
        oi = _own_index_from_state(state)
        if oi is None or oi != own_idx:
            continue
        if (best_q is None) or (q > best_q):
            best_q, best_a = float(q), action
    return best_a, best_q

def _heuristic_step_toward(start_rc, goal_rc, allowed: Iterable[str]) -> str|None:
    """Greedy Manhattan direction toward goal if Q has no info."""
    r, c = start_rc; gr, gc = goal_rc
    prefs = []
    if r > gr: prefs.append("UP")
    if r < gr: prefs.append("DOWN")
    if c > gc: prefs.append("LEFT")
    if c < gc: prefs.append("RIGHT")
    # fall back to a stable order if blocked
    prefs += ["UP","RIGHT","DOWN","LEFT"]
    allowed = set(allowed)
    for a in prefs:
        if a in allowed:
            return a
    return None

def greedy_rollout(env: Any,
                   Q: Dict[Tuple[Tuple[Any, ...], str], float],
                   agent_id: str,
                   slice_fn: Callable[[int, int], Tuple[Any, ...]],  # kept for signature compatibility; unused now
                   start_cell: Tuple[int, int],
                   goal_cell: Tuple[int, int],
                   max_steps: int = 200) -> List[Tuple[int, int]]:
    """
    Greedy path using Q collapsed over context:
      - pick best action among all states whose own_idx == current cell
      - if no Q available, take a heuristic step toward the goal
      - break on loops or 'no progress'
    """
    env.reset_to(start_cell, goal_cell)
    path = [start_cell]
    seen = set([start_cell])

    for _ in range(max_steps):
        cur = path[-1]
        if cur == goal_cell:
            break

        allowed = [a for a in env.applicable_actions(agent_id) if a in _MOVE_DIR]
        if not allowed:
            break

        own_idx = cur[0] * env.W + cur[1]
        a, q = _best_action_collapsed(Q, own_idx, allowed)

        # no learned info for this cell → heuristic step toward goal
        if a is None:
            a = _heuristic_step_toward(cur, goal_cell, allowed)
            if a is None:
                break

        # step
        _, _, info = env.step(agent_id, a)
        nxt = tuple(info.get("pos_agent", cur))

        # guard: stalled or loop → stop
        if nxt == cur or nxt in seen:
            # try one heuristic alternative before giving up
            if a is not None:
                # remove the chosen action and try another heuristic direction
                alt_allowed = [x for x in allowed if x != a]
                alt = _heuristic_step_toward(cur, goal_cell, alt_allowed)
            else:
                alt = None
            if alt:
                _, _, info = env.step(agent_id, alt)
                nxt = tuple(info.get("pos_agent", cur))
                if nxt == cur or nxt in seen:
                    break
            else:
                break

        path.append(nxt)
        seen.add(nxt)

    return path

def draw_overlay(grid_shape: Tuple[int, int],
                 obstacles: List[Tuple[int, int]],
                 pickups: List[Tuple[int, int]],
                 drops: List[Tuple[int, int]],
                 path_q: Optional[List[Tuple[int, int]]],
                 path_bfs: Optional[List[Tuple[int, int]]],
                 outpath: str) -> None:
    """Overlay attractive (Q-greedy) and BFS shortest path on grid and save PNG."""
    H, W = grid_shape
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(np.zeros((H, W)), vmin=0, vmax=1)
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.3))
    for (r, c) in pickups:
        ax.add_patch(plt.Circle((c, r), 0.25, fill=True, alpha=0.8))
    for (r, c) in drops:
        ax.add_patch(plt.Circle((c, r), 0.25, fill=False, linewidth=2))

    if path_bfs:
        xs = [c for r, c in path_bfs]
        ys = [r for r, c in path_bfs]
        ax.plot(xs, ys, linestyle='--', linewidth=1.5, label='Shortest (BFS)')
    if path_q:
        xs = [c for r, c in path_q]
        ys = [r for r, c in path_q]
        ax.plot(xs, ys, linewidth=2.5, label='Attractive (Greedy Q)')

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
