# visual_helper.py
# Robust visualization utilities for PD-World.
# - Works with string-keyed Q (e.g., "[9,41,0,0,3]|UP") or tuple keys ((state), action).
# - Auto-infers a populated slice for heatmaps; collapsed quiver ignores context.
# - Greedy rollout collapses over context with heuristic fallback.
# - Generates overlays vs BFS baseline and optional animation.

from __future__ import annotations
from pathlib import Path
from ast import literal_eval
from collections import Counter
from typing import Dict, Tuple, List, Optional, Any, Iterable, Callable

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# --- Project imports ---
from environment.world import PDWorld
from environment.shortest_path import shortest_path
from agents.state import SliceFnFactory
from visualization.grid_animation import animate as animate_grid

# Public constants
MOVE_SET = {"UP", "DOWN", "LEFT", "RIGHT"}
MOVE_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]  # tie-break order
_DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

# ======================================================================
# Q-key handling
# ======================================================================

def parse_state_action_key(key: Any) -> Optional[Tuple[tuple, str]]:
    """
    Accepts:
      • (state_tuple5, action_str)
      • "[s0, s1, s2, s3, s4]|ACTION"
      • "(s0, s1, s2, s3, s4)|ACTION"
    Returns (state_tuple5, action_str) or None.
    """
    # Already tuple?
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str):
        st, a = key
        st = tuple(st) if not isinstance(st, tuple) else st
        return (st, a)

    # String format
    if isinstance(key, str) and "|" in key:
        st_str, a = key.rsplit("|", 1)
        a = a.strip()
        try:
            st_raw = literal_eval(st_str.strip())
        except Exception:
            return None
        if isinstance(st_raw, (list, tuple)) and len(st_raw) >= 5:
            st = tuple(int(x) for x in st_raw[:5])
            return (st, a)
    return None


def _unpack_state5(state) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Works for:
      state = (own_idx, other_idx, carry_self, carry_other, mask)
      state = ((own_idx, other_idx, carry_self, carry_other, mask), ...)
    """
    try:
        s0 = state[0]
        tup = s0 if isinstance(s0, tuple) and len(s0) >= 5 else state
        own_idx, other_idx, cs, co, mask = tup[:5]
        return int(own_idx), int(other_idx), int(cs), int(co), int(mask)
    except Exception:
        return None


def _own_index_from_state(state) -> Optional[int]:
    st = _unpack_state5(state)
    return st[0] if st else None


def infer_common_context(Q: dict) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (other_idx, carry_self, carry_other, mask) most frequently seen
    across movement entries in Q, regardless of key format.
    """
    other_ctr = Counter(); cs_ctr = Counter(); co_ctr = Counter(); mask_ctr = Counter()
    saw_any = False

    for raw_key, _q in Q.items():
        parsed = parse_state_action_key(raw_key)
        if not parsed:
            continue
        state, action = parsed
        if action not in MOVE_SET:
            continue
        st = _unpack_state5(state)
        if st is None:
            continue
        _, other_idx, cs, co, mask = st
        other_ctr[other_idx] += 1
        cs_ctr[cs] += 1
        co_ctr[co] += 1
        mask_ctr[mask] += 1
        saw_any = True

    if not saw_any or not other_ctr:
        return None

    def most(cnt: Counter) -> int:
        return int(cnt.most_common(1)[0][0])

    return most(other_ctr), most(cs_ctr), most(co_ctr), most(mask_ctr)

# ======================================================================
# Heatmaps
# ======================================================================

def _cell_qvals(Q: Dict, s: tuple) -> List[float]:
    """Collect Q(s, a) over move actions for a canonical state tuple s."""
    vals = []
    for k, q in Q.items():
        parsed = parse_state_action_key(k)
        if not parsed:
            continue
        st, a = parsed
        if a not in MOVE_SET:
            continue
        if st == s:
            vals.append(float(q))
    return vals


def q_heatmap(Q: Dict,
              grid_shape: Tuple[int, int],
              slice_fn: Callable[[int, int], tuple],
              outpath: str,
              stat: str = "max",
              center0: bool = True,
              percentile_clip: Optional[Tuple[float, float]] = (5, 95)) -> None:
    """
    Visualize Q values for a fixed canonical slice:
      - stat in {"max","mean","spread"}
      - NaN shown as gray when no data exists for a cell.
    """
    H, W = grid_shape
    Z = np.full((H, W), np.nan, dtype=float)

    for r in range(H):
        for c in range(W):
            s = slice_fn(r, c)
            vv = _cell_qvals(Q, s)
            if not vv:
                continue
            if stat == "max":
                Z[r, c] = float(np.max(vv))
            elif stat == "mean":
                Z[r, c] = float(np.mean(vv))
            elif stat == "spread":
                Z[r, c] = float(np.max(vv) - np.min(vv))
            else:
                Z[r, c] = float(np.max(vv))

    finite = np.isfinite(Z)
    fig, ax = plt.subplots(figsize=(6, 6))
    if finite.any():
        data = Z[finite]
        vmin, vmax = float(np.min(data)), float(np.max(data))

        if percentile_clip is not None and len(data) > 4:
            lo, hi = np.percentile(data, percentile_clip)
            # Only clip if the percentile range is valid
            if lo < hi:
                vmin, vmax = lo, hi

        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad((0.8, 0.8, 0.8, 1.0))
        if center0 and vmin < 0 < vmax:
            im = ax.imshow(Z, cmap=cmap,
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
                           interpolation="nearest")
        else:
            im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Q Heatmap ({stat})")
        fig.colorbar(im, ax=ax).set_label(f"Q {stat}")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No learned Q-values for selected slice.",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)

    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

# ======================================================================
# Quiver (collapsed over context)
# ======================================================================

def policy_quiver_any(Q: Dict,
                      grid_shape: Tuple[int, int],
                      out_png: str,
                      actions: Iterable[str] = ("UP", "DOWN", "LEFT", "RIGHT")) -> str:
    """
    Draw a quiver field: per cell, choose best move among all Q entries sharing own_idx.
    Ignores other_pos/carry/mask; robust even when slices are empty.
    """
    H, W = grid_shape
    X: List[int] = []; Y: List[int] = []; U: List[float] = []; V: List[float] = []; C: List[float] = []
    actions = tuple(a for a in actions if a in MOVE_SET)

    for r in range(H):
        for c in range(W):
            own = r * W + c
            best_a, best_q = None, None
            for k, q in Q.items():
                parsed = parse_state_action_key(k)
                if not parsed:
                    continue
                st, a = parsed
                if a not in actions:
                    continue
                oi = _own_index_from_state(st)
                if oi is None or oi != own:
                    continue
                if (best_q is None) or (q > best_q):
                    best_q, best_a = float(q), a
            if best_a is None:
                continue
            dr, dc = _DIRS[best_a]
            X.append(c); Y.append(r); U.append(dc); V.append(dr); C.append(0.0 if best_q is None else best_q)

    fig, ax = plt.subplots(figsize=(W * 0.6, H * 0.6))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks(range(W)); ax.set_yticks(range(H)); ax.grid(True, alpha=0.3)
    if X:
        QV = ax.quiver(X, Y, U, V, C, angles="xy", scale_units="xy", scale=1.5)
        cbar = fig.colorbar(QV, ax=ax, shrink=0.75); cbar.set_label("Q(max) per cell")
    else:
        ax.text(0.5, 0.5, "No learned moves yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Policy (collapsed over other/carry/mask)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

# ======================================================================
# Greedy rollout (collapsed) + overlay
# ======================================================================

def _best_action_collapsed(Q: Dict, own_idx: int, allowed: Iterable[str]) -> Tuple[Optional[str], Optional[float]]:
    best_a, best_q = None, None
    allowed = set(a for a in allowed if a in MOVE_SET)
    if not allowed:
        return None, None
    for k, q in Q.items():
        parsed = parse_state_action_key(k)
        if not parsed:
            continue
        st, a = parsed
        if a not in allowed:
            continue
        oi = _own_index_from_state(st)
        if oi is None or oi != own_idx:
            continue
        if (best_q is None) or (q > best_q):
            best_q, best_a = float(q), a
    return best_a, best_q


def _heuristic_step_toward(start_rc: Tuple[int, int], goal_rc: Tuple[int, int], allowed: Iterable[str]) -> Optional[str]:
    r, c = start_rc; gr, gc = goal_rc
    prefs = []
    if r > gr: prefs.append("UP")
    if r < gr: prefs.append("DOWN")
    if c > gc: prefs.append("LEFT")
    if c < gc: prefs.append("RIGHT")
    prefs += MOVE_ORDER  # stable fallback
    allowed = set(allowed)
    for a in prefs:
        if a in allowed:
            return a
    return None


def greedy_rollout(env: Any,
                   Q: Dict,
                   agent_id: str,
                   _slice_fn_unused: Callable[[int, int], tuple],  # kept for compat
                   start_cell: Tuple[int, int],
                   goal_cell: Tuple[int, int],
                   max_steps: int = 200) -> List[Tuple[int, int]]:
    """
    Greedy path using Q collapsed over context:
      - pick best action among all states with own_idx == current cell
      - if no Q available, take a heuristic step toward the goal
      - stop on loops or no progress
    """
    env.reset_to(start_cell, goal_cell, agent_id=agent_id)
    H, W = env.H, env.W
    path = [start_cell]
    seen = {start_cell}

    for _ in range(max_steps):
        cur = path[-1]
        if cur == goal_cell:
            break
        allowed = [a for a in env.applicable_actions(agent_id) if a in MOVE_SET]
        if not allowed:
            break
        own_idx = cur[0] * W + cur[1]
        a, _ = _best_action_collapsed(Q, own_idx, allowed)
        if a is None:
            a = _heuristic_step_toward(cur, goal_cell, allowed)
            if a is None:
                break
        _, _, info = env.step(agent_id, a)
        nxt = tuple(info.get("pos_agent", cur))
        if nxt == cur or nxt in seen:
            # try one alternative heuristic
            alt = _heuristic_step_toward(cur, goal_cell, [x for x in allowed if x != a])
            if alt:
                _, _, info = env.step(agent_id, alt)
                nxt = tuple(info.get("pos_agent", cur))
                if nxt == cur or nxt in seen:
                    break
            else:
                break
        path.append(nxt); seen.add(nxt)
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

    # Obstacles
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black", alpha=0.25))
    # Pickups (filled)
    for (r, c) in pickups:
        ax.add_patch(plt.Circle((c, r), 0.25, color="tab:green", fill=True, alpha=0.9))
    # Drops (outlined)
    for (r, c) in drops:
        ax.add_patch(plt.Circle((c, r), 0.28, fill=False, linewidth=2.2, color="tab:blue"))

    if path_bfs:
        xs = [c for r, c in path_bfs]; ys = [r for r, c in path_bfs]
        ax.plot(xs, ys, linestyle="--", linewidth=2.0, label="Shortest (BFS)")
    if path_q:
        xs = [c for r, c in path_q]; ys = [r for r, c in path_q]
        ax.plot(xs, ys, linewidth=3.0, label="Attractive (Greedy Q)")

    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)

# ======================================================================
# Helpers
# ======================================================================

def _safe_pickups_dict(d: dict) -> Dict[Tuple[int, int], int]:
    """env_spec['pickups'] may be {'(r, c)': n} or {(r,c): n}. Normalize to {(r,c): n}."""
    out: Dict[Tuple[int, int], int] = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) == 2:
            rc = (int(k[0]), int(k[1]))
        else:
            try:
                rc = literal_eval(str(k))
            except Exception:
                s = str(k).strip()
                if s.startswith("(") and s.endswith(")"):
                    s = s[1:-1]
                a, b = [p.strip() for p in s.split(",")]
                rc = (int(a), int(b))
        out[(int(rc[0]), int(rc[1]))] = int(v)
    return out

def _pickups_from_world(world: dict) -> Dict[Tuple[int,int], int]:
    # Prefer list form ONLY if this run recorded a mid-run change
    if world.get("pickups_changed_at_episode") is not None and world.get("pickups_list"):
        return {(int(r), int(c)): int(n) for r, c, n in world["pickups_list"]}
    # Fallback (default for other experiments): use 'pickups'
    return _safe_pickups_dict(world["pickups"])

# ======================================================================
# Orchestrator
# ======================================================================

def make_visuals(outdir: str, env_spec: dict, QF: dict, QM: dict) -> None:
    """
    Create static + animated visuals for a finished run.
    - Saves into {outdir}/viz/
      * F_maxQ_heatmap.png / M_maxQ_heatmap.png
      * F_policy_quiver.png / M_policy_quiver.png
      * overlay_F_greedyQ_vs_BFS.png / overlay_M_greedyQ_vs_BFS.png
      * trajectory.mp4 (if steps.csv exists)
    """
    vizdir = Path(outdir) / "viz"
    vizdir.mkdir(parents=True, exist_ok=True)

    # --- Rebuild env from spec (no side effects) ---
    env = PDWorld(
        H=env_spec["H"], W=env_spec["W"],
        obstacles=set(map(tuple, env_spec["obstacles"])),
        pickups=_pickups_from_world(env_spec),   # <-- changed
        drops=[tuple(d) for d in env_spec["drops"]],
        start_F=tuple(env_spec["start_F"]),
        start_M=tuple(env_spec["start_M"]),
    )

    # --- 1) Heatmaps + Policy arrows ---
    # Auto-pick contexts that actually exist in Q for canonical heatmaps.
    ctxF = infer_common_context(QF)
    ctxM = infer_common_context(QM)

    # F agent heatmap
    if ctxF:
        other_idx, cs, co, mask = ctxF
        other_pos_rc = (other_idx // env.W, other_idx % env.W)
        slicerF = SliceFnFactory(H=env.H, W=env.W,
                                 other_pos=other_pos_rc,
                                 carry_self=cs, carry_other=co,
                                 task_mask=mask).make()
    else:
        slicerF = SliceFnFactory(H=env.H, W=env.W, other_pos=env.start_M,
                                 carry_self=0, carry_other=0, task_mask=0).make()
    q_heatmap(QF, (env.H, env.W), slicerF, str(vizdir / "F_maxQ_heatmap.png"), stat="max")

    # M agent heatmap
    if ctxM:
        other_idx, cs, co, mask = ctxM
        other_pos_rc = (other_idx // env.W, other_idx % env.W)
        slicerM = SliceFnFactory(H=env.H, W=env.W,
                                 other_pos=other_pos_rc,
                                 carry_self=cs, carry_other=co,
                                 task_mask=mask).make()
    else:
        slicerM = SliceFnFactory(H=env.H, W=env.W, other_pos=env.start_F,
                                 carry_self=0, carry_other=0, task_mask=0).make()
    q_heatmap(QM, (env.H, env.W), slicerM, str(vizdir / "M_maxQ_heatmap.png"), stat="max")

    # Collapsed quivers (don’t depend on slice)
    policy_quiver_any(QF, (env.H, env.W), str(vizdir / "F_policy_quiver.png"))
    policy_quiver_any(QM, (env.H, env.W), str(vizdir / "M_policy_quiver.png"))

    # --- 2) Attractive path overlays (with BFS fallback) ---
    pickups = list(env.pickups.keys())
    drops = list(env.drops)
    if pickups and drops:
        start = pickups[0]
        goal = drops[0]

        env.reset_to(start, goal, agent_id='F')
        path_q_F = greedy_rollout(env, QF, 'F', slicerF, start, goal, max_steps=env.H * env.W * 2) or []
        env.reset_to(start, goal, agent_id='M')
        path_q_M = greedy_rollout(env, QM, 'M', slicerM, start, goal, max_steps=env.H * env.W * 2) or []

        path_bfs = shortest_path(start, goal, env.H, env.W, set(env.obstacles))

        draw_overlay((env.H, env.W), list(env.obstacles), [start], [goal],
                     path_q=path_q_F, path_bfs=path_bfs,
                     outpath=str(vizdir / "overlay_F_greedyQ_vs_BFS.png"))
        draw_overlay((env.H, env.W), list(env.obstacles), [start], [goal],
                     path_q=path_q_M, path_bfs=path_bfs,
                     outpath=str(vizdir / "overlay_M_greedyQ_vs_BFS.png"))

    # --- 3) Animation from steps.csv (with pickups/drops for cues) ---
    steps_csv = Path(outdir) / "steps.csv"
    if steps_csv.exists():
        try:
            import pandas as pd, json
            df = pd.read_csv(steps_csv)
        except Exception:
            df = None

        if df is not None:
            # Detect mid-run pickup change (from our log_event)
            change_rows = df[df["agent"] == "PICKUPS_CHANGED"]
            if not change_rows.empty:
                change_step = int(change_rows.iloc[0]["global_step"])
                # Parse target pickups from 'action' JSON payload if present
                new_pk = None
                try:
                    payload = change_rows.iloc[0]["action"]
                    if isinstance(payload, str) and payload.strip():
                        obj = json.loads(payload)
                        if "to" in obj:
                            # 'to' may be [[r,c],[r,c]] or [[r,c,n],...]
                            arr = obj["to"]
                            new_pk = {}
                            for item in arr:
                                r, c = int(item[0]), int(item[1])
                                n = int(item[2]) if len(item) >= 3 else 1
                                new_pk[(r, c)] = n
                except Exception:
                    pass

                # Old pickups come from env_spec["pickups"] (original)
                old_pk = _safe_pickups_dict(env_spec["pickups"])
                # If meta had pickups_list updated, prefer that as the "after" if payload missing
                if new_pk is None and "pickups_list" in env_spec:
                    new_pk = {(int(r), int(c)): int(n) for r, c, n in env_spec["pickups_list"]}

                # Split dataframe
                df_before = df[df["global_step"] <= change_step].copy()
                df_after  = df[df["global_step"] >= change_step].copy()

                # Render two clips
                if not df_before.empty:
                    animate_grid(
                        df_before,
                        (env.H, env.W),
                        list(env.obstacles),
                        str(vizdir / "trajectory_before.mp4"),
                        pickups=old_pk,
                        drops=env.drops,
                        fps=10,
                        show_rewards=True,
                        trail_len=30,
                    )
                if not df_after.empty and new_pk:
                    animate_grid(
                        df_after,
                        (env.H, env.W),
                        list(env.obstacles),
                        str(vizdir / "trajectory_after.mp4"),
                        pickups=new_pk,
                        drops=env.drops,
                        fps=10,
                        show_rewards=True,
                        trail_len=30,
                    )
            else:
                # No change event – single clip with (possibly updated) pickups
                animate_grid(
                    df,
                    (env.H, env.W),
                    list(env.obstacles),
                    str(vizdir / "trajectory.mp4"),
                    pickups=env.pickups,
                    drops=env.drops,
                    fps=10,
                    show_rewards=True,
                    trail_len=30,
                )