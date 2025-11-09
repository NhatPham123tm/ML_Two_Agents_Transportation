# heatmaps & arrow fields
# Q-heatmaps and policy arrow-field visualizations for PD-World, relies on Matplotlib.

from typing import Dict, Tuple, Callable, Any
import numpy as np
import matplotlib.pyplot as plt

# Movement actions only (arrows); pickup/dropoff are excluded from arrow maps
ARROWS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
# Map actions to (dr, dc)
_DIRS = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}

def maxq_grid(Q: Dict[Tuple[Tuple[Any, ...], str], float],
              grid_shape: Tuple[int, int],
              slice_fn: Callable[[int, int], Tuple[Any, ...]]) -> np.ndarray:
    """Return an HxW array with max-Q over movement actions for each cell under a state slice.
    Parameters
    ----------
    Q : mapping of (state, action) -> value
    grid_shape : (H, W)
    slice_fn : (r, c) -> canonical state tuple for that cell (e.g., no-carry, fixed other-agent pos)
    """
    H, W = grid_shape
    Z = np.zeros((H, W), dtype=float)
    for r in range(H):
        for c in range(W):
            s = slice_fn(r, c)
            qs = [Q.get((s, a), 0.0) for a in ARROWS.keys()]
            Z[r, c] = max(qs) if qs else 0.0
    return Z

def _best_move_for_cell(Q, cell_idx, actions=("UP","DOWN","LEFT","RIGHT")):
    """
    Return (best_action, best_q) for a given own-position index, collapsing over
    all other dimensions in the state tuple.
    State key is assumed like: ( (own_idx, other_idx, carry_self, carry_other, mask), action )
    """
    best_a, best_q = None, None
    for (state, a), q in Q.items():
        if a not in actions:
            continue
        try:
            own_idx = state[0]
        except Exception:
            continue
        if own_idx != cell_idx:
            continue
        if (best_q is None) or (q > best_q):
            best_q, best_a = q, a
    return best_a, best_q

def policy_quiver_collapsed(Q, grid_shape, out_png, actions=("UP","DOWN","LEFT","RIGHT")):
    """
    Draw a quiver over the grid using the best move per cell, collapsing Q over
    other_pos/carry/mask. This will produce arrows as long as Q has *any*
    non-zero info for each own cell.
    """
    H, W = grid_shape
    X, Y, U, V, M = [], [], [], [], []

    for r in range(H):
        for c in range(W):
            idx = r*W + c
            a, q = _best_move_for_cell(Q, idx, actions=actions)
            if a is None:
                continue
            dr, dc = _DIRS[a]
            # Quiver expects x=column, y=row
            X.append(c); Y.append(r); U.append(dc); V.append(dr); M.append(q if q is not None else 0.0)

    fig, ax = plt.subplots(figsize=(W*0.6, H*0.6))
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, alpha=0.3)

    if X:
        # scale arrow length a bit
        QV = ax.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', scale=1.5)
        cbar = fig.colorbar(QV, ax=ax, shrink=0.75)
        cbar.set_label("Q(max) per cell")
    else:
        ax.text(0.5, 0.5, "No learned moves yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Policy (collapsed over mask/carry/other)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

MOVE_ACTIONS = ("UP","DOWN","LEFT","RIGHT")

def _cell_qvals(Q, s, actions=MOVE_ACTIONS):
    """Return list of q-values for state s over given actions (missing -> skip)."""
    vals = [Q[(s, a)] for a in actions if (s, a) in Q]
    return vals

def stat_grid(Q, grid_shape, slice_fn,
              actions=MOVE_ACTIONS,
              stat: str = "max",
              default=np.nan) -> np.ndarray:
    """
    Build HxW grid of a statistic over Q(s,a) for each cell's canonical slice.
    stat in {"max","mean","spread","adv"}:
        - max:   max_a Q(s,a)
        - mean:  mean_a Q(s,a)
        - spread: max_a Q - min_a Q
        - adv:   max_a Q - second_best_a Q   (0 if only one value)
    Missing cells -> default (NaN by default).
    """
    H, W = grid_shape
    Z = np.full((H, W), default, dtype=float)
    for r in range(H):
        for c in range(W):
            s = slice_fn(r, c)
            vals = _cell_qvals(Q, s, actions)
            if not vals:
                continue
            vals = np.array(vals, dtype=float)
            if stat == "max":
                Z[r, c] = float(vals.max())
            elif stat == "mean":
                Z[r, c] = float(vals.mean())
            elif stat == "spread":
                Z[r, c] = float(vals.max() - vals.min())
            elif stat == "adv":
                if vals.size == 1:
                    Z[r, c] = 0.0
                else:
                    top2 = np.partition(vals, -2)[-2:]
                    Z[r, c] = float(top2.max() - top2.min())
            else:
                raise ValueError(f"Unknown stat: {stat}")
    return Z

def q_heatmap(Q,
              grid_shape,
              slice_fn,
              outpath: str,
              actions=MOVE_ACTIONS,
              stat: str = "max",
              vmin: float | None = None,
              vmax: float | None = None,
              center0: bool = True,
              percentile_clip: tuple[float,float] | None = (5,95),
              walls: np.ndarray | None = None,
              title: str | None = None):
    """
    Render a Q-based heatmap with sane defaults and comparable scaling.

    - stat: "max", "mean", "spread", or "adv" (see stat_grid)
    - vmin/vmax: fixed color scale for comparability across runs;
                 if None, computed from finite values (optionally percentile-clipped).
    - center0: if True and vmin<0<vmax, use a diverging norm centered at 0.
    - walls: optional HxW boolean array; True cells overlaid in gray with alpha.
    """
    Z = stat_grid(Q, grid_shape, slice_fn, actions=actions, stat=stat, default=np.nan)
    finite = np.isfinite(Z)
    if not finite.any():
        # Nothing learned yet → show an informative figure
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No learned Q-values for selected slice.", ha="center", va="center", fontsize=14)
        fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)
        return

    data = Z[finite]
    if percentile_clip is not None:
        lo, hi = np.percentile(data, percentile_clip)
        data = np.clip(data, lo, hi)

    auto_vmin = float(np.nanmin(data))
    auto_vmax = float(np.nanmax(data))
    if vmin is None: vmin = auto_vmin
    if vmax is None: vmax = auto_vmax
    if vmin == vmax:  # edge case: constant surface
        vmin, vmax = vmin - 1e-6, vmax + 1e-6

    fig, ax = plt.subplots(figsize=(6,6))

    # Colormap setup: diverging centered at 0 if requested and straddling 0
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color=(0.8,0.8,0.8,1.0))  # NaN → light gray
    if center0 and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        im = ax.imshow(Z, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    # Optional walls overlay
    if walls is not None:
        ax.imshow(np.where(walls, 1.0, np.nan), cmap="gray", alpha=0.35, interpolation="nearest")

    ax.set_xticks(range(grid_shape[1])); ax.set_yticks(range(grid_shape[0]))
    ax.set_title(title or f"Q Heatmap ({stat})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Q {stat}")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

