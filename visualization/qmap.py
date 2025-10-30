# heatmaps & arrow fields
# Q-heatmaps and policy arrow-field visualizations for PD-World, relies on Matplotlib.

from typing import Dict, Tuple, Callable, Any
import numpy as np
import matplotlib.pyplot as plt

# Movement actions only (arrows); pickup/dropoff are excluded from arrow maps
ARROWS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

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

def policy_quiver(Q: Dict[Tuple[Tuple[Any, ...], str], float],
                  grid_shape: Tuple[int, int],
                  slice_fn: Callable[[int, int], Tuple[Any, ...]],
                  outpath: str) -> None:
    """Greedy movement-policy arrow field for a canonical state slice and saves PNG.
    """
    H, W = grid_shape
    U = np.zeros((H, W))  # x-components
    V = np.zeros((H, W))  # y-components (note inverted image y-axis)
    for r in range(H):
        for c in range(W):
            s = slice_fn(r, c)
            best_a, best_q = None, float('-inf')
            for a, (dr, dc) in ARROWS.items():
                q = Q.get((s, a))
                if q is not None and q > best_q:
                    best_q, best_a = q, a
            if best_a is not None:
                dr, dc = ARROWS[best_a]
                # For image coordinates: down is +r, right is +c
                V[r, c], U[r, c] = -dr, dc

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    X = np.arange(W)
    Y = np.arange(H)
    ax.quiver(X, Y[:, None], U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_title('Policy (Greedy Arrows)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def q_heatmap(Q: Dict[Tuple[Tuple[Any, ...], str], float],
              grid_shape: Tuple[int, int],
              slice_fn: Callable[[int, int], Tuple[Any, ...]],
              outpath: str) -> None:
    """Max-Q heatmap (movement actions only) for a canonical state slice and saves PNG."""
    Z = maxq_grid(Q, grid_shape, slice_fn)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Z, interpolation='nearest')
    ax.set_title('Max-Q Heatmap (moves only)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
