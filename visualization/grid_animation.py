# visualization/grid_animation.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

def _parse_pair(v):
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    s = str(v).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    a, b = [p.strip() for p in s.split(",")]
    return int(a), int(b)

def _to_set(cells):
    """
    Accepts: None, list of (r,c), set of (r,c), dict like {(r,c): n}, or stringified tuples.
    Returns: set of (r,c)
    """
    if cells is None:
        return set()
    if isinstance(cells, dict):
        it = cells.keys()
    else:
        it = cells
    out = set()
    for v in it:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            out.add((int(v[0]), int(v[1])))
        else:
            out.add(_parse_pair(v))
    return out

def animate(
    df: pd.DataFrame,
    grid_shape,
    obstacles,
    outpath: str,
    *,
    pickups=None,
    drops=None,
    fps: int = 10,
    show_rewards: bool = True,
    trail_len: int = 30,
):
    """
    Create an MP4 (or GIF fallback) animation from a step log.

    Required df columns:
      global_step, episode_idx, agent, action, reward, pos_F, pos_M
      (pos_* may be '(r,c)' strings; they are parsed automatically)

    Optional visual cues:
      - pickups (list/dict/set of cells): green hatch
      - drops (list/dict/set of cells): blue hatch
      - obstacles (list/set of cells): gray fills
      - show per-step reward/action in HUD and flash marker for PICKUP/DROPOFF
      - short trails for both agents
    """
    # Normalize input
    df = df.copy()
    df["pos_F"] = df["pos_F"].apply(_parse_pair)
    df["pos_M"] = df["pos_M"].apply(_parse_pair)

    H, W = grid_shape
    obstacles = _to_set(obstacles)
    pickups = _to_set(pickups)
    drops = _to_set(drops)

    # Figure + axes
    fig, ax = plt.subplots(figsize=(W * 0.6, H * 0.6))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, linewidth=0.5, alpha=0.35, zorder=1)

    # Paint cells (push behind everything else)
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.3, zorder=2))
    for (r, c) in pickups:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2, zorder=3))
        ax.text(c - 0.35, r - 0.35, "P", fontsize=10, va="top", zorder=4)
    for (r, c) in drops:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2, zorder=3))
        ax.text(c - 0.35, r - 0.35, "D", fontsize=10, va="top", zorder=4)

    # Artists (set explicit zorders so they sit above patches)
    dotF, = ax.plot([], [], marker="o", markersize=10, label="F", zorder=10)
    dotM, = ax.plot([], [], marker="s", markersize=10, label="M", zorder=10)
    trailF, = ax.plot([], [], linewidth=2, alpha=0.6, zorder=9)
    trailM, = ax.plot([], [], linewidth=2, alpha=0.6, zorder=9)
    flashF, = ax.plot([], [], marker="o", markersize=0, alpha=0.0, zorder=11)
    flashM, = ax.plot([], [], marker="s", markersize=0, alpha=0.0, zorder=11)

    # HUD (put on top, give readable background, no clipping)
    hud = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=100, clip_on=False
    )

    ax.legend(loc="upper right", frameon=True)

    # Build frames
    frames = list(df.sort_values("global_step").itertuples(index=False))
    histF, histM = [], []

    def init():
        dotF.set_data([], [])
        dotM.set_data([], [])
        trailF.set_data([], [])
        trailM.set_data([], [])
        flashF.set_data([], [])
        flashM.set_data([], [])
        flashF.set_markersize(0); flashF.set_alpha(0.0)
        flashM.set_markersize(0); flashM.set_alpha(0.0)
        hud.set_text("")
        return dotF, dotM, trailF, trailM, flashF, flashM, hud

    def update(row):
        # positions
        rF, cF = row.pos_F
        rM, cM = row.pos_M
        dotF.set_data([cF], [rF])
        dotM.set_data([cM], [rM])

        # trails (truncate to trail_len)
        histF.append((rF, cF))
        histM.append((rM, cM))
        if len(histF) > trail_len: del histF[0]
        if len(histM) > trail_len: del histM[0]
        trailF.set_data([c for (_, c) in histF], [r for (r, _) in histF])
        trailM.set_data([c for (_, c) in histM], [r for (r, _) in histM])

        # Event flash for pickup/dropoff on the acting agent
        action = str(row.action).upper()
        reward = float(row.reward) if show_rewards else None
        agent = str(row.agent)

        # reset flashes
        flashF.set_markersize(0); flashF.set_alpha(0.0)
        flashM.set_markersize(0); flashM.set_alpha(0.0)

        if action in ("PICKUP", "DROPOFF"):
            if agent == "F":
                flashF.set_data([cF], [rF])
                flashF.set_markersize(18)     # bigger blip
                flashF.set_alpha(0.9)
            else:
                flashM.set_data([cM], [rM])
                flashM.set_markersize(18)
                flashM.set_alpha(0.9)

        # HUD
        if show_rewards:
            hud.set_text(
                f"t={row.global_step}  ep={row.episode_idx}  agent={agent}  "
                f"act={action}  r={reward:+.3f}"
            )
        else:
            hud.set_text(
                f"t={row.global_step}  ep={row.episode_idx}  agent={agent}  act={action}"
            )

        return dotF, dotM, trailF, trailM, flashF, flashM, hud

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=frames,
        interval=max(15, 1000 // max(fps, 1)), blit=True
    )

    # Prefer ffmpeg, fallback to GIF if not available
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata={"artist": "pd-world"})
        anim.save(outpath, writer=writer)
    except Exception:
        gif_out = outpath.rsplit(".", 1)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_out, writer=writer)
    finally:
        plt.close(fig)
