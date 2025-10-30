import os, sys, time
import pygame

# Make sure we can import viz/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization.live_pygame_viewer import run as run_view

def demo_step_log():
    """
    Create a short fake step log:
    - Columns expected: pos_F, pos_M (row, col), agent, action, episode_idx, global_step
    """
    steps = []
    posF = [0, 0]; posM = [4, 0]
    for t in range(60):
        agent = 'F' if t % 2 == 0 else 'M'
        if agent == 'F':
            # Move F down first, then right
            if t // 2 < 6:
                posF = [posF[0] + 1, posF[1]]; action = 'DOWN'
            else:
                posF = [posF[0], posF[1] + 1]; action = 'RIGHT'
        else:
            # M just moves right slowly
            if (t // 2) % 2 == 0:
                posM = [posM[0], posM[1] + 1]; action = 'RIGHT'
            else:
                action = 'STAY'
        steps.append({
            "global_step": t,
            "episode_idx": 0,
            "agent": agent,
            "action": action,
            "pos_F": tuple(posF),
            "pos_M": tuple(posM),
        })
    return steps

if __name__ == "__main__":
    # Grid & obstacles for the viewer
    GRID_SHAPE = (6, 10)            # rows, cols
    OBSTACLES  = [(2, 3), (2, 4), (3, 4)]

    # Build a fake log (replace with your real step log list or CSV reader)
    step_log = demo_step_log()

    # Run the live viewer (Esc or close window to exit)
    run_view(step_log, GRID_SHAPE, OBSTACLES, cell_px=64)
