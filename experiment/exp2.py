# SARSA config
#!/usr/bin/env python3
"""
Experiment 2 (SARSA, 2 agents, independent Q-tables)

Spec:
- α = 0.3, γ = 0.5, total steps = 8000
- First 500 steps: PRANDOM (warmup)
- Remaining steps: PEXPLOIT (on-policy SARSA)
- Run twice with different seeds
- Log steps/episodes + meta; aggregate + plot; save final Q-tables
- (Optional) Visuals if a helper is available: visualization/visual_helper.py::make_visuals
"""

import os, sys, json, argparse
from datetime import datetime

# repo-local imports
sys.path.append(".")
from environment.world import PDWorld
from agents.learner import SarsaAgent
from utils.logger import ExperimentLogger
from utils.plot import aggregate_episodes, plot_learning_curves, plot_coordination

# Optional visuals (safe if missing)
try:
    from visualization.visual_helper import make_visuals  # expects (outdir, env_meta_dict, q_F, q_M)
    VIZ_OK = True
except Exception:
    VIZ_OK = False

# ----- World defaults (same as Exp1) -----
DEFAULT_OBSTACLES = {(2, 3), (2, 4), (3, 4)}
DEFAULT_PICKUPS   = {(0, 0): 2, (5, 0): 1}
DEFAULT_DROPS     = [(5, 7)]
DEFAULT_START_F   = (0, 1)
DEFAULT_START_M   = (5, 1)


def run_single(
    outdir: str,
    *,
    steps: int = 8000,
    warmup: int = 500,
    alpha: float = 0.3,
    gamma: float = 0.5,
    seedF: int = 111,
    seedM: int = 333,
    world_kwargs: dict | None = None,
    print_every: int = 500,
    verbose: bool = True,
    animate = True,
):
    os.makedirs(outdir, exist_ok=True)

    world_kwargs = world_kwargs or {}
    env = PDWorld(
        H=6, W=8,
        obstacles=world_kwargs.get("obstacles", DEFAULT_OBSTACLES),
        pickups=world_kwargs.get("pickups", DEFAULT_PICKUPS),
        drops=world_kwargs.get("drops", DEFAULT_DROPS),
        start_F=world_kwargs.get("start_F", DEFAULT_START_F),
        start_M=world_kwargs.get("start_M", DEFAULT_START_M),
    )

    # On-policy SARSA agents (default policy PEXPLOIT; we override to PRANDOM during warmup)
    aF = SarsaAgent("F", alpha=alpha, gamma=gamma, seed=seedF, policy_name="PEXPLOIT")
    aM = SarsaAgent("M", alpha=alpha, gamma=gamma, seed=seedM, policy_name="PEXPLOIT")

    # ---- helpers & flags ----
    def q_to_json(qdict):
        return {f"{list(k[0])}|{k[1]}": float(v) for k, v in qdict.items()}

    def save_q_snapshot(tag: str):
        """Save a snapshot of both agents' Q-tables with a tag in the filename."""
        snap_dir = os.path.join(outdir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)

        with open(os.path.join(snap_dir, f"qtable_F_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(q_to_json(aF.Q), f, indent=2)
        with open(os.path.join(snap_dir, f"qtable_M_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(q_to_json(aM.Q), f, indent=2)

    saved_first_dropoff = False
    saved_terminal_snapshot = False

    meta_world = {
        "H": env.H, "W": env.W,
        "obstacles": list(env.obstacles),
        "pickups": {str(k): v for k, v in env.pickups.items()},
        "drops": list(env.drops),
        "start_F": env.start_F,
        "start_M": env.start_M,
    }
    meta = {
        "experiment": "2",
        "algo": "SARSA",
        "alpha": alpha,
        "gamma": gamma,
        "steps_total": steps,
        "warmup_steps": warmup,
        "policy_warmup": "PRANDOM",
        "policy_after": "PEXPLOIT",
        "seeds": {"F": seedF, "M": seedM},
        "world": meta_world,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    log = ExperimentLogger(root=outdir, meta=meta)
    log.start_run()

    episode_idx = 0
    for t in range(steps):

        # announce warmup end exactly when it switches
        if verbose and t == warmup:
            print("[INFO] Warmup finished at step "
                  f"{t}. Switching to PEXPLOIT.", flush=True)

        if env.is_terminal():
            env.reset()
            episode_idx += 1
            if verbose:
                print(f"[EP] Ended episode {episode_idx-1}; "
                      f"reset -> episode {episode_idx}", flush=True)

        agent_id = "F" if (t % 2 == 0) else "M"
        policy = "PRANDOM" if t < warmup else "PEXPLOIT"

        if agent_id == "F":
            r, info, tr = aF.step(env, policy); action_taken = tr[1]
        else:
            r, info, tr = aM.step(env, policy); action_taken = tr[1]

        # (a) snapshot: first drop-off filled (5th block at first drop)
        if (not saved_first_dropoff) and info.get("first_drop_filled", False):
            save_q_snapshot(f"first_dropoff_filled_step{t}_ep{episode_idx}")
            saved_first_dropoff = True

        log.log_step(
            step=t,
            episode=episode_idx,
            agent=agent_id,
            action=action_taken,
            reward=r,
            pos_F=info["pos_F"], pos_M=info["pos_M"],
            carry_F=int(info["carry_F"]), carry_M=int(info["carry_M"]),
            terminal=int(info["terminal"]),
        )

        # (b) snapshot: first time we hit a terminal state
        if info["terminal"] and (not saved_terminal_snapshot):
            save_q_snapshot(f"terminal_ep{episode_idx}_step{t}")
            saved_terminal_snapshot = True

        # heartbeat: print every N steps and on last step
        if verbose and (t % print_every == 0 or t == steps - 1):
            disp = t + 1  # 1-based for display
            print(
                f"STEP {disp}/{steps} | ep={episode_idx} | agent={agent_id} "
                f"| pol={policy} | act={action_taken} | r={r:.3f} "
                f"| F{info['pos_F']} M{info['pos_M']} | term={int(info['terminal'])}",
                flush=True
            )

        if info["terminal"]:
            log.end_episode(episode=episode_idx)

    log.close()

    # Aggregate + plots
    steps_csv = os.path.join(outdir, "steps.csv")
    ep_df = aggregate_episodes(steps_csv)
    ep_df.to_csv(os.path.join(outdir, "episodes_from_steps.csv"), index=False)
    plot_learning_curves(ep_df, os.path.join(outdir, "learning_curve.png"))
    plot_coordination(ep_df,   os.path.join(outdir, "coordination.png"))

    # (c) Save final Q-tables (whole experiment)
    with open(os.path.join(outdir, "qtable_F.json"), "w", encoding="utf-8") as f:
        json.dump(q_to_json(aF.Q), f, indent=2)
    with open(os.path.join(outdir, "qtable_M.json"), "w", encoding="utf-8") as f:
        json.dump(q_to_json(aM.Q), f, indent=2)
    
    # Optional visuals (heatmaps, quiver, overlays, animation)
    if VIZ_OK and animate:
        print("Generating visual output (May take a while) ...", flush=True)
        make_visuals(outdir, meta_world, aF.Q, aM.Q)

    return {
        "outdir": outdir,
        "steps_csv": steps_csv,
        "episodes_csv": os.path.join(outdir, "episodes.csv"),
        "q_F": os.path.join(outdir, "qtable_F.json"),
        "q_M": os.path.join(outdir, "qtable_M.json"),
    }



def main():
    print("Bulding PDWorld ...")
    ap = argparse.ArgumentParser(description="Experiment 2 (SARSA: PRANDOM warmup → PEXPLOIT)")
    ap.add_argument("--runs", type=int, default=2, help="number of independent runs (default 2)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seedF", type=int, nargs="*", default=[135, 246],
                    help="Seeds for agent F (≥ runs)")
    ap.add_argument("--seedM", type=int, nargs="*", default=[864, 975],
                    help="Seeds for agent M (≥ runs)")
    ap.add_argument("--outroot", type=str, default="artifacts",
                    help="Output root folder")
    ap.add_argument("--print-every", type=int, default=500,
                    help="print progress every N steps")
    ap.add_argument("--verbose", action="store_true",
                    help="enable console progress prints")
    args = ap.parse_args()

    os.makedirs(args.outroot, exist_ok=True)
    print("Running Agents ...")
    for i in range(args.runs):
        outdir = os.path.join(args.outroot, f"exp2_run{i+1}")
        res = run_single(
            outdir=outdir,
            steps=args.steps,
            warmup=args.warmup,
            alpha=args.alpha,
            gamma=args.gamma,
            seedF=args.seedF[i],
            seedM=args.seedM[i],
            print_every=args.print_every,
            verbose=args.verbose,
        )
        print(f"[OK] Wrote: {res['outdir']}")


if __name__ == "__main__":
    main()
