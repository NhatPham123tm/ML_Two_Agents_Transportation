# alpha sweep
#!/usr/bin/env python3
"""
Experiment 3 (PD-World, 2 agents, independent Q-tables)

Spec from assignment:
- Re-run either Exp 1.c (Q-learning) or Exp 2 (SARSA) with α in {0.15, 0.45} (γ unchanged).
- Same schedule: first 500 steps PRANDOM (warmup), then PEXPLOIT.
- Run each α twice with different seeds; log steps/episodes; plot; save Q-tables.
- Optional: generate visuals if visualization modules are available.

Usage:
  # Q-learning (default), 8000 steps, α in {0.15, 0.45}, two runs each
  python experiments/exp3.py

  # SARSA instead
  python experiments/exp3.py --algo sarsa

  # Customize steps/warmup/seeds and tag
  python experiments/exp3.py --steps 12000 --warmup 1000 --gamma 0.5 --runs 2 --tag exp3_demo
"""

import os, sys, json, argparse
from datetime import datetime
import pandas as pd

# Make project root importable when launched from elsewhere
sys.path.insert(0, os.getcwd())

from environment.world import PDWorld
from utils.logger import ExperimentLogger
from utils.plot import aggregate_episodes, plot_learning_curves, plot_coordination

# Agents
from agents.learner import QLearningAgent, SarsaAgent

# Optional visuals
MAKE_VIS = None
try:
    from visualization.visual_helper import make_visuals as MAKE_VIS  # (outdir, env_meta, QF, QM)
except Exception:
    MAKE_VIS = None

# --- World defaults (same as your Exp1/Exp2) ---
DEFAULT_OBSTACLES = {(2, 3), (2, 4), (3, 4)}
DEFAULT_PICKUPS   = {(0, 0): 2, (5, 0): 1}
DEFAULT_DROPS     = [(5, 7)]
DEFAULT_START_F   = (0, 1)
DEFAULT_START_M   = (5, 1)

def _mk_env(world_kwargs=None):
    wk = world_kwargs or {}
    return PDWorld(
        H=6, W=8,
        obstacles=wk.get("obstacles", DEFAULT_OBSTACLES),
        pickups=wk.get("pickups", DEFAULT_PICKUPS),
        drops=wk.get("drops", DEFAULT_DROPS),
        start_F=wk.get("start_F", DEFAULT_START_F),
        start_M=wk.get("start_M", DEFAULT_START_M),
    )

def _q_to_json(qdict):
    return {f"{list(k[0])}|{k[1]}": float(v) for k, v in qdict.items()}

def _run_single_alpha(outdir, algo, alpha, gamma, steps, warmup, seedF, seedM, world_kwargs=None):
    os.makedirs(outdir, exist_ok=True)
    env = _mk_env(world_kwargs)

    if algo == "qlearning":
        aF = QLearningAgent("F", alpha=alpha, gamma=gamma, seed=seedF)
        aM = QLearningAgent("M", alpha=alpha, gamma=gamma, seed=seedM)
    else:  # sarsa
        aF = SarsaAgent("F", alpha=alpha, gamma=gamma, seed=seedF, policy_name="PEXPLOIT")
        aM = SarsaAgent("M", alpha=alpha, gamma=gamma, seed=seedM, policy_name="PEXPLOIT")

    meta_world = {
        "H": env.H, "W": env.W,
        "obstacles": list(env.obstacles),
        "pickups": {str(k): v for k, v in env.pickups.items()},
        "drops": list(env.drops),
        "start_F": env.start_F,
        "start_M": env.start_M,
    }
    meta = {
        "experiment": "3",
        "algo": algo.upper(),
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

    log = ExperimentLogger(root=outdir, meta=meta); log.start_run()

    episode_idx = 0
    for t in range(steps):
        if env.is_terminal():
            env.reset()
            episode_idx += 1

        agent_id = "F" if (t % 2 == 0) else "M"
        policy = "PRANDOM" if t < warmup else "PEXPLOIT"

        if agent_id == "F":
            r, info, tr = aF.step(env, policy); action_taken = tr[1]
        else:
            r, info, tr = aM.step(env, policy); action_taken = tr[1]

        log.log_step(
            step=t, episode=episode_idx, agent=agent_id, action=action_taken,
            reward=r, pos_F=info["pos_F"], pos_M=info["pos_M"],
            carry_F=int(info["carry_F"]), carry_M=int(info["carry_M"]),
            terminal=int(info["terminal"]),
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

    # Save Q-tables
    with open(os.path.join(outdir, "qtable_F.json"), "w", encoding="utf-8") as f:
        json.dump(_q_to_json(aF.Q), f, indent=2)
    with open(os.path.join(outdir, "qtable_M.json"), "w", encoding="utf-8") as f:
        json.dump(_q_to_json(aM.Q), f, indent=2)

    # Optional visuals
    if MAKE_VIS:
        try:
            MAKE_VIS(outdir, meta_world, aF.Q, aM.Q)
        except Exception as e:
            with open(os.path.join(outdir, "viz", "_viz_error.txt"), "w", encoding="utf-8") as f:
                import traceback; f.write(str(e) + "\n\n" + traceback.format_exc())

    # Simple Q stats back
    nonzeroF = sum(1 for v in aF.Q.values() if abs(v) > 1e-9)
    nonzeroM = sum(1 for v in aM.Q.values() if abs(v) > 1e-9)
    maxF = max([0.0] + [float(v) for v in aF.Q.values()])
    maxM = max([0.0] + [float(v) for v in aM.Q.values()])
    return {"outdir": outdir, "nzF": nonzeroF, "nzM": nonzeroM, "maxF": maxF, "maxM": maxM}

def main():
    ap = argparse.ArgumentParser(description="Experiment 3: compare α=0.15 vs α=0.45 for Q-learning or SARSA")
    ap.add_argument("--algo", choices=["qlearning","sarsa"], default="qlearning",
                    help="Base algorithm to replicate (default: qlearning like Exp 1.c)")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.15, 0.45],
                    help="Learning rates to compare (default: 0.15, 0.45)")
    ap.add_argument("--runs", type=int, default=2, help="Independent runs per alpha")
    ap.add_argument("--seedF", type=int, nargs="*", default=[101, 202],
                    help="Seeds for agent F (≥ runs)")
    ap.add_argument("--seedM", type=int, nargs="*", default=[303, 404],
                    help="Seeds for agent M (≥ runs)")
    ap.add_argument("--outroot", type=str, default="artifacts",
                    help="Output root directory (default: artifacts)")
    ap.add_argument("--tag", type=str, default=None,
                    help="Subfolder name under outroot (default: exp3-<timestamp>)")
    args = ap.parse_args()

    if len(args.seedF) < args.runs or len(args.seedM) < args.runs:
        raise SystemExit("Provide at least --runs seeds for both --seedF and --seedM")

    stamp = args.tag or datetime.now().strftime("exp3-%Y%m%d-%H%M%S")
    batch_root = os.path.join(args.outroot, stamp)
    os.makedirs(batch_root, exist_ok=True)

    summary_rows = []
    for alpha in args.alphas:
        for i in range(args.runs):
            folder = os.path.join(batch_root, f"{args.algo}_a{str(alpha).replace('.','')}_run{i+1}")
            print(f"[exp3] α={alpha} run{i+1} → {folder}")
            stats = _run_single_alpha(
                outdir=folder,
                algo=args.algo,
                alpha=alpha,
                gamma=args.gamma,
                steps=args.steps,
                warmup=args.warmup,
                seedF=args.seedF[i],
                seedM=args.seedM[i],
            )
            # Per-run episode metrics
            ep = os.path.join(folder, "episodes.csv")
            row = {
                "folder": folder,
                "algo": args.algo,
                "alpha": alpha,
                "run": i+1,
                "nzF": stats["nzF"], "nzM": stats["nzM"],
                "maxF": stats["maxF"], "maxM": stats["maxM"],
                "episodes": 0, "mean_return": None, "best_return": None,
                "mean_steps": None, "avg_manhattan": None
            }
            if os.path.exists(ep):
                try:
                    df = pd.read_csv(ep)
                    row.update({
                        "episodes": int(len(df)),
                        "mean_return": float(df["return_sum"].mean()) if len(df) else None,
                        "best_return": float(df["return_sum"].max()) if len(df) else None,
                        "mean_steps": float(df["steps"].mean()) if len(df) else None,
                        "avg_manhattan": float(df["avg_manhattan_FM"].mean()) if len(df) else None,
                    })
                except Exception:
                    pass
            summary_rows.append(row)

    # Write batch summary as Markdown + CSV
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(os.path.join(batch_root, "summary.csv"), index=False)

        cols = ["folder","algo","alpha","run","episodes","mean_return","best_return","mean_steps","avg_manhattan","nzF","nzM","maxF","maxM"]
        md = ["# Experiment 3 Summary", "",
              f"- Algo: {args.algo.upper()}",
              f"- Steps: {args.steps} | Warmup: {args.warmup} | γ={args.gamma}",
              f"- Alphas: {args.alphas}",
              f"- Seeds F: {args.seedF} | Seeds M: {args.seedM}",
              "", "## Per-run metrics", "",
              sdf[cols].to_markdown(index=False), ""]
        with open(os.path.join(batch_root, "summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    print(f"[exp3] Done. See {batch_root}")

if __name__ == "__main__":
    main()
