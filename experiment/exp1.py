# 1a/1b/1c configs
#!/usr/bin/env python3
"""
Experiment 1 driver (PD-World, 2 agents, independent Q-tables)

Spec:
- Q-learning, alpha=0.3, gamma=0.5, 8000 steps total
- First 500 steps PRANDOM (warmup)
- Then:
  a) continue PRANDOM
  b) switch to PGREEDY
  c) switch to PEXPLOIT
- Run each variant twice with different seeds (F and M can differ)
- Log steps + episodes; aggregate + plot; save final Q-tables
"""

import os, sys, json, csv, argparse, pathlib, random
from datetime import datetime

# Repo-local imports (assume you run from project root)
sys.path.append(".")
from environment.world import PDWorld
from agents.learner import QLearningAgent
from utils.logger import ExperimentLogger
from utils.plot import aggregate_episodes, plot_learning_curves, plot_coordination

try:
    from visualization.visual_helper import make_visuals  # expects (outdir, env_meta_dict, q_F, q_M)
    VIZ_OK = True
except Exception:
    VIZ_OK = False

DEFAULT_OBSTACLES = {(2, 3), (2, 4), (3, 4)}
DEFAULT_PICKUPS   = {(0, 0): 2, (5, 0): 1}
DEFAULT_DROPS     = [(5, 7)]
DEFAULT_START_F   = (0, 1)
DEFAULT_START_M   = (5, 1)

def policy_after_warmup(variant: str) -> str:
    v = variant.lower()
    if v == "a": return "PRANDOM"
    if v == "b": return "PGREEDY"
    if v == "c": return "PEXPLOIT"
    raise ValueError("variant must be a/b/c")

def run_single(
    outdir: str,
    variant: str,
    steps: int = 8000,
    warmup: int = 500,
    alpha: float = 0.3,
    gamma: float = 0.5,
    seedF: int = 123,
    seedM: int = 321,
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

    aF = QLearningAgent("F", alpha=alpha, gamma=gamma, seed=seedF)
    aM = QLearningAgent("M", alpha=alpha, gamma=gamma, seed=seedM)

    after = policy_after_warmup(variant)
    meta = {
        "experiment": "1" + variant.lower(),
        "algo": "Q-learning",
        "alpha": alpha,
        "gamma": gamma,
        "steps_total": steps,
        "warmup_steps": warmup,
        "policy_warmup": "PRANDOM",
        "policy_after": after,
        "seeds": {"F": seedF, "M": seedM},
        "world": {
            "H": env.H, "W": env.W,
            "obstacles": list(env.obstacles),
            "pickups": {str(k): v for k, v in env.pickups.items()},
            "drops": list(env.drops),
            "start_F": env.start_F,
            "start_M": env.start_M,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    log = ExperimentLogger(root=outdir, meta=meta)
    log.start_run()

    episode_idx = 0
    for t in range(steps):

        # Mark end of warmup right when we cross it
        if verbose and t == warmup:
            print(f"[INFO] Warmup finished at step {t}. Switching to {policy_after_warmup(variant)}.", flush=True)

        if env.is_terminal():
            env.reset()
            episode_idx += 1
            if verbose:
                print(f"[EP] Ended episode {episode_idx-1}; reset -> episode {episode_idx}", flush=True)

        agent_id = "F" if (t % 2 == 0) else "M"
        policy = "PRANDOM" if t < warmup else policy_after_warmup(variant)

        if agent_id == "F":
            r, info, tr = aF.step(env, policy); action_taken = tr[1]
        else:
            r, info, tr = aM.step(env, policy); action_taken = tr[1]

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

        # Heartbeat print
        if verbose and (t % print_every == 0 or t == steps - 1):
            # 1-based step for display
            disp = t + 1
            print(
                f"STEP {disp}/{steps} | ep={episode_idx} | agent={agent_id} | pol={policy} "
                f"| act={action_taken} | r={r:.3f} | F{info['pos_F']} M{info['pos_M']} "
                f"| term={int(info['terminal'])}",
                flush=True
            )

        if info["terminal"]:
            log.end_episode(episode=episode_idx)

    log.close()

    # Aggregate & plots
    steps_csv = os.path.join(outdir, "steps.csv")
    ep_csv    = os.path.join(outdir, "episodes.csv")  # already written by logger
    ep_df = aggregate_episodes(steps_csv)
    agg_csv = os.path.join(outdir, "episodes_from_steps.csv")
    ep_df.to_csv(agg_csv, index=False)

    plot_learning_curves(ep_df, os.path.join(outdir, "learning_curve.png"))
    plot_coordination(ep_df,   os.path.join(outdir, "coordination.png"))

    # Save Q-tables
    def q_to_json(qdict):
        return {f"{list(k[0])}|{k[1]}": float(v) for k, v in qdict.items()}

    with open(os.path.join(outdir, "qtable_F.json"), "w", encoding="utf-8") as f:
        json.dump(q_to_json(aF.Q), f, indent=2)
    with open(os.path.join(outdir, "qtable_M.json"), "w", encoding="utf-8") as f:
        json.dump(q_to_json(aM.Q), f, indent=2)

    def q_stats(Q):
        nonzero = sum(1 for v in Q.values() if abs(v) > 1e-9)
        maxq = max([0.0]+[float(v) for v in Q.values()])
        return nonzero, maxq

    # Make visuals (if viz modules available)
    if VIZ_OK and animate:
        print("Generating visual output (May take a while) ...", flush=True)
        make_visuals(
            outdir=outdir,
            env_spec=meta["world"],   # we already built this dict above
            QF=aF.Q,
            QM=aM.Q,
        )

    return {
        "outdir": outdir,
        "steps_csv": steps_csv,
        "episodes_csv": ep_csv,
        "episodes_from_steps_csv": agg_csv,
        "q_F": os.path.join(outdir, "qtable_F.json"),
        "q_M": os.path.join(outdir, "qtable_M.json"),
    }

def main():
    print("Bulding PDWorld ...")
    parser = argparse.ArgumentParser(description="Experiment 1 (Q-learning, warmup PRANDOM, variants a/b/c)")
    parser.add_argument("--variant", choices=["a","b","c"], required=True,
                        help="a: PRANDOM→PRANDOM, b: PRANDOM→PGREEDY, c: PRANDOM→PEXPLOIT")
    parser.add_argument("--runs", type=int, default=2, help="number of independent runs (default 2)")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seedF", type=int, nargs="*", default=[123, 223],
                        help="Seeds for agent F (provide >= runs)")
    parser.add_argument("--seedM", type=int, nargs="*", default=[321, 421],
                        help="Seeds for agent M (provide >= runs)")
    parser.add_argument("--outroot", type=str, default="artifacts",
                        help="Where to write run folders (default artifacts)")
    parser.add_argument("--print-every", type=int, default=500, help="print a heartbeat every N steps")
    parser.add_argument("--verbose", action="store_true", help="enable console progress prints")
    args = parser.parse_args()
    
    if len(args.seedF) < args.runs or len(args.seedM) < args.runs:
        raise SystemExit("Provide at least --runs seeds for both --seedF and --seedM")
    
    os.makedirs(args.outroot, exist_ok=True)
    print("Running Agents ...")
    
    for i in range(args.runs):
        outdir = os.path.join(args.outroot, f"exp1{args.variant}_run{i+1}")
        res = run_single(
            outdir=outdir,
            variant=args.variant,
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
