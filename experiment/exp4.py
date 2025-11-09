# pickup change mid‑run
#!/usr/bin/env python3
"""
Experiment 4 (PD-World, 2 agents, independent Q-tables)

Spec:
- Use alpha=0.3, gamma=0.5 with either Q-learning (default) or SARSA (--algo sarsa)
- Warmup: first 500 steps PRANDOM
- Then PEXPLOIT until the 3rd terminal episode is reached
- Immediately change pickup locations to (1,2) and (4,5); keep dropoffs and Q-tables as-is
- Continue PEXPLOIT until the 6th terminal episode
- Run each configuration twice with different seeds
- Log steps/episodes; aggregate & plot; save final Q-tables; optional visuals

Usage:
  python experiment/exp4.py
  python experiment/exp4.py --algo sarsa --runs 2 --steps_cap 40000
"""

import os, sys, json, argparse
from datetime import datetime
import pandas as pd

# ensure project root on path
sys.path.insert(0, os.getcwd())

from environment.world import PDWorld
from utils.logger import ExperimentLogger
from utils.plot import aggregate_episodes, plot_learning_curves, plot_coordination
from agents.learner import QLearningAgent, SarsaAgent
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
# Optional visuals
MAKE_VIS = None
try:
    from visualization.visual_helper import make_visuals as MAKE_VIS  # (outdir, env_meta, QF, QM)
except Exception:
    MAKE_VIS = None

# ---- World defaults (match your other experiments) ----
DEFAULT_OBSTACLES = {(2, 3), (2, 4), (3, 4)}
DEFAULT_PICKUPS   = {(0, 0): 2, (5, 0): 1}
DEFAULT_DROPS     = [(5, 7)]
DEFAULT_START_F   = (0, 1)
DEFAULT_START_M   = (5, 1)

# The "changed" pickups required by the spec:
PICKUPS_AFTER_CHANGE = {(1, 2): 1, (4, 5): 1}


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


def _apply_pickup_change(env: PDWorld, new_pickups: dict):
    """
    Change the environment's pickup locations during a run.
    Prefer a dedicated method if your PDWorld exposes one; otherwise set attribute and
    refresh any cached masks if available.
    """
    if hasattr(env, "set_pickups") and callable(env.set_pickups):
        env.set_pickups(new_pickups, reset_blocks=True)
    else:
        env.pickups = dict(new_pickups)

    for fn in ("rebuild_task_mask", "refresh_pickup_cache", "recompute_masks"):
        if hasattr(env, fn) and callable(getattr(env, fn)):
            try:
                getattr(env, fn)()
            except Exception:
                pass  # best effort


def _q_to_json(qdict):
    return {f"{list(k[0])}|{k[1]}": float(v) for k, v in qdict.items()}


def run_single(
    outdir: str,
    *,
    algo: str = "qlearning",
    alpha: float = 0.3,
    gamma: float = 0.5,
    warmup: int = 500,
    seedF: int = 111,
    seedM: int = 333,
    steps_cap: int = 80000,    
    world_kwargs: dict | None = None,
    print_every: int = 500,
    verbose: bool = True,
    animate = True,   
):
    """
    Run until 6 terminal episodes; switch pickups right after the 3rd terminal.
    """
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
        "experiment": "4",
        "algo": algo.upper(),
        "alpha": alpha,
        "gamma": gamma,
        "warmup_steps": warmup,
        "policy_warmup": "PRANDOM",
        "policy_after": "PEXPLOIT",
        "seeds": {"F": seedF, "M": seedM},
        "world": meta_world,
        "pickup_change_at_episode": 3,
        "pickup_change_to": list(PICKUPS_AFTER_CHANGE.keys()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    log = ExperimentLogger(root=outdir, meta=meta)
    log.start_run()

    episode_idx = 0
    terminals_reached = 0
    pickups_changed = False

    t = 0
    while terminals_reached < 6 and t < steps_cap:
        # announce warmup end once
        if verbose and t == warmup:
            print(f"[INFO] Warmup finished at step {t}. Switching to PEXPLOIT.", flush=True)

        agent_id = "F" if (t % 2 == 0) else "M"
        policy = "PRANDOM" if t < warmup else "PEXPLOIT"

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

        # heartbeat: print at step 0, every N steps, and you can also add "or t == steps_cap-1"
        if verbose and (t % print_every == 0):
            print(
                f"STEP {t}/{steps_cap} | ep={episode_idx} | agent={agent_id} "
                f"| pol={policy} | act={action_taken} | r={r:.3f} "
                f"| F{info['pos_F']} M{info['pos_M']} | term={int(info['terminal'])}",
                flush=True
            )

        t += 1

        if info["terminal"]:
            terminals_reached += 1
            log.end_episode(episode=episode_idx)
            if verbose:
                print(f"[TERM] episode {episode_idx} ended "
                      f"(terminals={terminals_reached}/6, steps={t})", flush=True)

            if (terminals_reached == 3) and (not pickups_changed):
                _apply_pickup_change(env, PICKUPS_AFTER_CHANGE)
                env.reset()
                pickups_changed = True

                meta["world"]["pickups_changed_at_episode"] = 3
                meta["world"]["pickups_list"] = [[r0, c0, n0] for (r0, c0), n0 in env.pickups.items()]
                log.update_meta(meta)

                log.log_event(
                    step=t,
                    episode=episode_idx + 1,
                    tag="PICKUPS_CHANGED",
                    payload={"to": meta["world"]["pickups_list"]},
                    pos_F=env.pos_F, pos_M=env.pos_M,
                    carry_F=int(env.carry_F), carry_M=int(env.carry_M),
                )

                if verbose:
                    print(f"[EVENT] PICKUPS_CHANGED at ep=3 → {meta['world']['pickups_list']}", flush=True)
            else:
                env.reset()

            episode_idx += 1

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

    # Optional visuals (heatmaps, quiver, overlays, animation)
    if MAKE_VIS and animate:
        try:
            print("Generating visual output ...", flush=True)
            MAKE_VIS(outdir, meta["world"], aF.Q, aM.Q)
        except Exception as e:
            viz_dir = os.path.join(outdir, "viz"); os.makedirs(viz_dir, exist_ok=True)
            import traceback
            with open(os.path.join(viz_dir, "_viz_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e) + "\n\n" + traceback.format_exc())

    # simple stats back
    nonzeroF = sum(1 for v in aF.Q.values() if abs(v) > 1e-9)
    nonzeroM = sum(1 for v in aM.Q.values() if abs(v) > 1e-9)

    if verbose:
        print(f"[DONE] steps={t} | terminals={terminals_reached} | pickups_changed={pickups_changed}", flush=True)
    return {
        "outdir": outdir,
        "episodes_written": len(ep_df) if isinstance(ep_df, pd.DataFrame) else 0,
        "terminals_reached": terminals_reached,
        "pickups_changed": pickups_changed,
        "nzF": nonzeroF, "nzM": nonzeroM
    }


def main():
    print("Bulding PDWorld ...", flush=True)
    ap = argparse.ArgumentParser(description="Experiment 4: adapt to changed pickup locations after 3 terminals")
    ap.add_argument("--algo", choices=["qlearning","sarsa"], default="qlearning",
                    help="Base algorithm")
    ap.add_argument("--runs", type=int, default=2, help="independent runs")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seedF", type=int, nargs="*", default=[111, 211],
                    help="Seeds for agent F (≥ runs)")
    ap.add_argument("--seedM", type=int, nargs="*", default=[333, 433],
                    help="Seeds for agent M (≥ runs)")
    ap.add_argument("--steps_cap", type=int, default=30000,
                    help="Global step safety cap (stop even if 6 terminals not reached)")
    ap.add_argument("--outroot", type=str, default="artifacts")
    ap.add_argument("--tag", type=str, default=None, help="Batch subfolder name")
    ap.add_argument("--print-every", type=int, default=500,help="print progress every N steps")
    ap.add_argument("--verbose", action="store_true",help="enable console progress prints")
    ap.add_argument("--animate", dest="animate", action="store_true",help="Generate visual/animation output")
    ap.add_argument("--no-animate", dest="animate", action="store_false",help="Disable animation output")
    ap.set_defaults(animate=True)
    args = ap.parse_args()

    if len(args.seedF) < args.runs or len(args.seedM) < args.runs:
        raise SystemExit("Provide at least --runs seeds for both --seedF and --seedM")

    stamp = args.tag or datetime.now().strftime("exp4-%Y%m%d-%H%M%S")
    batch_root = os.path.join(args.outroot, stamp)
    os.makedirs(batch_root, exist_ok=True)

    rows = []
    total_runs = args.runs
    print("Running Agents ...", flush=True)
    for i in range(args.runs):
        k = i + 1
        outdir = os.path.join(batch_root, f"{args.algo}_run{i+1}")
        print(f"[exp4] ({k}/{total_runs}) {args.algo} run{k} → {outdir}", flush=True)
        print(f"PROGRESS RUN {k}/{total_runs}", flush=True)
        res = run_single(
            outdir=outdir,
            algo=args.algo,
            alpha=args.alpha,
            gamma=args.gamma,
            warmup=args.warmup,
            seedF=args.seedF[i],
            seedM=args.seedM[i],
            steps_cap=args.steps_cap,
            print_every=args.print_every,
            verbose=True,
            animate=args.animate,
        )
        rows.append(res)

    # Write quick summary
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(batch_root, "summary.csv"), index=False)
        md_cols = ["outdir","episodes_written","terminals_reached","pickups_changed","nzF","nzM"]
        md = ["# Experiment 4 Summary", "", df[md_cols].to_markdown(index=False), ""]
        with open(os.path.join(batch_root, "summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    print(f"[exp4] Done. See {batch_root}")


if __name__ == "__main__":
    main()
