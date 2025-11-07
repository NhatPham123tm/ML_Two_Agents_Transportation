#!/usr/bin/env python3
"""
Warper to call experiment py files

Usage:
  # Run Experiment 1c (Q-learning with PEXPLOIT) for 2 runs
  python run_experiment.py 1c --runs 2

  # Run Experiment 2 (SARSA)
  python run_experiment.py 2

  # Run Experiment 3 (alpha sweep) with SARSA
  python run_experiment.py 3 --algo sarsa

  # Run Experiment 4 (adapting to change)
  python run_experiment.py 4
"""
import argparse
import os
import sys
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.getcwd())

# Import the core runner functions from each experiment script
from experiment.exp1 import run_single as run_exp1
from experiment.exp2 import run_single as run_exp2
from experiment.exp3 import main as run_exp3_main
from experiment.exp4 import main as run_exp4_main

def _normalize_seeds(user_list, fallback, runs):
    """
    Returns a list of length `runs`.
    - If user_list is empty: use fallback
    - If shorter than runs: repeat/cycle to fill
    - If longer: truncate
    """
    base = user_list if user_list else fallback
    if len(base) == runs:
        return base
    if len(base) > runs:
        return base[:runs]
    # len(base) < runs -> repeat
    out = []
    i = 0
    while len(out) < runs:
        out.append(base[i % len(base)])
        i += 1
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Master Control Panel for RL Experiments",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "experiment",
        type=str,
        choices=["1a", "1b", "1c", "2", "3", "4"],
        help="The experiment to run:\n"
             "  1a: Q-learning, PRANDOM after warmup\n"
             "  1b: Q-learning, PGREEDY after warmup\n"
             "  1c: Q-learning, PEXPLOIT after warmup\n"
             "  2:  SARSA, PEXPLOIT after warmup\n"
             "  3:  Alpha sweep (0.15, 0.45) for Q-learning or SARSA\n"
             "  4:  Adapt to pickup location change mid-run"
    )
    parser.add_argument("--runs", type=int, default=2, help="Number of independent runs (for Exp 1, 2, 3, 4)")
    parser.add_argument("--algo", choices=["qlearning", "sarsa"], default="qlearning",
                        help="Algorithm to use for Exp 3 and 4 (default: qlearning)")
    parser.add_argument("--outroot", type=str, default="artifacts", help="Root directory for output artifacts")
    parser.add_argument("--tag", type=str, default=None, help="Optional custom subfolder name for the run batch")
    parser.add_argument("--seedF", type=int, nargs="*", default=[123],
                    help="Seed(s) for agent F (default: predefined list)")
    parser.add_argument("--seedM", type=int, nargs="*", default=[321],
                    help="Seed(s) for agent M (default: predefined list)")
    args = parser.parse_args()

    # Default seeds
    fallback_F = [111, 222, 333, 444, 555]
    fallback_M = [999, 888, 777, 666, 555]

    seeds_F = _normalize_seeds(args.seedF, fallback_F, args.runs)
    seeds_M = _normalize_seeds(args.seedM, fallback_M, args.runs)

    if args.runs > len(seeds_F):
        raise ValueError(f"This script only has {len(seeds_F)} default seeds. Please reduce --runs or add more seeds.")

    print(f"--- Launching Experiment {args.experiment} ---")

    # === Dispatch to the correct experiment runner ===

    if args.experiment.startswith("1"):
        variant = args.experiment[-1]
        for i in range(args.runs):
            outdir = os.path.join(args.outroot, args.tag or f"exp1{variant}_run{i+1}")
            print(f"Run {i+1}/{args.runs} -> {outdir}")
            run_exp1(
                outdir=outdir,
                variant=variant,
                seedF=seeds_F[i],
                seedM=seeds_M[i],
            )

    elif args.experiment == "2":
        for i in range(args.runs):
            outdir = os.path.join(args.outroot, args.tag or f"exp2_run{i+1}")
            print(f"Run {i+1}/{args.runs} -> {outdir}")
            run_exp2(
                outdir=outdir,
                seedF=seeds_F[i],
                seedM=seeds_M[i],
            )


    elif args.experiment == "3":
        tag = args.tag or datetime.now().strftime("exp3-%Y%m%d-%H%M%S")
        sys.argv = [
            "experiment/exp3.py",
            "--algo", args.algo,
            "--runs", str(args.runs),
            "--outroot", args.outroot,
            "--tag", tag,
            # pass seeds as lists
            "--seedF", *map(str, seeds_F),
            "--seedM", *map(str, seeds_M),
        ]
        print(f"Starting Experiment 3 batch. See output in {os.path.join(args.outroot, tag)}")
        run_exp3_main()

    elif args.experiment == "4":
        tag = args.tag or datetime.now().strftime("exp4-%Y%m%d-%H%M%S")
        sys.argv = [
            "experiment/exp4.py",
            "--algo", args.algo,
            "--runs", str(args.runs),
            "--outroot", args.outroot,
            "--tag", tag,
            # pass seeds as lists
            "--seedF", *map(str, seeds_F),
            "--seedM", *map(str, seeds_M),
        ]
        print(f"Starting Experiment 4 batch. See output in {os.path.join(args.outroot, tag)}")
        run_exp4_main()

    print(f"--- Experiment {args.experiment} finished. ---")


if __name__ == "__main__":
    main()

