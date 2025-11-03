# PRANDOM, PGREEDY, PEXPLOIT

"""Policies for PD-World with required operator priority and tie-breaking.

Policies:
  - PRANDOM: If PICKUP/DROPOFF applicable, choose it; else uniform random among applicable moves.
  - PGREEDY: If PICKUP/DROPOFF applicable, choose it; else argmax-Q among applicable (tie-break random).
  - PEXPLOIT: If PICKUP/DROPOFF applicable, choose it; else with prob 0.8 choose greedy, with prob 0.2 choose a different applicable action uniformly.
"""
from typing import Dict, Tuple, List, Any, Optional
import random

PRIORITY_OPS = ('PICKUP','DROPOFF')

def _best_actions_by_q(Q: Dict[Tuple[tuple, str], float], state: tuple, actions: List[str]) -> List[str]:
    best_q = None
    best = []
    for a in actions:
        q = Q.get((state, a), 0.0)
        if best_q is None or q > best_q:
            best_q = q; best = [a]
        elif q == best_q:
            best.append(a)
    return best

def choose_action_PRANDOM(state: tuple, applicable: List[str], Q: Dict[Tuple[tuple,str], float], rng: random.Random) -> str:
    for p in PRIORITY_OPS:
        if p in applicable:
            return p
    return rng.choice(applicable)

def choose_action_PGREEDY(state: tuple, applicable: List[str], Q: Dict[Tuple[tuple,str], float], rng: random.Random) -> str:
    for p in PRIORITY_OPS:
        if p in applicable:
            return p
    best = _best_actions_by_q(Q, state, applicable)
    return rng.choice(best)

def choose_action_PEXPLOIT(state: tuple, applicable: List[str], Q: Dict[Tuple[tuple,str], float], rng: random.Random, greedy_prob: float = 0.8) -> str:
    for p in PRIORITY_OPS:
        if p in applicable:
            return p
    best = _best_actions_by_q(Q, state, applicable)
    if rng.random() < greedy_prob:
        return rng.choice(best)
    # explore among non-best if possible
    others = [a for a in applicable if a not in best]
    if others:
        return rng.choice(others)
    return rng.choice(best)

def choose_action(policy_name: str, state: tuple, applicable: List[str], Q: Dict[Tuple[tuple,str], float], rng: Optional[random.Random] = None) -> str:
    if not applicable:
        raise ValueError("No applicable actions available.")
    rng = rng or random
    name = policy_name.upper()
    if name == 'PRANDOM':
        return choose_action_PRANDOM(state, applicable, Q, rng)
    if name == 'PGREEDY':
        return choose_action_PGREEDY(state, applicable, Q, rng)
    if name == 'PEXPLOIT':
        return choose_action_PEXPLOIT(state, applicable, Q, rng)
    raise ValueError(f"Unknown policy: {policy_name}")
