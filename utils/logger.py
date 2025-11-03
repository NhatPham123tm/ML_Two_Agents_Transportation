
"""Lightweight experiment logger for PD-World.

- Step log (CSV): one row per environment step (agent alternation)
- Episode summary (CSV): one row per completed episode
- Run meta (JSON): configuration and notes

Usage:
    from utils.logger import ExperimentLogger
    log = ExperimentLogger(root='artifacts/run1', meta={'exp':'1c','alpha':0.3,'gamma':0.5})
    log.start_run()
    for t in range(T):    # per-step
        log.log_step(step=t, episode=ep, agent=agent_id, action=act, reward=r,
                     pos_F=posF, pos_M=posM, carry_F=carryF, carry_M=carryM, terminal=term)
        if term:
            log.end_episode(episode=ep)
    log.close()
"""
from __future__ import annotations
import csv, json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

@dataclass
class EpisodeStats:
    episode_idx: int
    return_sum: float = 0.0
    steps: int = 0
    conflicts: int = 0
    manhattan_sum: int = 0

    def update(self, reward: float, pos_F: Tuple[int,int], pos_M: Tuple[int,int]) -> None:
        self.return_sum += float(reward)
        self.steps += 1
        if pos_F == pos_M:
            self.conflicts += 1
        self.manhattan_sum += manhattan(pos_F, pos_M)

    def to_row(self) -> Dict[str, Any]:
        avg_manhattan = (self.manhattan_sum / self.steps) if self.steps else 0.0
        return {
            "episode_idx": self.episode_idx,
            "return_sum": round(self.return_sum, 6),
            "steps": self.steps,
            "conflicts": self.conflicts,
            "avg_manhattan_FM": round(avg_manhattan, 6),
        }

class ExperimentLogger:
    def __init__(self, root: str, meta: Optional[Dict[str, Any]] = None):
        self.root = Path(root)
        self.meta = meta or {}
        self.root.mkdir(parents=True, exist_ok=True)
        # file paths
        self.f_steps = self.root / "steps.csv"
        self.f_eps = self.root / "episodes.csv"
        self.f_meta = self.root / "meta.json"
        # open writers
        self._steps_fp = open(self.f_steps, "w", newline="", encoding="utf-8")
        self._eps_fp = open(self.f_eps, "w", newline="", encoding="utf-8")
        self.steps_writer = csv.DictWriter(self._steps_fp, fieldnames=[
            "global_step","episode_idx","agent","action","reward",
            "pos_F","pos_M","carry_F","carry_M","terminal"
        ])
        self.steps_writer.writeheader()
        self.eps_writer = csv.DictWriter(self._eps_fp, fieldnames=[
            "episode_idx","return_sum","steps","conflicts","avg_manhattan_FM"
        ])
        self.eps_writer.writeheader()
        # state
        self._ep_stats: Dict[int, EpisodeStats] = {}
        self._run_started = False

    def start_run(self) -> None:
        if self._run_started: 
            return
        with open(self.f_meta, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        self._run_started = True

    def log_step(self, *, step: int, episode: int, agent: str, action: str, reward: float,
                 pos_F: tuple, pos_M: tuple, carry_F: int, carry_M: int, terminal: int) -> None:
        # write step row
        self.steps_writer.writerow({
            "global_step": step,
            "episode_idx": episode,
            "agent": agent,
            "action": action,
            "reward": float(reward),
            "pos_F": str(tuple(pos_F)),
            "pos_M": str(tuple(pos_M)),
            "carry_F": int(carry_F),
            "carry_M": int(carry_M),
            "terminal": int(terminal),
        })
        # accumulate episode stats
        stats = self._ep_stats.get(episode)
        if stats is None:
            stats = self._ep_stats[episode] = EpisodeStats(episode_idx=episode)
        stats.update(reward, tuple(pos_F), tuple(pos_M))

    def end_episode(self, *, episode: int) -> None:
        stats = self._ep_stats.get(episode)
        if not stats:
            return
        self.eps_writer.writerow(stats.to_row())
        # keep stats but do not reset; episodes are unique keys

    def close(self) -> None:
        try:
            self._steps_fp.flush(); self._eps_fp.flush()
        finally:
            self._steps_fp.close()
            self._eps_fp.close()
