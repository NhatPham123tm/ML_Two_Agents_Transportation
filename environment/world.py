# grid, pickups/dropoffs, obstacles, reset, step()
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from . import rules

Coord = Tuple[int,int]

@dataclass
class Rewards:
    # softer living cost; separate idle penalty
    step_cost: float = -0.03
    idle_penalty: float = -0.02
    pickup_reward: float = 2.0
    drop_reward: float = 8.0
    finish_reward: float = 5.0
    invalid_penalty: float = -0.3
    collision_penalty: float = -1.0

    # Potential-based shaping (policy invariant):
    # reward += shaping_coef * (gamma_shaping * Phi(s') - Phi(s))
    shaping_coef: float = 0.2
    gamma_shaping: float = 0.95

    def as_dict(self):
        return dict(step_cost=self.step_cost,
                    idle_penalty=self.idle_penalty,
                    pickup_reward=self.pickup_reward,
                    drop_reward=self.drop_reward,
                    finish_reward=self.finish_reward,
                    invalid_penalty=self.invalid_penalty,
                    collision_penalty=self.collision_penalty)

@dataclass
class PDWorld:
    H: int
    W: int
    obstacles: set = field(default_factory=set)             # set[(r,c)]
    pickups: Dict[Coord, int] = field(default_factory=dict) # blocks per pickup cell
    drops: List[Coord] = field(default_factory=list)        # drop cells
    start_F: Coord = (0,0)
    start_M: Coord = (0,1)
    rewards: Rewards = field(default_factory=Rewards)
    max_steps_per_episode: int = 500

    # runtime state
    pos_F: Coord = field(init=False)
    pos_M: Coord = field(init=False)
    carry_F: bool = field(init=False, default=False)
    carry_M: bool = field(init=False, default=False)
    _blocks: Dict[Coord, int] = field(init=False)
    steps_in_episode: int = field(init=False, default=0)
    _viz_goal: Optional[Coord] = field(init=False, default=None)

    def __post_init__(self):
        self.reset()

    # ---- Episode control ----
    def reset(self) -> None:
        self.pos_F = self.start_F
        self.pos_M = self.start_M
        self.carry_F = False
        self.carry_M = False
        self._blocks = dict(self.pickups)
        self.steps_in_episode = 0
        self._viz_goal = None

    def is_terminal(self) -> bool:
        no_blocks = all(v == 0 for v in self._blocks.values()) if self._blocks else True
        if no_blocks and not self.carry_F and not self.carry_M:
            return True
        if self.steps_in_episode >= self.max_steps_per_episode:
            return True
        return False

    # ---- State encoding for independent learners ----
    def encode_state(self, agent_id: str) -> tuple:
        mask = 0
        for i,cell in enumerate(sorted(self.pickups.keys())):
            if self._blocks.get(cell, 0) > 0:
                mask |= (1 << i)
        def idx(rc: Coord) -> int:
            r,c = rc
            return r * self.W + c
        if agent_id == 'F':
            s = (idx(self.pos_F), idx(self.pos_M), int(self.carry_F), int(self.carry_M), mask)
        else:
            s = (idx(self.pos_M), idx(self.pos_F), int(self.carry_M), int(self.carry_F), mask)
        return s

    # ---- Applicable actions for the acting agent ----
    def applicable_actions(self, agent_id: str) -> List[str]:
        if agent_id == 'F':
            agent_pos, other_pos, carrying = self.pos_F, self.pos_M, self.carry_F
        else:
            agent_pos, other_pos, carrying = self.pos_M, self.pos_F, self.carry_M
        return rules.applicable_actions(agent_id, agent_pos, other_pos, carrying,
                                        self._blocks, self.drops, self.H, self.W, self.obstacles)

    # ---- Potential function for shaping ----
    def _phi(self, agent_id: str) -> float:
        """Higher is better. Use negative Manhattan distance to nearest target."""
        def manh(a: Coord, b: Coord) -> int:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        if agent_id == 'F':
            pos, carrying = self.pos_F, self.carry_F
        else:
            pos, carrying = self.pos_M, self.carry_M

        # choose target set
        if carrying:
            targets = self.drops
        else:
            # only pickups with remaining blocks
            targets = [cell for cell, n in self._blocks.items() if n > 0]

        if not targets:
            return 0.0

        dmin = min(manh(pos, t) for t in targets)
        # negative distance, so moving closer -> larger Phi
        return -float(dmin)

    # ---- Step transition ----
    def step(self, agent_id: str, action: str):
        self.steps_in_episode += 1

        if agent_id == 'F':
            agent_pos, other_pos = self.pos_F, self.pos_M
            carrying = self.carry_F
        else:
            agent_pos, other_pos = self.pos_M, self.pos_F
            carrying = self.carry_M

        acts = self.applicable_actions(agent_id)
        waited = (action == rules.IDLE_ACTION)

        # Before-state potential
        phi_s = self._phi(agent_id)

        valid = True
        picked = dropped = False
        attempted_conflict = False
        new_pos = agent_pos

        if action in rules.MOVE_ACTIONS:
            dr, dc = rules.MOVES[action]
            cand = (agent_pos[0] + dr, agent_pos[1] + dc)

            # mark attempted conflicts (useful for logging + penalty)
            if rules.inside(cand, self.H, self.W) and (cand == other_pos):
                attempted_conflict = True

            # validate move against geometry/occupancy
            if rules.can_move_to(cand, self.H, self.W, self.obstacles, other_pos):
                new_pos = cand
            else:
                valid = False

        elif action == 'PICKUP':
            if (agent_pos in self._blocks) and self._blocks.get(agent_pos, 0) > 0 and not carrying:
                self._blocks[agent_pos] -= 1
                carrying = True
                picked = True
            else:
                valid = False

        elif action == 'DROPOFF':
            if carrying and (agent_pos in self.drops):
                carrying = False
                dropped = True
            else:
                valid = False

        elif action == rules.IDLE_ACTION:
            # always valid, just wait in place
            pass
        else:
            valid = False

        # apply new state
        if agent_id == 'F':
            self.pos_F = new_pos
            self.carry_F = carrying
        else:
            self.pos_M = new_pos
            self.carry_M = carrying

        # After-state potential & shaping
        phi_sp = self._phi(agent_id)
        shaping = self.rewards.shaping_coef * (self.rewards.gamma_shaping * phi_sp - phi_s)

        finished = self.is_terminal()

        base = rules.compute_reward(
            action, valid, picked, dropped, self.rewards.as_dict(),
            attempted_conflict=attempted_conflict,
            waited=waited,
            finished=finished
        )
        reward = base + shaping

        info = {
            'valid': valid,
            'picked': picked,
            'dropped': dropped,
            'attempted_conflict': attempted_conflict,
            'pos_agent': new_pos,
            'pos_F': self.pos_F,
            'pos_M': self.pos_M,
            'carry_F': self.carry_F,
            'carry_M': self.carry_M,
            'terminal': finished,
            'steps_in_episode': self.steps_in_episode,
            'phi_before': phi_s,
            'phi_after': phi_sp,
            'shaping': shaping,
        }
        return None, reward, info

    # ---- Viz helpers ----
    def reset_to(self, start_cell, goal_cell=None, agent_id: str='F'):
        if agent_id == 'F':
            self.pos_F = tuple(start_cell)
        else:
            self.pos_M = tuple(start_cell)
        self._viz_goal = tuple(goal_cell) if goal_cell is not None else None
        return None
