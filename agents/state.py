# state encoding/decoding utilities

"""State utilities for PD-World independent learners."""
from typing import Tuple, Callable, Any
from dataclasses import dataclass

# Action aliases for consistency across modules
MOVE_ACTIONS = ('UP','DOWN','LEFT','RIGHT')
ALL_ACTIONS = ('UP','DOWN','LEFT','RIGHT','PICKUP','DROPOFF')

def pos_to_idx(r: int, c: int, W: int) -> int:
    return r * W + c

def idx_to_pos(idx: int, W: int) -> Tuple[int, int]:
    return (idx // W, idx % W)

def build_state_from_env(env, agent_id: str) -> tuple:
    """Delegate to env.encode_state(agent_id)."""
    return env.encode_state(agent_id)

@dataclass
class SliceFnFactory:
    """Factory to generate canonical state slicers for visualization.
    Use this to create a (r,c)->state function with fixed assumptions
    for other-agent position, carry flags, and task mask.
    """
    H: int
    W: int
    other_pos: Tuple[int,int] = (0,0)
    carry_self: int = 0
    carry_other: int = 0
    task_mask: int = 0

    def make(self) -> Callable[[int,int], tuple]:
        def slice_fn(r: int, c: int) -> tuple:
            self_idx = pos_to_idx(r, c, self.W)
            other_idx = pos_to_idx(self.other_pos[0], self.other_pos[1], self.W)
            return (self_idx, other_idx, int(self.carry_self), int(self.carry_other), int(self.task_mask))
        return slice_fn
