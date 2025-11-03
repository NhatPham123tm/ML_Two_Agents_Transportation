# applicability checks, rewards, terminal
from typing import Tuple, List, Dict

Action = str  # 'UP','DOWN','LEFT','RIGHT','PICKUP','DROPOFF','WAIT'

MOVES = {
    'UP':    (-1, 0),
    'DOWN':  ( 1, 0),
    'LEFT':  ( 0,-1),
    'RIGHT': ( 0, 1),
}
MOVE_ACTIONS = tuple(MOVES.keys())
SPECIAL_ACTIONS = ('PICKUP','DROPOFF')
IDLE_ACTION = 'WAIT'  # new

def inside(cell: Tuple[int,int], H: int, W: int) -> bool:
    r,c = cell
    return 0 <= r < H and 0 <= c < W

def occupied(cell: Tuple[int,int], other_pos: Tuple[int,int]) -> bool:
    return cell == other_pos

def can_move_to(cell: Tuple[int,int], H:int, W:int, obstacles:set, other_pos:Tuple[int,int]) -> bool:
    return inside(cell, H, W) and (cell not in obstacles) and not occupied(cell, other_pos)

def applicable_actions(agent_id: str,
                       agent_pos: Tuple[int,int],
                       other_pos: Tuple[int,int],
                       carrying: bool,
                       pickups: Dict[Tuple[int,int], int],
                       drops: List[Tuple[int,int]],
                       H: int, W: int, obstacles:set) -> List[Action]:
    acts: List[Action] = []

    # movement
    for a,(dr,dc) in MOVES.items():
        nr, nc = agent_pos[0]+dr, agent_pos[1]+dc
        if can_move_to((nr,nc), H, W, obstacles, other_pos):
            acts.append(a)

    # pickup: if on pickup cell with remaining blocks and not carrying
    if (agent_pos in pickups) and pickups.get(agent_pos, 0) > 0 and not carrying:
        acts.append('PICKUP')

    # dropoff: if carrying and on drop cell
    if carrying and (agent_pos in drops):
        acts.append('DROPOFF')

    # always allow WAIT to break stalemates
    acts.append(IDLE_ACTION)
    return acts

# ---- Rewards ----
# rewards schema example:
# {
#   'step_cost': -0.03,
#   'idle_penalty': -0.02,
#   'pickup_reward': 2.0,
#   'drop_reward': 8.0,
#   'finish_reward': 5.0,
#   'invalid_penalty': -0.3,
#   'collision_penalty': -1.0,
# }
def compute_reward(action: Action,
                   valid: bool,
                   picked: bool,
                   dropped: bool,
                   rewards: dict,
                   *,
                   attempted_conflict: bool = False,
                   waited: bool = False,
                   finished: bool = False) -> float:
    r = rewards.get('step_cost', 0.0)
    if waited:
        r += rewards.get('idle_penalty', 0.0)
    if not valid:
        r += rewards.get('invalid_penalty', 0.0)
    if attempted_conflict:
        r += rewards.get('collision_penalty', 0.0)
    if picked:
        r += rewards.get('pickup_reward', 0.0)
    if dropped:
        r += rewards.get('drop_reward', 0.0)
    if finished:
        r += rewards.get('finish_reward', 0.0)
    return r
