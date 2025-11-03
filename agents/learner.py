# QLearningAgent, SarsaAgent (independent tables)
"""Independent Q-learning and SARSA agents for PD-World.

Expected env API:
  - env.encode_state(agent_id) -> tuple
  - env.applicable_actions(agent_id) -> List[str]
  - env.step(agent_id, action) -> (s', reward, info)  # s' can be None; re-encode after applying
"""
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import random
from . import policy as pol
from . import state as st

class BaseAgent:
    def __init__(self, agent_id: str, alpha: float, gamma: float, seed: Optional[int] = None):
        self.agent_id = agent_id
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.Q: Dict[Tuple[tuple, str], float] = defaultdict(float)
        self.rng = random.Random(seed)

    def select_action(self, env, policy_name: str) -> Tuple[tuple, str, List[str]]:
        s = st.build_state_from_env(env, self.agent_id)
        applicable = env.applicable_actions(self.agent_id)
        a = pol.choose_action(policy_name, s, applicable, self.Q, self.rng)
        return s, a, applicable

class QLearningAgent(BaseAgent):
    def step(self, env, policy_name: str):
        """Pick action by policy, act, and perform Q-learning update.
        Returns: (reward, info, transition) where transition=(s,a,s',applicable_next)
        """
        s, a, applicable = self.select_action(env, policy_name)
        _sp, r, info = env.step(self.agent_id, a)
        s_prime = st.build_state_from_env(env, self.agent_id)  # re-encode from new env state
        applicable_next = env.applicable_actions(self.agent_id)
        max_next = 0.0
        if applicable_next:
            max_next = max(self.Q[(s_prime, ap)] for ap in applicable_next)
        td_target = r + self.gamma * max_next
        td = td_target - self.Q[(s, a)]
        self.Q[(s, a)] += self.alpha * td
        return r, info, (s, a, s_prime, applicable_next)

class SarsaAgent(BaseAgent):
    def __init__(self, agent_id: str, alpha: float, gamma: float, seed: Optional[int] = None, policy_name: str = 'PEXPLOIT'):
        super().__init__(agent_id, alpha, gamma, seed)
        self._prev: Optional[Tuple[tuple, str]] = None
        self.policy_name = policy_name  # on-policy selection

    def step(self, env, policy_name: Optional[str] = None):
        """On-policy SARSA update using current policy for a' selection.
        If this is the first call (no previous action), it behaves like action selection only and defers update to next call.
        Returns: (reward, info, transition) where transition=(s_prev,a_prev,s_curr,a_curr)
        """
        polname = policy_name or self.policy_name
        s_t = st.build_state_from_env(env, self.agent_id)
        applicable_t = env.applicable_actions(self.agent_id)
        a_t = pol.choose_action(polname, s_t, applicable_t, self.Q, self.rng)
        _sp, r_t, info = env.step(self.agent_id, a_t)
        s_tp1 = st.build_state_from_env(env, self.agent_id)
        applicable_tp1 = env.applicable_actions(self.agent_id)
        if self._prev is not None:
            (s_prev, a_prev) = self._prev
            q_prev = self.Q[(s_prev, a_prev)]
            q_next = self.Q[(s_t, a_t)] if applicable_t else 0.0
            td_target = r_t + self.gamma * q_next
            td = td_target - q_prev
            self.Q[(s_prev, a_prev)] = q_prev + self.alpha * td
        self._prev = (s_t, a_t)
        return r_t, info, (self._prev[0], self._prev[1], s_tp1, None)
