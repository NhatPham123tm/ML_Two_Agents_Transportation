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
        # The value of the next state is 0 if it's a terminal state.
        if applicable_next and not info.get("terminal"):
            max_next = max(self.Q[(s_prime, ap)] for ap in applicable_next)

        td_target = r + self.gamma * max_next
        td = td_target - self.Q[(s, a)]
        self.Q[(s, a)] += self.alpha * td
        return r, info, (s, a, s_prime, applicable_next)

class SarsaAgent(BaseAgent):
    def __init__(self, agent_id: str, alpha: float, gamma: float, seed: Optional[int] = None, policy_name: str = 'PEXPLOIT'):
        super().__init__(agent_id, alpha, gamma, seed)
        # Previous state, action, and reward for the SARSA update
        self._s_prev: Optional[tuple] = None
        self._a_prev: Optional[str] = None
        self._r_prev: Optional[float] = None
        self.policy_name = policy_name  # on-policy selection

    def step(self, env, policy_name: Optional[str] = None):
        """On-policy SARSA update using current policy for a' selection.
        The update for (s,a) happens on the next call to step(), after r' is observed.
        Returns: (reward, info, transition) where transition=(s_curr, a_curr, s_next, None)
        """
        polname = policy_name or self.policy_name
        s_curr = st.build_state_from_env(env, self.agent_id)
        applicable_curr = env.applicable_actions(self.agent_id)
        a_curr = pol.choose_action(polname, s_curr, applicable_curr, self.Q, self.rng)

        # Perform SARSA update for the *previous* step (s_prev, a_prev)
        # We now have (s, a, r, s', a') where s=s_prev, a=a_prev, r=r_prev, s'=s_curr, a'=a_curr
        if self._s_prev is not None and self._a_prev is not None and self._r_prev is not None:
            q_prev = self.Q[(self._s_prev, self._a_prev)]
            q_curr = self.Q[(s_curr, a_curr)]  # Q(s', a')
            
            td_target = self._r_prev + self.gamma * q_curr
            td_error = td_target - q_prev
            self.Q[(self._s_prev, self._a_prev)] += self.alpha * td_error

        # Act in the environment and store results for the *next* update
        _s_next, r_curr, info = env.step(self.agent_id, a_curr)
        self._s_prev = s_curr
        self._a_prev = a_curr
        self._r_prev = r_curr

        s_next = st.build_state_from_env(env, self.agent_id)
        return r_curr, info, (s_curr, a_curr, s_next, None)
