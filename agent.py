"""Tabular Q-learning agent for traffic-signal control.

The agent observes per-approach queue counts, discretizes them into a small
set of buckets to keep the state space tractable, and chooses which green
phase to activate. Reward is the negative of total waiting time, so the agent
learns to minimise how long vehicles sit queued.
"""

from collections import defaultdict

import numpy as np

# Upper edges (inclusive) that map a raw queue count onto a bucket index:
#   count == 0        -> 0
#   1 <= count <= 3   -> 1
#   4 <= count <= 8   -> 2
#   count >= 9        -> 3
_BUCKET_EDGES = (0, 3, 8)
NUM_BUCKETS = len(_BUCKET_EDGES) + 1


def bucketize(count):
    """Map a single queue count to its bucket index (0..NUM_BUCKETS-1)."""
    for bucket, edge in enumerate(_BUCKET_EDGES):
        if count <= edge:
            return bucket
    return NUM_BUCKETS - 1


class QTableAgent:
    """Epsilon-greedy tabular Q-learning agent for green-phase selection.

    Parameters
    ----------
    n_phases : int
        Number of discrete green-phase actions the agent can choose from.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability.
    epsilon_min : float
        Floor for epsilon during decay.
    epsilon_decay : float
        Multiplicative decay applied to epsilon each time ``decay_epsilon`` is
        called (typically once per episode).
    seed : int or None
        Seed for the internal random number generator.
    """

    def __init__(self, n_phases, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, seed=None):
        self.n_phases = int(n_phases)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

        # state tuple -> action-value array of length n_phases.
        self.q_table = defaultdict(lambda: np.zeros(self.n_phases))

    def discretize(self, queue_counts):
        """Turn an iterable of raw queue counts into a hashable state tuple.

        Each approach's queue count is mapped to one of NUM_BUCKETS buckets
        (0, 1-3, 4-8, 9+); the resulting tuple is the state key for the table.
        """
        return tuple(bucketize(c) for c in queue_counts)

    @staticmethod
    def reward(waiting_times):
        """Reward for a step: the negative of total waiting time.

        ``waiting_times`` is any iterable of per-vehicle or per-approach
        waiting-time values. Larger backlogs give a more negative reward, so
        the agent is pushed to keep total waiting low.
        """
        return -float(np.sum(waiting_times))

    def select_action(self, state):
        """Epsilon-greedy action selection for the given state tuple."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_phases))
        q_values = self.q_table[state]
        # break ties randomly so the agent doesn't lock onto action 0 early.
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def learn(self, state, action, reward, next_state, done=False):
        """Apply the Q-learning update for one transition.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        When ``done`` is True the bootstrap term is dropped.
        """
        q_sa = self.q_table[state][action]
        best_next = 0.0 if done else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * best_next
        self.q_table[state][action] = q_sa + self.alpha * (td_target - q_sa)
        return td_target - q_sa

    def decay_epsilon(self):
        """Shrink epsilon toward its floor (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    @property
    def num_states(self):
        """Number of distinct states the agent has encountered so far."""
        return len(self.q_table)
