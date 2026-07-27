"""Train the QTableAgent to control the Intersection2D signal, with an
optional safety layer on signal timing.

Two safety constraints are enforced by a SafetyGovernor that filters the
agent's raw action before it reaches the environment:

  * minimum green : a phase must stay green for at least MIN_GREEN steps
                    before the agent is allowed to switch it.
  * maximum red   : no approach may stay red for more than MAX_RED steps;
                    once the current phase has been green that long, a switch
                    is forced so the opposing approach is served.

We train one unconstrained agent and one constrained agent (governor active
during training and evaluation), then compare both against the FixedTimer and
MaxQueue baselines on held-out test seeds, reporting mean and 95th-percentile
vehicle wait time.
"""

import matplotlib.pyplot as plt
import numpy as np

from agent import QTableAgent
from controllers import FixedTimerController, MaxQueueController
from grid import Intersection2D

EPISODES = 200
EPISODE_STEPS = 1000
EVAL_STEPS = 1000
ARM_LENGTH = 12
P_ARRIVAL = 0.4

MIN_GREEN = 5     # minimum green phase duration (steps)
MAX_RED = 30      # maximum time an approach may stay red (steps)

# training seeds are disjoint from the test seeds so evaluation is held out.
TRAIN_SEED_BASE = 1000
TEST_SEEDS = (0, 1, 2)


class SafetyGovernor:
    """Filter a requested phase to honour min-green and max-red constraints."""

    def __init__(self, min_green=MIN_GREEN, max_red=MAX_RED):
        self.min_green = int(min_green)
        self.max_red = int(max_red)
        self.phase = Intersection2D.PHASE_NS
        self.time_in_phase = 0

    def filter(self, requested):
        if self.time_in_phase >= self.max_red:
            chosen = 1 - self.phase                 # force switch: red too long
        elif self.time_in_phase < self.min_green:
            chosen = self.phase                     # hold: green too short
        else:
            chosen = int(requested)                 # obey the agent

        if chosen != self.phase:
            self.phase = chosen
            self.time_in_phase = 1
        else:
            self.time_in_phase += 1
        return chosen


def make_env(seed):
    return Intersection2D(arm_length=ARM_LENGTH, p_arrival=P_ARRIVAL, seed=seed)


# ------------------------------------------------------------------ training
def train(agent, constrained):
    """Train ``agent`` for EPISODES episodes; return per-episode mean wait."""
    curve = []
    for ep in range(EPISODES):
        inter = make_env(TRAIN_SEED_BASE + ep)
        gov = SafetyGovernor() if constrained else None
        state = agent.discretize(inter.queue_counts())
        for t in range(EPISODE_STEPS):
            action = agent.select_action(state)
            applied = gov.filter(action) if gov else action

            before = inter.total_wait_steps
            inter.step(phase=applied)
            reward = -float(inter.total_wait_steps - before)   # penalise waiting

            next_state = agent.discretize(inter.queue_counts())
            # learn on the action actually executed (post-safety filter).
            agent.learn(state, applied, reward, next_state,
                        done=(t == EPISODE_STEPS - 1))
            state = next_state

        agent.decay_epsilon()
        waits = inter.completed_waits
        curve.append(float(np.mean(waits)) if waits else 0.0)
        if (ep + 1) % 40 == 0:
            print(f"  episode {ep + 1:3d}/{EPISODES}  "
                  f"mean_wait={curve[-1]:6.3f}  eps={agent.epsilon:.3f}")
    return curve


# ---------------------------------------------------------------- evaluation
def collect_waits(act, seed, steps, governor=None):
    """Run a policy for ``steps`` steps; return the list of per-vehicle waits."""
    inter = make_env(seed)
    for _ in range(steps):
        action = act(inter)
        if governor is not None:
            action = governor.filter(action)
        inter.step(phase=action)
    return inter.completed_waits


def rl_policy(agent):
    """Greedy action function for a trained agent."""
    def act(inter):
        return agent.select_action(agent.discretize(inter.queue_counts()))
    return act


def summarize(list_of_wait_lists):
    """Pool per-vehicle waits across seeds; return (mean, p95)."""
    pooled = np.concatenate([np.asarray(w) for w in list_of_wait_lists])
    return float(pooled.mean()), float(np.percentile(pooled, 95))


def evaluate(agent_unc, agent_con):
    agent_unc.epsilon = 0.0
    agent_con.epsilon = 0.0

    # each entry: name -> (act-factory, governor-factory or None)
    policies = {
        "RL unconstrained": (lambda: rl_policy(agent_unc), lambda: None),
        "RL constrained": (lambda: rl_policy(agent_con), lambda: SafetyGovernor()),
        FixedTimerController.name: (lambda: FixedTimerController(12).act,
                                    lambda: None),
        MaxQueueController.name: (lambda: MaxQueueController(5).act,
                                  lambda: None),
    }

    results = {}
    for name, (make_act, make_gov) in policies.items():
        per_seed = [collect_waits(make_act(), s, EVAL_STEPS, make_gov())
                    for s in TEST_SEEDS]
        results[name] = summarize(per_seed)
    return results


# --------------------------------------------------------------------- main
def main():
    """Train both agents, evaluate against baselines, and save the learning curve."""
    print("Training unconstrained agent...")
    agent_unc = QTableAgent(n_phases=2, alpha=0.1, gamma=0.9, epsilon=1.0,
                            epsilon_min=0.05, epsilon_decay=0.97, seed=0)
    curve_unc = train(agent_unc, constrained=False)

    print("Training constrained agent (min_green=%d, max_red=%d)..."
          % (MIN_GREEN, MAX_RED))
    agent_con = QTableAgent(n_phases=2, alpha=0.1, gamma=0.9, epsilon=1.0,
                            epsilon_min=0.05, epsilon_decay=0.97, seed=0)
    curve_con = train(agent_con, constrained=True)

    print("\nEvaluating on test seeds", TEST_SEEDS, "(greedy)...")
    results = evaluate(agent_unc, agent_con)

    print(f"\n{'Controller':<20}{'MeanWait':>12}{'p95Wait':>12}")
    print("-" * 44)
    for name, (mean_w, p95_w) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<20}{mean_w:>12.3f}{p95_w:>12.3f}")

    print(f"\nSteps/run: {EVAL_STEPS} | test seeds: {TEST_SEEDS} | "
          f"safety: min_green={MIN_GREEN}, max_red={MAX_RED}")

    # learning curves for both agents, with baseline mean-wait references.
    episodes = np.arange(1, EPISODES + 1)
    window = 10

    def smooth(c):
        return np.convolve(c, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes[window - 1:], smooth(curve_unc), color="tab:blue",
            linewidth=2, label="RL unconstrained")
    ax.plot(episodes[window - 1:], smooth(curve_con), color="tab:orange",
            linewidth=2, label="RL constrained")
    ax.axhline(results[FixedTimerController.name][0], color="tab:red",
               linestyle="--", label="FixedTimer")
    ax.axhline(results[MaxQueueController.name][0], color="tab:green",
               linestyle="--", label="MaxQueue")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean vehicle wait time")
    ax.set_title("Q-Learning Signal Control — Learning Curve "
                 "(constrained vs unconstrained)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "rl_learning_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved learning curve to {out_path}")


if __name__ == "__main__":
    main()
