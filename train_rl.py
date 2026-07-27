"""Train the QTableAgent to control the Intersection2D signal.

The agent observes bucketised queue counts (state), chooses a green phase
(action), and is rewarded by the negative of the waiting time incurred each
step. After 200 training episodes it is evaluated greedily against the
FixedTimer and MaxQueue baselines on the same held-out test seeds, and the
per-episode training wait time is saved as rl_learning_curve.png.
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

# training seeds are disjoint from the test seeds so evaluation is held out.
TRAIN_SEED_BASE = 1000
TEST_SEEDS = (0, 1, 2)


def make_env(seed):
    return Intersection2D(arm_length=ARM_LENGTH, p_arrival=P_ARRIVAL, seed=seed)


def avg_wait(inter):
    """Mean vehicle-steps spent blocked per vehicle that crossed."""
    return inter.total_wait_steps / max(inter.total_crossed, 1)


def run_controller(act, seed, steps):
    """Run a controller (act(inter) -> phase) for ``steps`` steps."""
    inter = make_env(seed)
    for _ in range(steps):
        inter.step(phase=act(inter))
    return avg_wait(inter)


def train(agent):
    """Train ``agent`` for EPISODES episodes; return per-episode wait times."""
    curve = []
    for ep in range(EPISODES):
        inter = make_env(TRAIN_SEED_BASE + ep)
        state = agent.discretize(inter.queue_counts())
        for t in range(EPISODE_STEPS):
            action = agent.select_action(state)

            before = inter.total_wait_steps
            inter.step(phase=action)
            reward = -float(inter.total_wait_steps - before)   # penalise waiting

            next_state = agent.discretize(inter.queue_counts())
            agent.learn(state, action, reward, next_state,
                        done=(t == EPISODE_STEPS - 1))
            state = next_state

        agent.decay_epsilon()
        curve.append(avg_wait(inter))
        if (ep + 1) % 20 == 0:
            print(f"episode {ep + 1:3d}/{EPISODES}  "
                  f"avg_wait={curve[-1]:6.3f}  eps={agent.epsilon:.3f}  "
                  f"states={agent.num_states}")
    return curve


def evaluate(agent):
    """Greedy evaluation of all three controllers on the test seeds."""
    agent.epsilon = 0.0

    def rl_act(inter):
        return agent.select_action(agent.discretize(inter.queue_counts()))

    # a fresh controller instance per seed keeps MaxQueue's dwell state clean.
    factories = {
        "Q-Learning": lambda: rl_act,
        FixedTimerController.name: lambda: FixedTimerController(12).act,
        MaxQueueController.name: lambda: MaxQueueController(5).act,
    }

    results = {}
    for name, factory in factories.items():
        waits = [run_controller(factory(), s, EVAL_STEPS) for s in TEST_SEEDS]
        results[name] = float(np.mean(waits))
    return results


def main():
    agent = QTableAgent(n_phases=2, alpha=0.1, gamma=0.9, epsilon=1.0,
                        epsilon_min=0.05, epsilon_decay=0.97, seed=0)

    print(f"Training over {EPISODES} episodes "
          f"({EPISODE_STEPS} steps each)...")
    curve = train(agent)

    print("\nEvaluating on test seeds", TEST_SEEDS, "(greedy)...")
    results = evaluate(agent)
    print(f"\n{'Controller':<14}{'AvgWait (test)':>16}")
    print("-" * 30)
    for name, wait in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"{name:<14}{wait:>16.3f}")

    # learning curve with a moving-average overlay and baseline references.
    episodes = np.arange(1, EPISODES + 1)
    window = 10
    smooth = np.convolve(curve, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, curve, color="tab:blue", alpha=0.3,
            label="Q-Learning (per episode)")
    ax.plot(episodes[window - 1:], smooth, color="tab:blue", linewidth=2,
            label=f"Q-Learning ({window}-ep moving avg)")
    ax.axhline(results[FixedTimerController.name], color="tab:red",
               linestyle="--", label="FixedTimer (test)")
    ax.axhline(results[MaxQueueController.name], color="tab:green",
               linestyle="--", label="MaxQueue (test)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average vehicle wait time")
    ax.set_title("Q-Learning Signal Control — Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "rl_learning_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved learning curve to {out_path}")


if __name__ == "__main__":
    main()
