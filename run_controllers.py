"""Benchmark baseline controllers on the Intersection2D environment.

Runs FixedTimer and MaxQueue for STEPS steps across several random seeds and
prints a table of average vehicle wait time and average queue length.
"""

import numpy as np

from controllers import FixedTimerController, MaxQueueController
from grid import Intersection2D

STEPS = 1000
SEEDS = (0, 1, 2)
ARM_LENGTH = 12
P_ARRIVAL = 0.4


def run_episode(make_controller, seed):
    """Run one controller for STEPS steps; return (avg_wait, avg_queue)."""
    inter = Intersection2D(arm_length=ARM_LENGTH, p_arrival=P_ARRIVAL, seed=seed)
    controller = make_controller()

    queue_total = 0
    for _ in range(STEPS):
        phase = controller.act(inter)
        inter.step(phase=phase)
        queue_total += sum(inter.queue_counts())

    # mean wait per vehicle that made it through; mean total queue per step.
    avg_wait = inter.total_wait_steps / max(inter.total_crossed, 1)
    avg_queue = queue_total / STEPS
    return avg_wait, avg_queue


def main():
    """Benchmark each controller across seeds and print the results table."""
    controllers = {
        FixedTimerController.name: lambda: FixedTimerController(phase_duration=12),
        MaxQueueController.name: lambda: MaxQueueController(min_green=5),
    }

    header = f"{'Controller':<12}{'Seed':>6}{'AvgWait':>12}{'AvgQueue':>12}"
    print(header)
    print("-" * len(header))

    for name, make in controllers.items():
        waits, queues = [], []
        for seed in SEEDS:
            avg_wait, avg_queue = run_episode(make, seed)
            waits.append(avg_wait)
            queues.append(avg_queue)
            print(f"{name:<12}{seed:>6}{avg_wait:>12.3f}{avg_queue:>12.3f}")
        print(f"{name:<12}{'mean':>6}{np.mean(waits):>12.3f}"
              f"{np.mean(queues):>12.3f}")
        print("-" * len(header))

    print(f"\nSteps per run: {STEPS} | seeds: {SEEDS} | "
          f"arm={ARM_LENGTH}, p_arrival={P_ARRIVAL}")
    print("AvgWait  = mean vehicle-steps spent blocked per vehicle crossed")
    print("AvgQueue = mean total queue length (all 4 approaches) per step")


if __name__ == "__main__":
    main()
