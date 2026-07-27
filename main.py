"""Density sweep for the Nagel-Schreckenberg 1D traffic CA.

For a range of vehicle densities we run the simulation, discard a warm-up
transient, and measure two steady-state quantities:

  * mean speed  -- average velocity of all vehicles over the measured steps.
  * throughput  -- flow q = (1/L) * sum of vehicle velocities per step,
                   i.e. the mean number of cells advanced per cell per step.

The result is the classic "fundamental diagram" of traffic flow, saved as
congestion_curve.png.
"""

import matplotlib.pyplot as plt
import numpy as np

from world import Road1D


def measure(density, length=500, steps=500, warmup=100,
            v_max=5, p_slow=0.3, seed=0):
    """Simulate one density and return (mean_speed, throughput).

    The first ``warmup`` steps are discarded so the measurement reflects the
    steady state rather than the initial transient.
    """
    road = Road1D(length=length, density=density, v_max=v_max,
                  p_slow=p_slow, seed=seed)
    history = road.run(steps)

    measured = history[warmup:]                       # (steps-warmup, length)
    occupied = measured != Road1D.EMPTY
    velocities = np.where(occupied, measured, 0)

    n_vehicles = occupied.sum()
    if n_vehicles == 0:
        return 0.0, 0.0

    # mean speed: average velocity over every vehicle-timestep observed.
    mean_speed = velocities.sum() / n_vehicles
    # throughput / flow: total distance covered per cell per step.
    throughput = velocities.sum() / (measured.shape[0] * length)
    return float(mean_speed), float(throughput)


def main():
    """Sweep density, print the flow statistics, and save the fundamental diagram."""
    densities = np.arange(0.05, 0.80 + 1e-9, 0.05)
    mean_speeds = np.empty_like(densities)
    throughputs = np.empty_like(densities)

    for i, rho in enumerate(densities):
        mean_speeds[i], throughputs[i] = measure(rho)
        print(f"density={rho:0.2f}  mean_speed={mean_speeds[i]:.3f}  "
              f"throughput={throughputs[i]:.3f}")

    fig, (ax_speed, ax_flow) = plt.subplots(1, 2, figsize=(13, 5))

    ax_speed.plot(densities, mean_speeds, "o-", color="tab:blue")
    ax_speed.set_xlabel("Density (vehicles / cell)")
    ax_speed.set_ylabel("Mean speed (cells / step)")
    ax_speed.set_title("Mean Speed vs. Density")
    ax_speed.grid(True, alpha=0.3)

    ax_flow.plot(densities, throughputs, "o-", color="tab:red")
    ax_flow.set_xlabel("Density (vehicles / cell)")
    ax_flow.set_ylabel("Throughput (vehicles / cell / step)")
    ax_flow.set_title("Throughput vs. Density")
    ax_flow.grid(True, alpha=0.3)

    fig.suptitle("Nagel-Schreckenberg Fundamental Diagram", fontsize=14)
    fig.tight_layout()
    out_path = "congestion_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved congestion curve to {out_path}")


if __name__ == "__main__":
    main()
