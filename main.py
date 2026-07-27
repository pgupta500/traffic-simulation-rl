"""Run the Nagel-Schreckenberg traffic CA and draw a space-time diagram."""

import matplotlib.pyplot as plt
import numpy as np

from world import Road1D


def main():
    length = 300
    steps = 300

    road = Road1D(length=length, density=0.25, v_max=5, p_slow=0.3, seed=42)
    history = road.run(steps)

    # occupancy: 1 where a vehicle sits, 0 elsewhere. Time runs downward.
    occupancy = (history != Road1D.EMPTY).astype(np.int64)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(occupancy, cmap="binary", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Position (cell)")
    ax.set_ylabel("Time (step)")
    ax.set_title("Nagel-Schreckenberg 1D Traffic — Space-Time Diagram")

    fig.tight_layout()
    out_path = "spacetime.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved space-time plot to {out_path}")

    # report a simple flow statistic: mean velocity of all vehicles over time.
    velocities = np.where(history != Road1D.EMPTY, history, 0)
    mean_v = velocities.sum() / max(occupancy.sum(), 1)
    print(f"Mean vehicle velocity: {mean_v:.3f} cells/step")


if __name__ == "__main__":
    main()
