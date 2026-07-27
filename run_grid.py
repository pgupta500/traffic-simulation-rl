"""Animate the 4-way signalised intersection and save it as intersection.gif.

The light follows a fixed-time controller (alternating NS/EW green every
``phase_duration`` steps) so the queues visibly build and drain. Vehicles are
squares on the road arms; the coloured dots near the centre show each
approach's current signal (green = go, red = stop).
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from grid import APPROACHES, Intersection2D

PHASE_DURATION = 12        # steps of green before switching NS <-> EW
FRAMES = 160
FPS = 8

# off-road background, empty road, vehicle.
CMAP = ListedColormap(["#e8e8e8", "#3a3a3a", "#4c9be8"])


def signal_marker_xy(inter, approach):
    """Return (x, y) just outside the centre on an arm, for the signal dot."""
    L = inter.L
    offset = {
        "N": (L - 0.9, L - 0.9),
        "S": (L + 0.9, L + 0.9),
        "E": (L + 0.9, L - 0.9),
        "W": (L - 0.9, L + 0.9),
    }[approach]
    # offset is (row, col); imshow x=col, y=row.
    return offset[1], offset[0]


def main():
    """Animate the intersection under a fixed-time controller and save the GIF."""
    inter = Intersection2D(arm_length=12, p_arrival=0.4, seed=7)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(inter.grid(), cmap=CMAP, vmin=0, vmax=2,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # one signal dot per approach, recoloured each frame.
    signals = {}
    for d in APPROACHES:
        x, y = signal_marker_xy(inter, d)
        (dot,) = ax.plot(x, y, "o", markersize=12,
                         markeredgecolor="black", markeredgewidth=0.8)
        signals[d] = dot

    title = ax.set_title("")

    def update(frame):
        # fixed-time controller: alternate phases by wall-clock step.
        phase = (inter.time // PHASE_DURATION) % 2
        inter.step(phase=phase)

        im.set_data(inter.grid())
        for d in APPROACHES:
            signals[d].set_color("#2ecc71" if inter.is_green(d) else "#e74c3c")

        phase_name = "NS" if inter.phase == Intersection2D.PHASE_NS else "EW"
        n, s, e, w = inter.queue_counts()
        title.set_text(
            f"step {inter.time}  |  green: {phase_name}  |  "
            f"queues N{n} S{s} E{e} W{w}  |  crossed {inter.total_crossed}")
        return [im, title, *signals.values()]

    anim = animation.FuncAnimation(fig, update, frames=FRAMES,
                                   interval=1000 / FPS, blit=False)

    out_path = "intersection.gif"
    anim.save(out_path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"Saved animation to {out_path}")


if __name__ == "__main__":
    main()
