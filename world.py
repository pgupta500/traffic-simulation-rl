"""1D traffic Cellular Automaton following the Nagel-Schreckenberg model.

Each cell of the road is either empty (-1) or holds a single vehicle whose
value is its integer velocity in the range [0, v_max]. Vehicles move to the
right on a ring (periodic boundary conditions).
"""

import numpy as np


class Road1D:
    """A single-lane road simulated with the Nagel-Schreckenberg rules.

    Parameters
    ----------
    length : int
        Number of cells on the road.
    density : float
        Fraction of cells initially occupied by a vehicle (in [0, 1]).
    v_max : int
        Maximum vehicle velocity (cells per step).
    p_slow : float
        Probability that a moving vehicle randomly decelerates by one.
    seed : int or None
        Seed for the internal random number generator (for reproducibility).
    """

    EMPTY = -1

    def __init__(self, length=200, density=0.3, v_max=5, p_slow=0.3, seed=None):
        self.length = int(length)
        self.v_max = int(v_max)
        self.p_slow = float(p_slow)
        self.rng = np.random.default_rng(seed)

        # cells hold the velocity of the vehicle occupying them, or EMPTY.
        self.cells = np.full(self.length, self.EMPTY, dtype=np.int64)

        n_cars = int(round(density * self.length))
        positions = self.rng.choice(self.length, size=n_cars, replace=False)
        # start every car with velocity 0.
        self.cells[positions] = 0

    def step(self):
        """Advance the simulation by one time step and return the new state.

        Applies the four Nagel-Schreckenberg update rules synchronously to
        every vehicle:
          1. Acceleration:   v -> min(v + 1, v_max)
          2. Slowing down:   v -> min(v, gap)
          3. Randomization:  v -> max(v - 1, 0) with probability p_slow
          4. Motion:         each vehicle advances v cells.
        """
        occupied = np.flatnonzero(self.cells != self.EMPTY)
        if occupied.size == 0:
            return self.cells

        # sort vehicle positions so gaps can be computed from neighbours.
        positions = np.sort(occupied)
        velocities = self.cells[positions].copy()

        # gap = number of empty cells ahead of each vehicle (wraps around ring).
        next_positions = np.roll(positions, -1)
        gaps = (next_positions - positions) % self.length - 1
        # with a single vehicle the gap spans the whole ring minus itself.
        if positions.size == 1:
            gaps[:] = self.length - 1

        # 1. acceleration
        velocities = np.minimum(velocities + 1, self.v_max)
        # 2. slowing down to avoid collision
        velocities = np.minimum(velocities, gaps)
        # 3. random deceleration
        slow = (self.rng.random(velocities.size) < self.p_slow) & (velocities > 0)
        velocities[slow] -= 1

        # 4. motion with periodic boundary conditions
        new_positions = (positions + velocities) % self.length
        new_cells = np.full(self.length, self.EMPTY, dtype=np.int64)
        new_cells[new_positions] = velocities
        self.cells = new_cells
        return self.cells

    def run(self, steps):
        """Run for ``steps`` iterations, returning a (steps, length) history.

        Each row is the road state at one time step, with EMPTY (-1) for empty
        cells and the vehicle velocity otherwise.
        """
        history = np.empty((steps, self.length), dtype=np.int64)
        for t in range(steps):
            history[t] = self.cells
            self.step()
        return history

    @property
    def occupancy(self):
        """Boolean array: True where a cell currently holds a vehicle."""
        return self.cells != self.EMPTY

    @property
    def num_vehicles(self):
        return int(np.count_nonzero(self.occupancy))
