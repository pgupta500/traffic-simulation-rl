"""A 4-way single-lane signalised intersection.

Four incoming approaches (North, South, East, West) each feed a single lane of
cells toward a central crossing. A traffic light runs one of two phases:

  * PHASE_NS -- North and South approaches have green.
  * PHASE_EW -- East and West approaches have green.

Vehicles advance one cell per step toward the intersection when the cell ahead
is free. The vehicle at the stop line may enter the crossing (and leave the
system) only when its approach has green. Vehicles arrive stochastically at the
tail of each approach. Queue counts on all four approaches are exposed so a
controller (e.g. a Q-learning agent) can act on them.
"""

import numpy as np

# approach order is fixed so queue_counts() lines up with agent state tuples.
APPROACHES = ("N", "S", "E", "W")


class Intersection2D:
    PHASE_NS = 0
    PHASE_EW = 1

    def __init__(self, arm_length=12, p_arrival=0.4, seed=None):
        self.L = int(arm_length)
        self.p_arrival = float(p_arrival)
        self.rng = np.random.default_rng(seed)

        # occupancy per approach: index 0 = tail (entrance), L-1 = stop line.
        self.lanes = {d: np.zeros(self.L, dtype=bool) for d in APPROACHES}
        # accumulated blocked-steps of the vehicle occupying each cell.
        self.waits = {d: np.zeros(self.L, dtype=int) for d in APPROACHES}
        self.phase = self.PHASE_NS
        self.time = 0
        self.total_crossed = 0
        # cumulative vehicle-steps spent blocked (used for mean wait time).
        self.total_wait_steps = 0
        # per-vehicle total wait recorded when a vehicle crosses (for p95 etc.).
        self.completed_waits = []

    # ------------------------------------------------------------------ state
    def is_green(self, approach):
        """True if the given approach currently has a green light."""
        if self.phase == self.PHASE_NS:
            return approach in ("N", "S")
        return approach in ("E", "W")

    def queue_counts(self):
        """Vehicles waiting on each approach, in APPROACHES order (N,S,E,W)."""
        return tuple(int(self.lanes[d].sum()) for d in APPROACHES)

    def cell_coords(self, approach, idx):
        """Grid (row, col) of cell ``idx`` on ``approach`` in a 2L+1 grid.

        The intersection centre sits at (L, L); each approach is one arm along
        the central row/column, with idx increasing toward the centre.
        """
        L = self.L
        if approach == "N":
            return idx, L
        if approach == "S":
            return 2 * L - idx, L
        if approach == "E":
            return L, 2 * L - idx
        return L, idx  # "W"

    # -------------------------------------------------------------- dynamics
    def _advance_lane(self, approach):
        """Advance one approach by a single step; return vehicles that crossed."""
        occ = self.lanes[approach]
        wait = self.waits[approach]
        new = occ.copy()
        new_wait = wait.copy()
        moved = np.zeros(self.L, dtype=bool)   # original vehicles that advanced
        crossed = 0

        # stop-line vehicle enters the crossing (leaves system) if green.
        if self.is_green(approach) and occ[-1]:
            new[-1] = False
            moved[-1] = True
            self.completed_waits.append(int(wait[-1]))
            new_wait[-1] = 0
            crossed = 1

        # move remaining vehicles forward one cell if the cell ahead is free.
        # process front-to-back so a freed cell propagates within the same step.
        for i in range(self.L - 2, -1, -1):
            if occ[i] and not new[i + 1]:
                new[i + 1] = True
                new_wait[i + 1] = wait[i]   # carry the vehicle's wait forward
                new[i] = False
                new_wait[i] = 0
                moved[i] = True

        # vehicles present that did not advance were blocked -> they waited.
        stayed = occ & ~moved
        new_wait[stayed] += 1
        self.total_wait_steps += int(stayed.sum())

        # stochastic arrival at the tail if the entrance cell is free.
        if not new[0] and self.rng.random() < self.p_arrival:
            new[0] = True
            new_wait[0] = 0

        self.lanes[approach] = new
        self.waits[approach] = new_wait
        return crossed

    def step(self, phase=None):
        """Advance the whole intersection one step.

        If ``phase`` is given (PHASE_NS or PHASE_EW) it is applied before the
        update, which is how a controller selects the green direction. Returns
        the queue counts after the step.
        """
        if phase is not None:
            self.phase = int(phase)
        for d in APPROACHES:
            self.total_crossed += self._advance_lane(d)
        self.time += 1
        return self.queue_counts()

    # ------------------------------------------------------------- rendering
    def grid(self):
        """Render the current state as an int array for imshow.

        Cell codes: 0 = off-road background, 1 = empty road, 2 = vehicle.
        """
        size = 2 * self.L + 1
        g = np.zeros((size, size), dtype=int)
        g[self.L, :] = 1          # horizontal (East-West) road
        g[:, self.L] = 1          # vertical (North-South) road
        for d in APPROACHES:
            occ = self.lanes[d]
            for idx in np.flatnonzero(occ):
                r, c = self.cell_coords(d, idx)
                g[r, c] = 2
        return g
