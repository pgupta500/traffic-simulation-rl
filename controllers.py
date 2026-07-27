"""Baseline signal controllers for the Intersection2D environment.

Each controller exposes ``act(intersection) -> phase`` returning the green
phase (Intersection2D.PHASE_NS or PHASE_EW) to apply on the next step.
"""

from grid import Intersection2D


class FixedTimerController:
    """Alternate NS/EW green on a fixed schedule, ignoring traffic state."""

    name = "FixedTimer"

    def __init__(self, phase_duration=12):
        self.phase_duration = int(phase_duration)

    def act(self, inter):
        """Return the scheduled green phase for the current step."""
        # switch phase every ``phase_duration`` steps regardless of queues.
        return (inter.time // self.phase_duration) % 2


class MaxQueueController:
    """Serve whichever axis (NS or EW) currently has the longer total queue.

    A ``min_green`` dwell time prevents the light from flickering every step
    when the two axes are closely matched.
    """

    name = "MaxQueue"

    def __init__(self, min_green=5):
        self.min_green = int(min_green)
        self._phase = Intersection2D.PHASE_NS
        self._time_in_phase = 0

    def act(self, inter):
        """Return the phase serving the longer-queued axis (min-green permitting)."""
        n, s, e, w = inter.queue_counts()
        ns_queue, ew_queue = n + s, e + w

        # hold the current phase until the minimum green has elapsed.
        if self._time_in_phase < self.min_green:
            self._time_in_phase += 1
            return self._phase

        desired = (Intersection2D.PHASE_NS if ns_queue >= ew_queue
                   else Intersection2D.PHASE_EW)
        if desired != self._phase:
            self._phase = desired
            self._time_in_phase = 0
        else:
            self._time_in_phase += 1
        return self._phase
