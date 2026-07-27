# traffic-simulation-rl

A 1D traffic simulation and reinforcement-learning sandbox.

- **`world.py`** — `Road1D`, a Nagel–Schreckenberg cellular-automaton traffic model
  (NumPy, periodic boundary conditions).
- **`main.py`** — runs the simulation and renders a space-time diagram
  (`spacetime.png`).
- **`agent.py`** — `QTableAgent`, a tabular Q-learning agent for traffic-signal
  control (bucketized queue-count states, green-phase actions, waiting-time reward).
