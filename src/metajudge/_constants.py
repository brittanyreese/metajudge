"""Package-internal numeric constants shared across pillars."""

from __future__ import annotations

# Bootstrap-reliability floor: below this many surviving resamples a 2.5/97.5 percentile CI
# is too thin to trust. Both pillars gate their ``ci_reliable`` on it (alpha in
# reliability.py, DIF in dif.py) so a single edit moves both floors together. Note the floor
# is only the shared *convergence* gate: each pillar may layer its own extra checks on top
# (e.g. report.py adds a drop-fraction tolerance for alpha; dif.py adds a min cluster size).
MIN_EFFECTIVE = 100
