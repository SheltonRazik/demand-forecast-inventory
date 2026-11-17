"""
Weekly inventory simulation.

Each week, per product and branch:
  1. stock is topped up to that week's target level (if stock is already above it, nothing is ordered)
  2. customers buy what they want, up to what is on the shelf. the rest is a lost sale.
  3. whatever is left carries into next week
"""

import numpy as np


def simulate_policy(targets, demand):
    """targets and demand are arrays shaped (weeks, series). Returns totals for the whole run."""
    weeks, n = demand.shape
    on_hand = np.zeros(n)
    sold = lost = held = 0.0
    for t in range(weeks):
        on_hand = np.maximum(on_hand, np.round(targets[t]))    # top up
        sales = np.minimum(on_hand, demand[t])
        lost += (demand[t] - sales).sum()
        sold += sales.sum()
        on_hand = on_hand - sales
        held += on_hand.sum()                                   # units left over at the end of the week
    return {"fill_rate": sold / demand.sum(), "lost_units": lost, "avg_leftover_units": held / weeks}
