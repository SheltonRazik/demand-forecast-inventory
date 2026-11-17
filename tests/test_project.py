import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import build_features, repair_stockouts
from inventory import simulate_policy


def small_frame(sales, stockout=None):
    n = len(sales)
    return pd.DataFrame({"week": range(n), "sku": "A", "branch": "X", "category": "C", "promo": 0,
                         "stockout": stockout or [0] * n, "sales": sales})


def test_lags_only_use_the_past():
    df = build_features(small_frame([10, 20, 30, 40, 50, 60]), "sales")
    assert df.loc[5, "lag_1"] == 50
    assert df.loc[5, "mean_4"] == (20 + 30 + 40 + 50) / 4      # does not include week 5 itself


def test_stockout_week_is_raised_to_recent_average():
    df = repair_stockouts(small_frame([20, 20, 20, 20, 5, 20], stockout=[0, 0, 0, 0, 1, 0]))
    assert df.loc[4, "sales_repaired"] == 20
    assert df.loc[5, "sales_repaired"] == 20


def test_repair_never_lowers_sales():
    df = repair_stockouts(small_frame([5, 5, 5, 5, 30, 5], stockout=[0, 0, 0, 0, 1, 0]))
    assert df.loc[4, "sales_repaired"] == 30


def test_inventory_by_hand():
    # one product, three weeks. target 10 each week, demand 6, 15, 4
    r = simulate_policy(np.array([[10.0], [10.0], [10.0]]), np.array([[6.0], [15.0], [4.0]]))
    # week 1: sell 6, 4 left. week 2: top up to 10, sell 10, lose 5, 0 left. week 3: sell 4, 6 left.
    assert r["lost_units"] == 5
    assert np.isclose(r["fill_rate"], 20 / 25)
    assert np.isclose(r["avg_leftover_units"], (4 + 0 + 6) / 3)


def test_more_stock_never_lowers_fill_rate():
    rng = np.random.default_rng(0)
    demand = rng.poisson(20, size=(26, 50)).astype(float)
    low = simulate_policy(np.full((26, 50), 20.0), demand)["fill_rate"]
    high = simulate_policy(np.full((26, 50), 30.0), demand)["fill_rate"]
    assert high >= low
