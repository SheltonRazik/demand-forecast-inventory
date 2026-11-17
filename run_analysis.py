"""
Forecast next week's demand for every product and branch, then turn the forecasts into
stocking decisions and measure what they would have cost.

Weeks 1-78 are for training. Weeks 79-104 are the test. Every forecast uses only
information from before the week it predicts.

Run:  python run_analysis.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
from features import FEATURES, build_features, repair_stockouts
from inventory import simulate_policy

BLUE, ORANGE, GREY = "#2E5FA3", "#C8691C", "#9A9A9A"
TEST_FROM = 78
CAT = [FEATURES.index(c) for c in ("sku_code", "branch_code", "category_code")]

sales = pd.read_csv("data/weekly_sales.csv")
truth = pd.read_csv("answer_key/true_demand.csv")
sales = repair_stockouts(sales).merge(truth, on=["week", "sku", "branch"])


def fit_predict(history_col, loss="poisson", quantile=None):
    """Train on weeks before TEST_FROM, predict the test weeks."""
    df = build_features(sales, history_col)
    train = df[(df["week"] >= 13) & (df["week"] < TEST_FROM)]
    test = df[df["week"] >= TEST_FROM]
    model = HistGradientBoostingRegressor(loss=loss, quantile=quantile, max_iter=300, learning_rate=0.05,
                                          categorical_features=CAT, random_state=0)
    # the model learns from recorded history. it never sees true demand.
    model.fit(train[FEATURES], train[history_col])
    return pd.Series(np.maximum(model.predict(test[FEATURES]), 0), index=test.index)


base = build_features(sales, "sales_repaired")
test = base[base["week"] >= TEST_FROM].copy()

# ---------------------------------------------------------------- forecasts
test["f_last4"] = build_features(sales, "sales").loc[test.index, "mean_4"]          # what a planner does by hand
test["f_ml_raw"] = fit_predict("sales")                                            # ML, stockouts ignored
test["f_ml"] = fit_predict("sales_repaired")                                       # ML, stockouts repaired
for q in (0.5, 0.8, 0.9, 0.95, 0.98):
    test[f"q{int(q * 100)}"] = fit_predict("sales_repaired", loss="quantile", quantile=q)


def wape(forecast):
    """Total absolute error as a share of total demand. Lower is better."""
    return (forecast - test["demand"]).abs().sum() / test["demand"].sum()


def bias(forecast):
    """Positive means forecasting too high overall, negative too low."""
    return (forecast - test["demand"]).sum() / test["demand"].sum()


acc = pd.DataFrame([
    {"forecast": "Average of last 4 weeks", "wape": wape(test["f_last4"]), "bias": bias(test["f_last4"])},
    {"forecast": "ML, stockouts ignored", "wape": wape(test["f_ml_raw"]), "bias": bias(test["f_ml_raw"])},
    {"forecast": "ML, stockouts repaired", "wape": wape(test["f_ml"]), "bias": bias(test["f_ml"])},
])
acc.to_csv("outputs/forecast_accuracy.csv", index=False)

by_cat = test.groupby("category").apply(lambda g: pd.Series({
    "last4_wape": (g["f_last4"] - g["demand"]).abs().sum() / g["demand"].sum(),
    "ml_wape": (g["f_ml"] - g["demand"]).abs().sum() / g["demand"].sum()}), include_groups=False)
by_cat.to_csv("outputs/accuracy_by_category.csv")

# is the 90% forecast really above demand 90% of the time?
cover = pd.DataFrame([{"target": q / 100, "actual": (test["demand"] <= test[f"q{q}"]).mean()} for q in (50, 80, 90, 95, 98)])
cover.to_csv("outputs/quantile_coverage.csv", index=False)

# ------------------------------------------------------- inventory policies
def grid(col):
    return test.pivot_table(index="week", columns=["sku", "branch"], values=col).values


demand = grid("demand")
mean4, std13 = grid("f_last4"), grid("std_13")
rows = []
# rule of thumb: stock some multiple of the recent average
for cover_weeks in (1.2, 1.5, 2.0, 2.5, 3.0):
    rows.append({"policy": "Rule of thumb", "setting": f"{cover_weeks}x recent average",
                 **simulate_policy(mean4 * cover_weeks, demand)})
# textbook: recent average plus safety stock sized by recent variability
for z in (0.5, 1.0, 1.5, 2.0, 2.5):
    rows.append({"policy": "Average + safety stock", "setting": f"z = {z}",
                 **simulate_policy(mean4 + z * std13, demand)})
# ML: stock up to a chosen quantile of the forecast
for q in (50, 80, 90, 95, 98):
    rows.append({"policy": "ML quantile forecast", "setting": f"{q}th percentile",
                 **simulate_policy(grid(f"q{q}"), demand)})
pol = pd.DataFrame(rows)
avg_cost = test["unit_cost"].mean()
pol["avg_leftover_value"] = pol["avg_leftover_units"] * avg_cost
pol.to_csv("outputs/inventory_policies.csv", index=False)


def stock_needed(policy, fill_target):
    """Leftover stock a policy needs to reach a fill rate, read off its curve."""
    p = pol[pol["policy"] == policy].sort_values("fill_rate")
    return float(np.interp(fill_target, p["fill_rate"], p["avg_leftover_units"]))


compare = pd.DataFrame([{"fill_rate": f, **{p: stock_needed(p, f) for p in pol["policy"].unique()}} for f in (0.90, 0.95)])
compare["ml_saving_vs_rule"] = 1 - compare["ML quantile forecast"] / compare["Rule of thumb"]
compare["ml_saving_vs_safety"] = 1 - compare["ML quantile forecast"] / compare["Average + safety stock"]
compare.to_csv("outputs/stock_needed_for_fill_rate.csv", index=False)

pd.set_option("display.width", 200)
print(acc.round(3).to_string(index=False), "\n")
print(by_cat.round(3).to_string(), "\n")
print(cover.round(3).to_string(index=False), "\n")
print(pol.round(3).to_string(index=False), "\n")
print(compare.round(3).to_string(index=False))

# ------------------------------------------------------------------- charts
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for policy, color in [("Rule of thumb", GREY), ("Average + safety stock", ORANGE), ("ML quantile forecast", BLUE)]:
    p = pol[pol["policy"] == policy].sort_values("avg_leftover_units")
    ax.plot(p["avg_leftover_units"] / 1000, p["fill_rate"] * 100, color=color, linewidth=2, marker="o", markersize=5, label=policy)
ax.set_xlabel("Average stock left on the shelf each week (thousand units)")
ax.set_ylabel("Fill rate (%)")
ax.set_title("Same service with less stock: up and to the left is better", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e6e6e6", linewidth=0.8)
fig.tight_layout()
fig.savefig("outputs/service_vs_stock.png", dpi=160)

# one product: actual demand against the forecast band
one = test[(test["category"] == "Pool Supply")].groupby(["sku", "branch"])["demand"].sum().idxmax()
s = test[(test["sku"] == one[0]) & (test["branch"] == one[1])]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(s["week"] + 1, s["q50"], s["q95"], color=BLUE, alpha=0.2, linewidth=0, label="Forecast: median to 95th percentile")
ax.plot(s["week"] + 1, s["q50"], color=BLUE, linewidth=2, label="Forecast: median")
ax.plot(s["week"] + 1, s["demand"], color=ORANGE, linewidth=2, label="Actual demand")
ax.set_xlabel("Week")
ax.set_ylabel("Units")
ax.set_title(f"{one[0]} (Pool Supply) at {one[1]}: test weeks", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
fig.tight_layout()
fig.savefig("outputs/forecast_example.png", dpi=160)
