"""
Builds two years of weekly sales for a made-up distributor: 40 products in 4 branches.

Demand has trend, yearly seasonality that depends on the product category, and promotions.
On top of that, some weeks the branch ran out of stock. In those weeks the sales number is
lower than what customers actually wanted. The true demand is saved in answer_key/ so the
forecasts can be judged against what really happened, not just what got recorded.

Run:  python generate_sales.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(72)
WEEKS = 104
BRANCHES = ["Manahawkin", "Newark", "Camden", "Allentown"]
# category -> (week of the year when demand peaks, how strong the season is)
CATEGORIES = {"Pool Supply": (27, 0.9), "HVAC": (3, 0.6), "Landscape": (20, 0.7), "Plumbing": (0, 0.0),
              "Paint": (22, 0.3), "Safety": (0, 0.0), "Roofing": (36, 0.5), "Fasteners": (0, 0.05)}

rows = []
sku = 0
for category, (peak, strength) in CATEGORIES.items():
    for _ in range(5):
        sku += 1
        base = rng.lognormal(3.3, 0.7)                 # typical weekly units
        trend = rng.normal(0.0, 0.002)                 # growth per week
        promo_lift = rng.uniform(1.3, 2.2)
        unit_cost = round(float(rng.lognormal(2.8, 0.6)), 2)
        for branch in BRANCHES:
            size = rng.uniform(0.5, 1.6)
            noise = rng.uniform(0.15, 0.45)            # some products are much less predictable
            promo = rng.random(WEEKS) < 0.08
            for week in range(WEEKS):
                season = 1 + strength * np.cos(2 * np.pi * (week % 52 - peak) / 52)
                mean = base * size * season * (1 + trend) ** week * (promo_lift if promo[week] else 1)
                # negative binomial: like Poisson but with extra spread
                shape = 1 / noise ** 2
                demand = rng.poisson(rng.gamma(shape, mean / shape))
                rows.append({"week": week, "sku": f"SKU-{sku:03d}", "category": category, "branch": branch,
                             "promo": int(promo[week]), "unit_cost": unit_cost, "demand": demand})

df = pd.DataFrame(rows)
df["week_start"] = pd.Timestamp("2024-11-04") + pd.to_timedelta(df["week"] * 7, unit="D")

# stockouts: about 6% of weeks, more likely in busy weeks. sales are capped at what was on the shelf.
busy = df["demand"] / df.groupby(["sku", "branch"])["demand"].transform("mean")
p_out = np.clip(0.03 + 0.04 * (busy - 1), 0.01, 0.25)
df["stockout"] = (rng.random(len(df)) < p_out).astype(int)
df["sales"] = np.where(df["stockout"] == 1, np.floor(df["demand"] * rng.uniform(0.3, 0.8, len(df))), df["demand"]).astype(int)

cols = ["week_start", "week", "sku", "category", "branch", "promo", "stockout", "unit_cost", "sales"]
df[cols].to_csv("data/weekly_sales.csv", index=False)
df[["week", "sku", "branch", "demand"]].to_csv("answer_key/true_demand.csv", index=False)
lost = (df["demand"] - df["sales"]).sum() / df["demand"].sum()
print(f"rows: {len(df):,}  series: {df.groupby(['sku', 'branch']).ngroups}  stockout weeks: {df['stockout'].mean():.1%}  demand hidden by stockouts: {lost:.1%}")
