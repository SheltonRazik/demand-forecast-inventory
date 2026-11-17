"""Turns the weekly sales table into one row per product, branch and week with only past information."""

import numpy as np
import pandas as pd

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "mean_4", "mean_13", "std_13",
            "promo", "promo_last_week", "week_sin", "week_cos", "sku_code", "branch_code", "category_code"]


def repair_stockouts(df):
    """
    A stockout week understates demand. Replace its sales with the larger of what was sold
    and the average of the last four normal weeks. It is an estimate, not the truth.
    """
    df = df.sort_values(["sku", "branch", "week"]).copy()
    normal = df["sales"].where(df["stockout"] == 0)
    recent = normal.groupby([df["sku"], df["branch"]]).transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df["sales_repaired"] = np.where(df["stockout"] == 1, np.maximum(df["sales"], recent.fillna(df["sales"])), df["sales"])
    return df


def build_features(df, history_col):
    """history_col is the column the lags are built from: 'sales' (raw) or 'sales_repaired'."""
    df = df.sort_values(["sku", "branch", "week"]).copy()
    g = df.groupby(["sku", "branch"])[history_col]
    for k in (1, 2, 3, 4, 52):
        df[f"lag_{k}"] = g.shift(k)
    df["mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["mean_13"] = g.transform(lambda s: s.shift(1).rolling(13).mean())
    df["std_13"] = g.transform(lambda s: s.shift(1).rolling(13).std())
    df["promo_last_week"] = df.groupby(["sku", "branch"])["promo"].shift(1)
    df["week_sin"] = np.sin(2 * np.pi * (df["week"] % 52) / 52)
    df["week_cos"] = np.cos(2 * np.pi * (df["week"] % 52) / 52)
    for col in ("sku", "branch", "category"):
        df[f"{col}_code"] = df[col].astype("category").cat.codes
    return df
