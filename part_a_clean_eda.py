"""Part A: clean the raw export, write the modelling table and the delay evidence."""
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

dedup = raw.drop_duplicates()
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

h = table["delivery_hours"]
q1, q3 = h.quantile(0.25), h.quantile(0.75)
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
metrics = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": int((dedup["parcel_weight_kg"] > 100).sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
    "service_type_spellings_after": int(clean["service_type"].nunique()),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": np.log(table["declared_value_inr"]).mean(),
    "log_declared_value_median": np.log(table["declared_value_inr"]).median(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(table["is_delayed"].sum()),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": int(((h < lo) | (h > hi)).sum()),
}
pd.DataFrame({"metric": list(metrics),
              "value": [v if isinstance(v, (int, np.integer)) else round(float(v), 4) for v in metrics.values()]}
             , dtype=object).to_csv("outputs/part_a_checks.csv", index=False)

table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
parts = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"),
                                              mean_hours=("delivery_hours", "mean")).reset_index()
    g.insert(0, "dimension", dim)
    parts.append(g.rename(columns={dim: "group"}))
pd.concat(parts, ignore_index=True).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

table.drop(columns=["distance_band"]).select_dtypes(include="number").corr().round(4).to_csv(
    "outputs/part_a_correlations.csv")
print("part A done:", len(table), "rows")
