"""Shared constants and preprocessing used by every Part and by predict_delay.py."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_THRESHOLD = 100


def clean_shipments(raw: pd.DataFrame, medians: Optional[Dict[str, float]] = None
                    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Clean a raw shipments export. Returns (clean_df, medians)."""
    df = raw.drop_duplicates().copy()                                        # 1
    df["service_type"] = df["service_type"].str.strip().str.title()          # 2
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")                # 3
    value_text = df["declared_value"].astype("string").str.replace("\u20b9", "", regex=False)
    value_text = value_text.str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype(float)  # 4
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])                # 5
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_THRESHOLD, "parcel_weight_kg"] = np.nan  # 6
    if medians is None:                                                      # 7
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])
    return df.reset_index(drop=True), medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub columns and derived features. Never touches delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
