from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb


# Must match your EDA notebook feature order exactly
EDA_FEATURES = [
    "hs_dist",
    "tms_drop_distance",
    "plz_drop_distance",
    "met_drop_distance",
    "hbk_drop_distance",
    "pickup_longitude",
    "nyc_drop_distance",
    "wtc_drop_distance",
    "dropoff_longitude",
    "sol_drop_distance",
    "ewr_drop_distance",
    "lga_drop_distance",
    "passenger_count",
    "dropoff_latitude",
    "pickup_latitude",
    "jfk_drop_distance",
    "hour",
]

# Exactly from Gold.ipynb (lon, lat)
SIGHTS_LONLAT: Dict[str, Tuple[float, float]] = {
    "jfk": (-73.7781, 40.6413),
    "lga": (-73.8740, 40.7769),
    "ewr": (-74.1745, 40.6895),
    "met": (-73.9632, 40.7794),
    "wtc": (-74.0099, 40.7126),
    "sol": (-74.0445, 40.6892),
    "nyc": (-74.0063889, 40.7141667),
    "tms": (-73.9854406, 40.7581047),
    "plz": (-73.9750593, 40.7651662),
    "hbk": (-74.0282202, 40.7356908),
    # "brb": (-73.9741970, 40.5882819),  # exists in Gold, not used in EDA_FEATURES
}

R_KM = 6378.0  # exactly as in Gold.ipynb


def _parse_datetime(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x
    if hasattr(x, "to_pydatetime"):  # pandas Timestamp
        return x.to_pydatetime()
    if isinstance(x, str):
        s = x.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    raise TypeError(f"Unsupported pickup_datetime type: {type(x)}")


def haversine_km_lonlat(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    R: float = R_KM,
) -> np.ndarray:
    """
    Matches Gold.ipynb formula:
      a = sin(dlat/2)^2 + cos(lat1)*cos(lat2)*sin(dlon/2)^2
      c = 2*asin(sqrt(a))
      d = R*c
    lon/lat are in degrees.
    """
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)

    dlon = lon2r - lon1r
    dlat = lat2r - lat1r

    a = (np.sin(dlat / 2.0) ** 2) + (np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2))
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def build_features(raw: Union[Mapping[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Build the exact EDA_FEATURES vector using the Gold.ipynb logic.
    Required raw fields:
      pickup_datetime, pickup_longitude, pickup_latitude,
      dropoff_longitude, dropoff_latitude, passenger_count
    """
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame([dict(raw)])

    required = [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # numeric conversions
    for c in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]].isna().any().any():
        raise ValueError("Some coordinates became NaN after numeric conversion.")

    # hour
    dt = df["pickup_datetime"].map(_parse_datetime)
    df["hour"] = dt.map(lambda d: d.hour).astype(int)

    # hs_dist: pickup -> dropoff
    df["hs_dist"] = haversine_km_lonlat(
        lon1=df["pickup_longitude"].to_numpy(dtype=float),
        lat1=df["pickup_latitude"].to_numpy(dtype=float),
        lon2=df["dropoff_longitude"].to_numpy(dtype=float),
        lat2=df["dropoff_latitude"].to_numpy(dtype=float),
    )

    # *_drop_distance: sight -> dropoff (Gold uses sight as point1, dropoff as point2)
    drop_lon = df["dropoff_longitude"].to_numpy(dtype=float)
    drop_lat = df["dropoff_latitude"].to_numpy(dtype=float)

    for name in ["tms", "plz", "met", "hbk", "nyc", "wtc", "sol", "ewr", "lga", "jfk"]:
        sight_lon, sight_lat = SIGHTS_LONLAT[name]
        df[f"{name}_drop_distance"] = haversine_km_lonlat(
            lon1=np.full_like(drop_lon, sight_lon, dtype=float),
            lat1=np.full_like(drop_lat, sight_lat, dtype=float),
            lon2=drop_lon,
            lat2=drop_lat,
        )

    X = df.reindex(columns=EDA_FEATURES)

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs in engineered features: {bad_cols}")

    return X


def load_booster(checkpoint_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=checkpoint_path)


def predict_fare(
    raw: Union[Mapping[str, Any], pd.DataFrame],
    checkpoint_path: str,
) -> Union[float, np.ndarray]:
    booster = load_booster(checkpoint_path)
    X = build_features(raw)
    preds = booster.predict(X)
    if isinstance(raw, pd.DataFrame):
        return preds
    return float(preds[0])


if __name__ == "__main__":
    ckpt = "/Volumes/final_project/default/files/lgb_model.txt"

    sample = {
        "pickup_datetime": "2013-07-06 17:18:00",
        "pickup_longitude": -73.9822,
        "pickup_latitude": 40.7612,
        "dropoff_longitude": -73.9995,
        "dropoff_latitude": 40.7320,
        "passenger_count": 1,
    }

    print("Predicted fare_amount:", predict_fare(sample, ckpt))