import os
import gc
from pathlib import Path
from glob import glob
from datetime import datetime
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm
from shapely.geometry import LineString

gc.enable()

EARTH_RADIUS_KM = 6371.009
TMP_DIR = "tmp_geolife"

def extract_user_id(data_path: str) -> int:
    return int(Path(data_path).parts[-3])

def haversine_polars():
    """
    Fully vectorized haversine expression in Polars (no Python UDF).
    """
    return (
        2
        * EARTH_RADIUS_KM
        * (
            (
                ((pl.col("lat2_rad") - pl.col("lat1_rad")) / 2).sin().pow(2)
                + pl.col("lat1_rad").cos()
                * pl.col("lat2_rad").cos()
                * ((pl.col("lon2_rad") - pl.col("lon1_rad")) / 2).sin().pow(2)
            )
            .clip(0, 1)
            .sqrt()
            .arcsin()
        )
    )

def process_data(path: str, resample: str | None = "10s") -> str:
    user_id = extract_user_id(path)

    df = pl.read_csv(
        path,
        skip_rows=6,
        columns=[0, 1, 5, 6],
        new_columns=["latitude", "longitude", "date_str", "time_str"],
        schema_overrides={
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "date_str": pl.String,
            "time_str": pl.String,
        },
    )

    # ----------------------------
    # Timestamp
    # ----------------------------
    df = (
        df.with_columns(
            (
                pl.col("date_str")
                + pl.lit(" ")
                + pl.col("time_str")
            )
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .alias("timestamp")
        )
        .drop(["date_str", "time_str"])
        .with_columns(pl.lit(user_id).alias("user_id"))
    )

    # ----------------------------
    # Resample (optional)
    # ----------------------------
    if resample:
        df = (
            df.sort("timestamp")
            .set_sorted("timestamp")
            .group_by_dynamic("timestamp", every=resample)
            .agg(
                pl.col("latitude").mean(),
                pl.col("longitude").mean(),
                pl.col("user_id").first(),
            )
        )

    # ----------------------------
    # Sort ONCE (critical)
    # ----------------------------
    df = df.sort(["user_id", "timestamp"])

    # ----------------------------
    # Shift-based trajectory build
    # ----------------------------
    df = df.with_columns([
        pl.col("timestamp").shift(-1).over("user_id").alias("end_timestamp"),
        pl.col("latitude").shift(-1).over("user_id").alias("end_latitude"),
        pl.col("longitude").shift(-1).over("user_id").alias("end_longitude"),
    ]).drop_nulls(["end_timestamp", "end_latitude", "end_longitude"])

    # ----------------------------
    # Vectorized haversine
    # ----------------------------
    df = df.with_columns([
        pl.col("latitude").radians().alias("lat1_rad"),
        pl.col("longitude").radians().alias("lon1_rad"),
        pl.col("end_latitude").radians().alias("lat2_rad"),
        pl.col("end_longitude").radians().alias("lon2_rad"),
    ])

    df = df.with_columns(
        haversine_polars().alias("distance_km")
    ).drop(["lat1_rad", "lon1_rad", "lat2_rad", "lon2_rad"])

    # ----------------------------
    # Time delta (hours)
    # ----------------------------
    df = df.with_columns(
        (
            (pl.col("end_timestamp") - pl.col("timestamp"))
            .dt.total_seconds()
            / 3600.0
        ).alias("time_delta_hr")
    ).filter(pl.col("time_delta_hr") > 0)

    # ----------------------------
    # Speed
    # ----------------------------
    df = df.with_columns(
        (pl.col("distance_km") / pl.col("time_delta_hr")).alias("speed_kmh")
    )

    # ----------------------------
    # Filter unrealistic movement
    # ----------------------------
    df = df.filter(
        (pl.col("speed_kmh") >= 5)
        & (pl.col("speed_kmh") <= 100)
    )

    # ----------------------------
    # Write per-worker parquet
    # ----------------------------
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"{user_id}_{Path(path).stem}.parquet")
    df.write_parquet(out_path)

    return out_path

def trip_to_line(row) -> LineString:
    return LineString([
        (row["longitude"], row["latitude"]),
        (row["end_longitude"], row["end_latitude"]),
    ])


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    start = datetime.now()

    data_dir = r"C:\Users\andrr\Documents\dev\data\geolife\Data"
    files = glob(os.path.join(data_dir, "*", "*", "*.plt"))

    results = []

    with tqdm(total=len(files), desc="Processing") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_data, f, "10s")
                for f in files
            ]

            for f in as_completed(futures):
                results.append(f.result())
                pbar.update(1)

    # ----------------------------
    # Lazy final dataset (NO concat)
    # ----------------------------
    lf = pl.scan_parquet(os.path.join(TMP_DIR, "*.parquet"))
    df = lf.collect()

    print(df.shape)
    print(df.head())

    # ----------------------------
    # OPTIONAL: build GeoDataFrame only if needed
    # ----------------------------
    # (avoid this for large-scale processing)
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        df.to_pandas(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    print(gdf.head())

    end = datetime.now()
    print(f"Total processing time: {end - start}")
    
    os.remove(TMP_DIR)