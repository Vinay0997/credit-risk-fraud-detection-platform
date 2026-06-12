"""Optional PySpark feature engineering pipeline.

This module mirrors how the local pandas demo can be lifted into a distributed
processing job for behavioral and transaction signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_spark_features(input_path: str, output_path: str) -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, hour, to_timestamp, when
    except Exception as exc:
        raise RuntimeError("PySpark is required. Install requirements-full.txt.") from exc

    spark = SparkSession.builder.appName("credit-risk-fraud-features").getOrCreate()
    frame = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

    featured = (
        frame.withColumn("event_ts", to_timestamp(col("event_timestamp")))
        .withColumn("hour_of_day", hour(col("event_ts")))
        .withColumn(
            "night_transaction_flag",
            when((col("hour_of_day") <= 5) | (col("hour_of_day") >= 23), 1).otherwise(0),
        )
        .withColumn(
            "high_velocity_flag",
            when((col("login_velocity_1h") >= 6) | (col("geo_velocity_km") >= 500), 1).otherwise(0),
        )
        .withColumn(
            "high_utilization_flag",
            when(col("credit_utilization") >= 0.75, 1).otherwise(0),
        )
    )

    featured.write.mode("overwrite").parquet(output_path)
    spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/spark_features"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_spark_features(str(args.input), str(args.output))
    print(f"Wrote Spark features to {args.output}")


if __name__ == "__main__":
    main()
