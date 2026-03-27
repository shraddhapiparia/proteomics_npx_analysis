#!/usr/bin/env python3

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from functools import reduce


def main(args):
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(args.input)

    # Example placeholders:
    # - customize these based on your selected UKB COVID and symptom fields
    # - current symptom fields coded as 1 = yes
    # - duration fields coded as 4 = >12 weeks

    symptom_pairs = [
        # ("participant.p28615", "participant.p28616"),
        # ("participant.p28609", "participant.p28610"),
    ]

    if symptom_pairs:
        current_exprs = [(col(cur) == 1) for cur, dur in symptom_pairs]
        duration_exprs = [(col(dur) == 4) for cur, dur in symptom_pairs]

        any_current = reduce(lambda a, b: a | b, current_exprs)
        any_long = reduce(lambda a, b: a | b, duration_exprs)

        df = df.withColumn(
            "long_covid_symptom_flag",
            when(any_current & any_long, lit(1)).otherwise(lit(0))
        )
    else:
        df = df.withColumn("long_covid_symptom_flag", lit(None))

    # Example COVID flag placeholder
    # Replace with your actual logic from self-report / antibody / test fields
    df = df.withColumn("covid_history_flag", lit(None))

    df.write.mode("overwrite").parquet(args.out)
    print(f"Saved grouped dataset to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define COVID and Long COVID subgroup labels from merged UKB phenotype-proteomics data."
    )
    parser.add_argument("--input", required=True, help="Merged parquet input")
    parser.add_argument("--out", required=True, help="Grouped parquet output")
    args = parser.parse_args()

    main(args)