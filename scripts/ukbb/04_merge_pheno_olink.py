#!/usr/bin/env python3

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd


def standardize_eid(df, candidates):
    for c in candidates:
        if c in df.columns:
            if c != "eid":
                return df.withColumnRenamed(c, "eid")
            return df
    raise ValueError(f"No eid column found. Available columns: {df.columns}")


def main(args):
    spark = SparkSession.builder.getOrCreate()

    with open(args.pheno_sql, "r") as f:
        pheno_sql = f.read()

    with open(args.olink_sql, "r") as f:
        olink_sql = f.read()

    pheno_df = spark.sql(pheno_sql)
    olink_df = spark.sql(olink_sql)

    pheno_df = standardize_eid(pheno_df, ["participant.eid", "eid"])
    olink_df = standardize_eid(olink_df, ["olink_instance_0.eid", "participant.eid", "eid"])

    eids_pd = pd.read_csv(args.eids, sep="\t")
    if "participant.eid" in eids_pd.columns:
        eids_pd = eids_pd.rename(columns={"participant.eid": "eid"})

    eids_pd = eids_pd[["eid"]].drop_duplicates()
    eids_pd["eid"] = eids_pd["eid"].astype(str)

    eids_df = spark.createDataFrame(eids_pd)

    pheno_df = pheno_df.withColumn("eid", col("eid").cast("string"))
    olink_df = olink_df.withColumn("eid", col("eid").cast("string"))
    eids_df = eids_df.withColumn("eid", col("eid").cast("string"))

    pheno_filt = pheno_df.join(eids_df, on="eid", how="inner")
    merged_df = pheno_filt.join(olink_df, on="eid", how="inner")

    print("pheno rows:", pheno_df.count())
    print("olink rows:", olink_df.count())
    print("selected eid rows:", eids_df.count())
    print("filtered pheno rows:", pheno_filt.count())
    print("merged rows:", merged_df.count())
    print("merged unique eids:", merged_df.select("eid").distinct().count())
    print("merged cols:", len(merged_df.columns))

    merged_df.write.mode("overwrite").parquet(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheno-sql", required=True)
    parser.add_argument("--olink-sql", required=True)
    parser.add_argument("--eids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)