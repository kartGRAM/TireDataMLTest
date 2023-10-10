import polars as pl

df = pl.read_csv(
    "TireData/Hoosier-LC0-Rim7/CorneringTest1.dat",
    skip_rows=1,
    skip_rows_after_header=1,
    separator="\t",
)

meanTemp = ((pl.col("TSTC") + pl.col("TSTI") + pl.col("TSTO")) / 3).alias("TSTM")
df = df.with_columns(meanTemp)
print(df.columns)
print(df.describe())
