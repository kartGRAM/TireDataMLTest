import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import GPy
import numpy as np


def showData(df: pl.DataFrame):
    # 概要を表示
    print(df.columns)
    print(df.describe())

    # データを間引く（表示高速化）
    dfds = df[::100]
    sns.set()

    fig, axes = plt.subplots(2, 3, figsize=(50, 50))
    axes = axes.ravel()
    sns.lineplot(x="ET", y="TSTM", data=dfds, ax=axes[0])
    sns.scatterplot(x="SA", y="FY", data=dfds, ax=axes[1])
    sns.scatterplot(x="SA", y="MZ", data=dfds, ax=axes[2])
    sns.scatterplot(x="FZ", y="FY", data=dfds, ax=axes[3])
    sns.scatterplot(x="P", y="FY", data=dfds, ax=axes[4])
    sns.scatterplot(x="IA", y="FY", data=dfds, ax=axes[5])

    plt.show()


df = pl.concat(
    [
        pl.read_csv(
            "TireData/Hoosier-LC0-Rim7/CorneringTest1.dat",
            skip_rows=1,
            skip_rows_after_header=1,
            separator="\t",
        ),
        pl.read_csv(
            "TireData/Hoosier-LC0-Rim7/CorneringTest2.dat",
            skip_rows=1,
            skip_rows_after_header=1,
            separator="\t",
        ),
    ]
)

# 平均タイヤ温度を算出
meanTemp = ((pl.col("TSTC") + pl.col("TSTI") + pl.col("TSTO")) / 3).alias("TSTM")
# 輪荷重は鉛直下向きが負なので、鉛直下向きを正にしてしまう
invFZ = pl.col("FZ") * -1
# タイヤ温度が平均して50℃以下のデータ捨てる
df = df.with_columns(meanTemp, invFZ).filter(pl.col("TSTM") > 50)


# showData(df)


m_sparse = GPy.models.SparseGPRegression()
m_sparse.load_model("output.dat")
m_sparse.p
