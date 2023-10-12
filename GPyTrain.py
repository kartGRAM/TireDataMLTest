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

df = df[::30]

# showData(df)

# SA IA FZ P FYを取得
X = df.select("SA", "IA", "FZ", "P")
Y = df.select("FY")
print(X.describe())
print(Y.describe())
X = X.to_numpy()
Y = Y.to_numpy()
print(X.shape)
print(Y.shape)
# showData(df)

# 格子点を作成
points = np.stack(
    np.meshgrid(
        np.linspace(-15, 15, 5),  # SA
        np.arange(5),  # IA
        np.linspace(0, 1500, 5),  # FZ
        np.linspace(40, 110, 5),  # P
    ),
    axis=-1,
).reshape([-1, 4])
print(points.shape)

# exit()

kernel = GPy.kern.RBF(4)
m_sparse = GPy.models.SparseGPRegression(X, Y, kernel, Z=points)
m_sparse.optimize()
m_sparse.save_model("output.dat")
print(m_sparse.log_likelihood())
