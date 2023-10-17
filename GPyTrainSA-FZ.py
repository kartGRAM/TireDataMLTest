import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import GPy
import numpy as np
from typing import Any


def showData(df: pl.DataFrame, thinout: int = None):
    # 概要を表示
    print(df.columns)
    print(df.describe())

    # データを間引く（表示高速化）
    dfds = df
    if thinout is not None:
        dfds = df[::thinout]
    sns.set()

    _, axes2D = plt.subplots(2, 3, figsize=(50, 50))
    axes = axes2D.ravel()
    sns.lineplot(x="ET", y="TSTM", data=dfds, ax=axes[0])
    sns.scatterplot(x="SA", y="FY", data=dfds, ax=axes[1])
    sns.scatterplot(x="SA", y="MZ", data=dfds, ax=axes[2])
    sns.scatterplot(x="FZ", y="FY", data=dfds, ax=axes[3])
    sns.scatterplot(x="P", y="FY", data=dfds, ax=axes[4])
    sns.scatterplot(x="IA", y="FY", data=dfds, ax=axes[5])

    return axes


def showPred(
    axes: np.ndarray[Any, Any],
    axis: int,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
):
    if X is not None and Y is not None:
        df = pl.from_numpy(
            np.concatenate([X, Y], axis=1),
            schema=["SA", "FY-pred"],
            orient="row",
        )
        sns.lineplot(x="SA", y="FY-pred", data=df, ax=axes[axis])


def lbToN(lb: float):
    return lb * 4.4482189159


def psiToKPa(psi: float):
    return psi * 6.89476


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
meanTemp = ((pl.col("TSTC") + pl.col("TSTI") + pl.col("TSTO")) / 3).alias(
    "TSTM"
)
# 輪荷重は鉛直下向きが負なので、鉛直下向きを正にしてしまう
invFZ = pl.col("FZ") * -1
# タイヤ温度が平均して50℃以下のデータ捨てる
df = df.with_columns(meanTemp, invFZ).filter(pl.col("TSTM") > 50)

# テスト用
df = (
    df.filter(pl.col("IA") < 0.5)
    # .filter(pl.col("P") < 60)
    # .filter(pl.col("FZ") < 800)
)


# SA IA FZ P FYを取得
X = df.select("SA", "FZ", "P")
Y = df.select("FY")
print(X.describe())
print(Y.describe())
X = X.to_numpy()
Y = Y.to_numpy()
Xmean = X.mean(axis=0)
Xstd = X.std(axis=0)
Ymean = Y.mean()
Ystd = Y.std()
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd

print(X.shape)
print(Y.shape)
# showData(df)
# plt.show()
# exit()

axes = showData(df, 10)

# 回帰曲線の作成
# 格子点を作成
points = np.stack(
    np.meshgrid(
        np.linspace(-2, 2, 21),  # SA
        np.linspace(-2, 2, 5),  # FZ
        np.linspace(-2, 2, 5),  # P
    ),
    axis=-1,
).reshape([-1, 3])
print(points.shape)


kernel = GPy.kern.RBF(3) + GPy.kern.White(3) + GPy.kern.Bias(3)
# m_sparse = GPy.models.SparseGPRegression(X, Y, kernel, Z=points)
m_sparse = GPy.models.GPRegression(X, Y, kernel)
# m_sparse.optimize(messages=True)
print(m_sparse.log_likelihood())

for fyLb in [50, 75, 100, 150, 250, 350]:
    # 予測点の作成
    xPred = np.array(
        [
            np.linspace(-2, 2, 100),
            (np.full(100, lbToN(fyLb) - Xmean[1]) / Xstd[1]),
            (np.full(100, psiToKPa(12) - Xmean[2]) / Xstd[2]),
        ]
    ).T

    yPred, sigma = m_sparse.predict(xPred)
    yPred = yPred * Ystd + Ymean
    showPred(
        axes, 1, (xPred[:, 0] * Xstd[0] + Xmean[0]).reshape([-1, 1]), yPred
    )

# FY予測点の作成
xPred = np.array(
    [
        np.full(100, -2),
        (np.linspace(0, lbToN(350), 100) - Xmean[1]) / Xstd[1],
        (np.full(100, psiToKPa(12) - Xmean[2]) / Xstd[2]),
    ]
).T
yPred, sigma = m_sparse.predict(xPred)
yPred = yPred * Ystd + Ymean
showPred(axes, 3, np.linspace(0, lbToN(350), 100).reshape([-1, 1]), yPred)


plt.show()
