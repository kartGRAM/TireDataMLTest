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
            "TireData/TestData/CorneringTest1.dat",
            skip_rows=1,
            skip_rows_after_header=1,
            separator="\t",
        ),
        pl.read_csv(
            "TireData/TestData/CorneringTest2.dat",
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
# タイヤ温度が平均して50℃以下のデータはウォームアップが足りてないと判断して捨てる
df = df.with_columns(meanTemp, invFZ).filter(pl.col("TSTM") > 50)

# 必要なデータを抽出
df = (
    df.filter(pl.col("IA") < 0.5)
    # .filter(pl.col("P") < 60)
    # .filter(pl.col("FZ") < 800)
)


# SA IA FZ P FYを取得データ点数が多いと計算できないので適当に間引く（4000点くらい?）
X = df.select("SA", "FZ", "P")[::10]
Y = df.select("FY")[::10]
print(X.describe())
print(Y.describe())
X = X.to_numpy()
Y = Y.to_numpy()
Xmean = X.mean(axis=0)
Xstd = X.std(axis=0)
Ymean = Y.mean()
Ystd = Y.std()


# 正規化する
def normalizeX(X: np.ndarray):
    return (X - Xmean) / Xstd


X = normalizeX(X)
Y = (Y - Ymean) / Ystd

print(X.shape)
print(Y.shape)

axes = showData(df, 1)

# 回帰曲線を求める
# 補助点を作成
points = np.stack(
    np.meshgrid(
        np.linspace(-2, 2, 21),  # SAは点数多めじゃないとうまく表現できなかった
        np.linspace(-2, 2, 10),  # FZ
        np.linspace(-2, 2, 10),  # P
    ),
    axis=-1,
).reshape([-1, 3])
print(points.shape)

# バイアスはたぶん入れたほうがいい感じになる。
kernel = GPy.kern.RBF(3) + GPy.kern.White(3) + GPy.kern.Bias(3)

# データの点数が多い場合はこちらのほうが速いらしい。
m_sparse = GPy.models.SparseGPRegression(X, Y, kernel, Z=points)
# 　こっちを使う場合は通常のガウス過程回帰
# _sparse = GPy.models.GPRegression(X, Y, kernel)

# カーネルのハイパーパラメータを最適化(やらなくてもよい)
m_sparse.optimize(messages=True)

for fyLb in [50, 75, 100, 150, 250, 350]:
    # SA-FY予測の作成
    xPred = normalizeX(
        np.array(
            [
                np.linspace(-14, 14, 100),
                np.full(100, lbToN(fyLb)),
                np.full(100, psiToKPa(12)),
            ]
        ).T
    )

    yPred, sigma = m_sparse.predict(xPred)
    yPred = yPred * Ystd + Ymean
    showPred(
        axes, 1, (xPred[:, 0] * Xstd[0] + Xmean[0]).reshape([-1, 1]), yPred
    )

# FZ-FY予測の作成
xPred = normalizeX(
    np.array(
        [
            np.full(100, -12),
            np.linspace(0, lbToN(350), 100),
            np.full(100, psiToKPa(12)),
        ]
    ).T
)
yPred, sigma = m_sparse.predict(xPred)
yPred = yPred * Ystd + Ymean
showPred(axes, 3, np.linspace(0, lbToN(350), 100).reshape([-1, 1]), yPred)

# P-FY予測の作成
xPred = normalizeX(
    np.array(
        [
            np.full(100, -12),
            np.full(100, 1100),
            np.linspace(psiToKPa(7), psiToKPa(15), 100),
        ]
    ).T
)
yPred, sigma = m_sparse.predict(xPred)
yPred = yPred * Ystd + Ymean
showPred(
    axes, 4, np.linspace(psiToKPa(7), psiToKPa(15), 100).reshape([-1, 1]), yPred
)

plt.show()
