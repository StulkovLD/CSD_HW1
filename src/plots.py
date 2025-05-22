import os
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def plot_qq_plot(series: pd.Series, dist: str = "norm", output_dir: str = "results") -> None:
    """
    Строит Q–Q plot ваших данных против теоретического распределения dist
    и сохраняет картинку в output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = series.dropna()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    stats.probplot(data, dist=dist, plot=ax)
    ax.set_title(f"Q–Q plot vs {dist}")
    ax.get_lines()[1].set_color('r')  # линия теоретического тренда красным

    fname = "qq-plot.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"QQ-plot сохранён: {path}")


def plot_pp_plot(series: pd.Series, dist: str = "norm", output_dir: str = "results") -> None:
    """
    Строит P–P plot: эмпирическая CDF vs теоретическая CDF(dist)
    и сохраняет картинку в output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = series.dropna().sort_values()
    n = len(data)

    # эмпирическая CDF
    emp_cdf = (pd.Series(range(1, n + 1), index=data.index) - 0.5) / n
    # теоретическая CDF
    rv = getattr(stats, dist)
    theo_cdf = rv.cdf(data)

    fig, ax = plt.subplots()
    ax.scatter(theo_cdf, emp_cdf, s=15)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_title(f"P–P plot vs {dist}")
    ax.set_xlabel("Теоретические вероятности")
    ax.set_ylabel("Эмпирические вероятности")
    ax.grid(True, which="both", linestyle=":")

    fname = "pp-plot.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"PP-plot сохранён: {path}")
