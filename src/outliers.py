import pandas as pd
from matplotlib import cbook
from typing import Tuple


def print_bounds(bounds: tuple[float, float, float, float, float]) -> None:
    """
    Печатает таблицу с Q1, Q3, IQR и границами выбросов.
    """
    q1, q3, iqr, lf, uf = bounds
    header = f"{'Показатель':<15} {'Значение':>12}"
    sep = '-' * len(header)
    print(header)
    print(sep)
    for name, val in [
        ("Q1", q1),
        ("Q3", q3),
        ("IQR", iqr),
        ("Нижняя граница", lf),
        ("Верхняя граница", uf),
    ]:
        print(f"{name:<15} {val:>12.2f}")


def compute_iqr_bounds(series: pd.Series) -> tuple[float, float, float, float, float]:
    """
    Считает 1, 3 квантили, IQR и границы выбросов
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lf = q1 - 1.5 * iqr
    uf = q3 + 1.5 * iqr

    return q1, q3, iqr, lf, uf


def count_iqr_outliers(series: pd.Series) -> int:
    """
    Считает число выбросов
    """

    _, _, _, lf, uf = compute_iqr_bounds(series)

    mask = (series < lf) | (series > uf)
    outlier_count = mask.sum()

    return outlier_count


def count_z_score_outliers(series: pd.Series) -> int:
    """
    Считает количество выбросов по Z-score.
    """
    mean_value = series.mean()
    standart_deviation = series.std(ddof=0)
    z_i = (series - mean_value) / standart_deviation
    count = (z_i.abs() > 3).sum().astype(int)
    return count


def compute_boxplot_bounds(series: pd.Series, whisker_coef: float = 1.5) -> tuple[float, float, float, float, float]:
    """
    Считает по правилу «ящик с усами»:
    """
    stats = cbook.boxplot_stats(series, whis=whisker_coef)[0]

    q1 = stats['q1']
    q3 = stats['q3']
    iqr = stats['iqr']

    # здесь — именно «теория»
    lf_theoretical = q1 - whisker_coef * iqr
    uf_theoretical = q3 + whisker_coef * iqr

    return q1, q3, iqr, lf_theoretical, uf_theoretical


def count_boxplot_outliers(series: pd.Series, whisker_coef: float = 1.5) -> int:
    """
    Считает число выбросов по правилу «ящик с усами», используя тот же алгоритм, что и matplotlib.
    """
    stats = cbook.boxplot_stats(series, whis=whisker_coef)[0]
    return len(stats['fliers'])


def classify_iqr_outliers(
    series: pd.Series,
    moderate_coef: float = 2.0,
    extreme_coef: float = 3.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Классифицирует выбросы на «умеренные» и «экстремальные» по отклонению
    от среднего в единицах IQR.
    """
    q1, q3, iqr, _, _ = compute_iqr_bounds(series)
    mean_val = series.mean()

    # Абсолютное отклонение от среднего
    deviation = (series - mean_val).abs()

    # Маски
    moderate_mask = (deviation >= moderate_coef * iqr) & (deviation <= extreme_coef * iqr)
    extreme_mask = deviation > extreme_coef * iqr

    # Возвращаем сами выбросы
    moderate_outliers = series[moderate_mask]
    extreme_outliers = series[extreme_mask]
    return moderate_outliers, extreme_outliers
