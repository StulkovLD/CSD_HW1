import pandas as pd


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


def count_iqr_outliers(series: pd.Series):
    """
    Считает число выбросов
    """

    _, _, _, lf, uf = compute_iqr_bounds(series)

    mask = (series < lf) | (series > uf)
    outlier_count = mask.sum()

    return outlier_count
