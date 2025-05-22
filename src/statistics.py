import pandas as pd
from typing import Tuple
from src.outliers import classify_iqr_outliers


def compute_structural_stats(series: pd.Series) -> Tuple[float, float, float]:
    """
    Вычисляет структурные показатели:
      – среднее (mean)
      – мода (mode, первый элемент)
      – медиана (median)
    """
    s = series.dropna()
    mean = s.mean()
    # mode() возвращает Series — берём первую моду
    modes = s.mode()
    mode = modes.iloc[0] if not modes.empty else float('nan')
    median = s.median()
    return mean, mode, median


def stats_variants(
    series: pd.Series,
    moderate_coef: float = 2.0,
    extreme_coef: float = 3.5
) -> pd.DataFrame:
    """
    Собирает таблицу со статистиками (mean/mode/median) по четырём вариантам:
      – Исходные данные
      – A: без экстремальных выбросов (>3.5 IQR)
      – B: без умеренных выбросов (2.0–3.5 IQR)
      – C: без умеренных и без экстремальных выбросов (<=2.0 IQR)
    """
    # 1) оригинал
    orig = series.dropna()

    # 2) классификация выбросов
    moderate, extreme = classify_iqr_outliers(
        series,
        moderate_coef=moderate_coef,
        extreme_coef=extreme_coef
    )

    # 3) варианты
    varA = orig.drop(index=extreme.index)              # без экстремальных
    varB = orig.drop(index=moderate.index)             # без умеренных
    out_idx = moderate.index.union(extreme.index)
    varC = orig.drop(index=out_idx)                    # без всех выбросов

    # 4) собираем статистики
    rows = {
        'Исходные данные':    compute_structural_stats(orig),
        'Вариант A (без экстр.)': compute_structural_stats(varA),
        'Вариант B (без умер.)':  compute_structural_stats(varB),
        'Вариант C (чистые данные)': compute_structural_stats(varC),
    }

    # 5) формируем DataFrame
    df_stats = pd.DataFrame.from_dict(
        rows,
        orient='index',
        columns=['Среднее', 'Мода', 'Медиана']
    )
    return df_stats
