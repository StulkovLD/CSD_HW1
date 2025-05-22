import pandas as pd
from src.outliers import (count_z_score_outliers, count_boxplot_outliers,
                          classify_iqr_outliers, count_iqr_outliers)


def top_iqr_regions(
        df: pd.DataFrame,
        series_col: str = 'Амортизация в растениеводстве, тыс. руб.',
        region_col: str = 'Субъект РФ',
        top_n: int = 3
) -> None:
    """
    Печатает топ N регионов по доле выбросов, определённых по IQR.
    """
    # считаем выбросы в каждом регионе
    counts = (
        df.groupby(region_col)[series_col]
        .apply(count_iqr_outliers)
        .rename('outliers')
    )
    total_outliers = counts.sum()
    # доля в процентах
    percents = (counts / total_outliers * 100).sort_values(ascending=False)

    print(f"Топ {top_n} регионов по выбросам IQR (доля от всех выбросов):")
    print(percents.head(top_n).round(2).to_string())


def top_zscore_regions(
        df: pd.DataFrame,
        series_col: str = 'Амортизация в растениеводстве, тыс. руб.',
        region_col: str = 'Субъект РФ',
        top_n: int = 3
) -> None:
    """
    Печатает топ N регионов по доле выбросов, определённых по Z-score.
    """
    counts = (
        df.groupby(region_col)[series_col]
        .apply(count_z_score_outliers)
        .rename('outliers')
    )
    total_outliers = counts.sum()
    percents = (counts / total_outliers * 100).sort_values(ascending=False)

    print(f"Топ {top_n} регионов по выбросам Z-score (доля от всех выбросов):")
    print(percents.head(top_n).round(2).to_string())


def top_boxplot_regions(
        df: pd.DataFrame,
        series_col: str = 'Амортизация в растениеводстве, тыс. руб.',
        region_col: str = 'Субъект РФ',
        whisker_coef: float = 1.5,
        top_n: int = 3
) -> None:
    """
    Печатает топ N регионов по доле выбросов, определённых по правилу «ящик с усами».
    """
    # считаем число выбросов в каждом регионе
    counts = (
        df.groupby(region_col)[series_col]
        .apply(lambda s: count_boxplot_outliers(s, whisker_coef=whisker_coef))
        .rename('outliers')
    )

    total = counts.sum()
    pct = (counts / total * 100).sort_values(ascending=False)

    print(f"Топ {top_n} регионов по выбросам Boxplot (coef={whisker_coef}):")
    print(pct.head(top_n).round(2).to_string())


def top_moderate_extreme_regions(
        df: pd.DataFrame,
        series_col: str = 'Амортизация в растениеводстве, тыс. руб.',
        region_col: str = 'Субъект РФ',
        top_n: int = 3,
        moderate_coef: float = 2.0,
        extreme_coef: float = 3.5
) -> None:
    """
    Печатает топ N регионов по доле 'умеренных' и 'экстремальных' выбросов (IQR-классификация).
    """
    # группируем и считаем
    agg = {'moderate': [], 'extreme': []}
    for region, group in df.groupby(region_col):
        mod, ext = classify_iqr_outliers(
            group[series_col],
            moderate_coef=moderate_coef,
            extreme_coef=extreme_coef
        )
        agg['moderate'].append((region, len(mod)))
        agg['extreme'].append((region, len(ext)))

    # превращаем в Series
    mod_counts = pd.Series(dict(agg['moderate']), name='moderate')
    ext_counts = pd.Series(dict(agg['extreme']), name='extreme')

    # считаем доли
    mod_pct = (mod_counts / mod_counts.sum() * 100).sort_values(ascending=False)
    ext_pct = (ext_counts / ext_counts.sum() * 100).sort_values(ascending=False)

    print(f"Топ {top_n} регионов по умеренным выбросам:")
    print(mod_pct.head(top_n).round(2).to_string())
    print()
    print(f"Топ {top_n} регионов по экстремальным выбросам:")
    print(ext_pct.head(top_n).round(2).to_string())
