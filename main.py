import pandas as pd
import src.preprocess as prep
import src.outliers as out
import src.plots as plot
import src.top_regions as top
import src.statistics as stat


def main() -> None:
    # 1. Подготовка данных
    df = pd.read_excel("data/data.xlsx")
    df = prep.drop_zero_amortization(df, "Амортизация в растениеводстве, тыс. руб.")
    series = df["Амортизация в растениеводстве, тыс. руб."]

    # 2. Определение выбросов
    print("a) Выбросы по интерквартильному размаху (IQR):")
    iqr_bounds = out.compute_iqr_bounds(series)
    number_iqr_outliers = out.count_iqr_outliers(series)
    out.print_bounds(iqr_bounds)
    print(f"\nОбщее число выбросов: {number_iqr_outliers}\n")
    top.top_iqr_regions(df)

    print("\n\nb) Выбросы по z-score:")
    number_z_score_outliers = out.count_z_score_outliers(series)
    print(f"Число выбросов: {number_z_score_outliers}\n")
    top.top_zscore_regions(df)

    print("\n\nc) Выбросы по диаграмме «ящик с усами»:")
    boxplot_bounds = out.compute_boxplot_bounds(series)
    number_boxplot_outliers = out.count_boxplot_outliers(series)
    out.print_bounds(boxplot_bounds)
    print(f"\nОбщее число выбросов: {number_boxplot_outliers}\n")
    top.top_boxplot_regions(df)

    print("\n\nd) Q-Q plot и P-P plot:")
    plot.plot_pp_plot(series)
    plot.plot_qq_plot(series)

    # 3. Классификация выбросов
    print("\n\nЭкстремальные (жесткие) и умеренные (мягкие) выбросы:")  # №5
    moderate, extreme = out.classify_iqr_outliers(series)
    print(f"Умеренных выбросов:  {len(moderate)}")
    print(f"Экстремальных выбросов: {len(extreme)}\n")
    top.top_moderate_extreme_regions(df)

    table = stat.stats_variants(series)
    print("\n", table.round(2))

    print("Общие выводы в конце README.md")


if __name__ == '__main__':
    main()
