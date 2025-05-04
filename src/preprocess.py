import pandas as pd


def drop_zero_amortization(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Удаляет из датафрейма df строки, в которых значение по столбцу col равно нулю.
    """
    return df[df[col] != 0].reset_index(drop=True)
