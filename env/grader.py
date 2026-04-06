# grader.py

import pandas as pd

class Grader:
    """
    Simple grader for cleaned dataset.
    Returns score between 0.0–1.0
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate(self):
        score = 1.0

        # Missing values
        if self.df.isna().sum().sum() > 0:
            score -= 0.3

        # Duplicates
        if self.df.duplicated().sum() > 0:
            score -= 0.3

        # Numeric normalization check
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            if self.df[col].min() < 0 or self.df[col].max() > 1:
                score -= 0.2
                break

        return max(score, 0.0)