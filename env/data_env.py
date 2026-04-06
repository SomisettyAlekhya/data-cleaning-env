import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from .models import Observation, Action, Reward

class DataCleaningEnv:
    def __init__(self, task: str = "easy"):
        assert task in ["easy", "medium", "hard"]
        self.task = task
        self.df = None
        self.initial_cells = 1
        self._generate_dataset()

    # ----------- OpenEnv API -----------
    def reset(self) -> Observation:
        self._generate_dataset()
        return self._get_obs()

    def state(self) -> Observation:
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        prev_score = self._compute_score()

        try:
            if action.action_type == "remove_duplicates":
                self.df = self.df.drop_duplicates().reset_index(drop=True)

            elif action.action_type in ["fill_missing_mean", "fill_missing_median", "fill_missing_mode"]:
                if action.column and action.column in self.df.columns:
                    if action.action_type == "fill_missing_mean":
                        if pd.api.types.is_numeric_dtype(self.df[action.column]):
                            self.df[action.column] = self.df[action.column].fillna(self.df[action.column].mean())
                    elif action.action_type == "fill_missing_median":
                        if pd.api.types.is_numeric_dtype(self.df[action.column]):
                            self.df[action.column] = self.df[action.column].fillna(self.df[action.column].median())
                    else:
                        mode = self.df[action.column].mode()
                        if len(mode) > 0:
                            self.df[action.column] = self.df[action.column].fillna(mode.iloc[0])

            elif action.action_type == "convert_to_numeric":
                if action.column and action.column in self.df.columns:
                    self.df[action.column] = pd.to_numeric(self.df[action.column], errors="coerce")

            elif action.action_type == "normalize_column":
                if action.column and action.column in self.df.columns:
                    col = self.df[action.column]
                    if pd.api.types.is_numeric_dtype(col):
                        min_v, max_v = col.min(), col.max()
                        if pd.notna(min_v) and pd.notna(max_v) and max_v != min_v:
                            self.df[action.column] = (col - min_v) / (max_v - min_v)

        except Exception as e:
            # Penalize invalid operations slightly
            reward = Reward(value=0.0, reason=f"Invalid action: {e}")
            obs = self._get_obs()
            done = False
            info = {"score": self._compute_score()}
            return obs, reward, done, info

        new_score = self._compute_score()
        delta = max(0.0, new_score - prev_score)  # reward only positive progress
        reward = Reward(value=float(delta), reason="Improved data quality" if delta > 0 else "No improvement")

        done = new_score >= 0.95
        info = {"score": float(new_score)}
        return self._get_obs(), reward, done, info

    # ----------- Internals -----------
    def _generate_dataset(self):
        rng = np.random.default_rng(42)
        n = {"easy": 30, "medium": 80, "hard": 150}[self.task]
        miss_rate = {"easy": 0.1, "medium": 0.3, "hard": 0.5}[self.task]
        dup_rate = {"easy": 0.05, "medium": 0.15, "hard": 0.3}[self.task]

        ages = rng.integers(18, 65, size=n).astype(object)
        salary = rng.normal(50000, 15000, size=n).astype(object)

        # inject strings
        for i in range(0, n, 7):
            ages[i] = str(ages[i])
        for i in range(0, n, 9):
            salary[i] = str(int(salary[i]))

        df = pd.DataFrame({
            "age": ages,
            "salary": salary,
            "dept": rng.choice(["A", "B", "C"], size=n)
        })

        # missing values
        for col in df.columns:
            mask = rng.random(n) < miss_rate
            df.loc[mask, col] = np.nan

        # duplicates
        dup_count = int(n * dup_rate)
        if dup_count > 0:
            df = pd.concat([df, df.sample(dup_count, random_state=42)], ignore_index=True)

        self.df = df.reset_index(drop=True)
        self.initial_cells = max(1, self.df.shape[0] * self.df.shape[1])

    def _get_obs(self) -> Observation:
        return Observation(
            rows=int(self.df.shape[0]),
            columns=list(self.df.columns),
            missing_values=int(self.df.isna().sum().sum()),
            duplicates=int(self.df.duplicated().sum()),
            dtypes={c: str(self.df[c].dtype) for c in self.df.columns}
        )

    def _compute_score(self) -> float:
        missing = self.df.isna().sum().sum()
        duplicates = self.df.duplicated().sum()
        total_cells = max(1, self.df.shape[0] * self.df.shape[1])

        # penalties
        p_missing = missing / total_cells
        p_dup = duplicates / max(1, self.df.shape[0])

        # numeric columns ratio
        numeric_cols = sum([pd.api.types.is_numeric_dtype(self.df[c]) for c in self.df.columns])
        p_types = 1.0 - (numeric_cols / max(1, len(self.df.columns)))

        score = 1.0 - (0.5 * p_missing + 0.3 * p_dup + 0.2 * p_types)
        return float(max(0.0, min(1.0, score)))
