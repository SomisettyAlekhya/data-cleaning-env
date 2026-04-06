# agent.py

from env.models import Action

class DataCleaningAgent:
    """
    Automatic Data Cleaning Agent for real-world datasets:
    - Removes duplicates
    - Fills missing values (mode)
    - Converts object columns to numeric
    - Normalizes numeric features
    """

    def __init__(self):
        pass

    def clean_dataset(self, env):
        """
        Clean dataset quickly by applying all actions per column type.
        Returns final observation and list of rewards/actions.
        """
        obs = env.reset()
        rewards = []
        df = env.df

        # 1️⃣ Remove duplicates
        if getattr(obs, "duplicates", 0) > 0:
            action = Action(action_type="remove_duplicates")
            obs, reward, done, info = env.step(action)
            rewards.append((action.action_type, action.column, reward.value, reward.reason))

        # 2️⃣ Fill missing values using mode
        for col in obs.columns:
            if df[col].isna().sum() > 0:
                action = Action(action_type="fill_missing_mode", column=col)
                obs, reward, done, info = env.step(action)
                rewards.append((action.action_type, action.column, reward.value, reward.reason))

        # 3️⃣ Convert object columns to numeric
        for col in obs.columns:
            if df[col].dtype == "object":
                action = Action(action_type="convert_to_numeric", column=col)
                obs, reward, done, info = env.step(action)
                rewards.append((action.action_type, action.column, reward.value, reward.reason))

        # 4️⃣ Normalize numeric columns
        for col in obs.columns:
            if df[col].dtype in ["int64", "float64"]:
                action = Action(action_type="normalize_column", column=col)
                obs, reward, done, info = env.step(action)
                rewards.append((action.action_type, action.column, reward.value, reward.reason))

        # 5️⃣ No-op to finish
        obs, _, done, _ = env.step(Action(action_type="noop"))
        return obs, rewards