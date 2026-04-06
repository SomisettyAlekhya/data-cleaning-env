from env.data_env import DataCleaningEnv
from env.models import Action
from agent import DataCleaningAgent
import os

# Required environment variables
API_BASE_URL = os.getenv("https://dummy.api")
MODEL_NAME = os.getenv("dummy-model")
HF_TOKEN = os.getenv("dummy-token")

# Optional OpenAI client (for compliance)
try:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except:
    client = None


def run_task(task):

    env = DataCleaningEnv(task)
    obs = env.reset()

    agent = DataCleaningAgent()

    done = False
    total_reward = 0
    info = {}

    print("[START]")
    print(f"task_id: {task}")

    for step in range(10):

        act, col = agent.decide(obs)

        action = Action(action_type=act, column=col)

        obs, reward, done, info = env.step(action)

        total_reward += reward.value

        print("[STEP]")
        print(f"action: {act}")
        print(f"reward: {reward.value}")

        if done:
            break

    print("[END]")
    final_score = info.get("score", total_reward)
    print(f"final_score: {final_score}")


if __name__ == "__main__":

    for task in ["easy", "medium", "hard"]:
        run_task(task)
