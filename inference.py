import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-agent")


def run_task(task):
    print(f"[START] task={task} env=data-cleaning-env model={MODEL_NAME}")

    # Reset
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task})
    obs = r.json()["observation"]

    done = False
    step = 0
    rewards = []

    actions = [
        {"action_type": "fill_missing_mean", "column": obs["columns"][0]},
        {"action_type": "remove_duplicates", "column": None},
    ]

    for act in actions:
        step += 1
        try:
            r = requests.post(f"{API_BASE_URL}/step", json=act)
            res = r.json()

            reward = float(res["reward"])
            done = res["done"]
            rewards.append(reward)

            print(f"[STEP] step={step} action={act['action_type']} reward={reward:.2f} done={str(done).lower()} error=null")

            if done:
                break

        except Exception as e:
            print(f"[STEP] step={step} action={act['action_type']} reward=0.00 done=false error={str(e)}")

    score = sum(rewards) / len(rewards) if rewards else 0

    print(f"[END] success={str(done).lower()} steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
