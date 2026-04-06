from fastapi import FastAPI
from pydantic import BaseModel
from env.data_env import DataCleaningEnv
from env.models import Action

app = FastAPI()

env = None

# ===== Request Models =====
class ResetRequest(BaseModel):
    task: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    column: str | None = None


# ===== RESET =====
@app.post("/reset")
def reset(req: ResetRequest):
    global env
    env = DataCleaningEnv(req.task)
    obs = env.reset()

    return {
        "observation": {
            "rows": obs.rows,
            "columns": obs.columns,
            "missing_values": obs.missing_values,
            "duplicates": obs.duplicates
        },
        "info": {}
    }


# ===== STEP =====
@app.post("/step")
def step(req: StepRequest):
    global env

    action = Action(
        action_type=req.action_type,
        column=req.column
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": {
            "rows": obs.rows,
            "columns": obs.columns,
            "missing_values": obs.missing_values,
            "duplicates": obs.duplicates
        },
        "reward": round(reward.value, 2),
        "done": done,
        "info": info
    }


# ===== STATE =====
@app.get("/state")
def state():
    global env
    return {
        "data": env.df.to_dict()
    }
