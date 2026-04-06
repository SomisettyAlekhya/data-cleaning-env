from fastapi import FastAPI
from pydantic import BaseModel
from env.data_env import DataCleaningEnv
from env.models import Action

app = FastAPI()
env = DataCleaningEnv("easy")

class StepRequest(BaseModel):
    action_type: str
    column: str | None = None

@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "rows": obs.rows,
        "columns": obs.columns,
        "missing_values": obs.missing_values,
        "duplicates": obs.duplicates
    }

@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, info = env.step(
        Action(action_type=req.action_type, column=req.column)
    )

    return {
        "reward": reward.value,
        "done": done
    }

@app.get("/state")
def state():
    df = env.df
    return {
        "rows": len(df),
        "columns": list(df.columns)
    }
