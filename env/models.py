from pydantic import BaseModel
from typing import List, Dict, Optional

class Observation(BaseModel):
    rows: int
    columns: List[str]
    missing_values: int
    duplicates: int
    dtypes: Dict[str, str]

class Action(BaseModel):
    action_type: str
    column: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: Optional[str] = ""