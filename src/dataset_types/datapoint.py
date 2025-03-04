from typing import Optional
from pydantic import BaseModel
import pandas as pd


class Datapoint(BaseModel):
    id: Optional[str] = None
    prompt: Optional[str] = None
    code: Optional[str] = None

    def to_dataframe(self):
        return pd.DataFrame([self.dict()])