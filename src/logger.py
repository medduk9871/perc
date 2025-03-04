import json
from pydantic import BaseModel

import wandb
from src.problem import Problem


class Logger(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)

    def run(
        self,
        problems,
        predictions,
        path,
    ):
        ds = []
        for i in range(len(problems)):
            problem: Problem = problems[i]
            prediction: dict = predictions[i]

            d = {
                **problem.to_dict(),
                **prediction,
            }
            ds.append(d)

        with open(path, "w") as f:
            json.dump(ds, f, indent=4)
        # wandb.log_artifact(path)

        # save wandb configs to the same directory
        with open(path.parent / "config.json", "w") as f:
            json.dump(wandb.config.as_dict(), f, indent=4)
