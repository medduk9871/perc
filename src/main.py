import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

import wandb
from src.logger import Logger
from src.problem import ProblemDataset

logger = logging.getLogger(__name__)


class Main:
    def __init__(
        self,
        run_name: Optional[str] = "debug",
        configs: str = "code-config.yaml",
        api_keys_path: str = "api_keys.json",
        result_root: str = "results",
    ):
        # initialize wandb
        wandb.init(
            entity="text-sketch-code",
            project="HuamnEval-with-Sketch",
            config=configs,
            name=run_name,
        )

        # load api keys
        api_keys = json.loads(Path(api_keys_path).read_text())
        # os.environ["OPENAI_API_TYPE"] = "azure"
        # os.environ["OPENAI_API_BASE"] = api_keys["azure_endpoint"]
        os.environ["OPENAI_API_KEY"] = api_keys["openai"]
        # os.environ["OPENAI_API_VERSION"] = api_keys["azure_api_version"]

        # set up the result directory
        self.run_name = run_name
        self.result_root = result_root
        self.result_dir = Path(self.result_root) / self.run_name
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

    def run(
        self,
        verbose: bool = False,
        thread_count: int = 16,
        indices: Optional[List[int]] = None,
    ):
        wandb.config.indices = indices

        problems = ProblemDataset(
            type=wandb.config["target.dataset_type"],
            split=wandb.config["target.dataset_split"],
            indices=wandb.config.indices,
            path=wandb.config.get("target.path", None),
            datapoint_kwargs=wandb.config.get("target.datapoint_kwargs", {}),
        )

        example_config = wandb.config.get("example", {})
        examples = {}
        for k, v in example_config.items():
            with open(v["path"], "r") as json_file:
                examples[k] = json.load(json_file)
            if "key" in v:
                for e in examples[k]:
                    e.update({k: e[v["key"]]})

        cache_config = wandb.config.get("cache", {})
        caches = {}
        for k, v in cache_config.items():
            with open(v["path"], "r") as json_file:
                caches[v["tgt_key"]] = []
                for d in json.load(json_file):
                    caches[v["tgt_key"]].append(d[v["src_key"]])

        if wandb.config["chain.model"] == "contract":
            from src.chain.contract.main import MainChain

            chain_class = MainChain
        elif wandb.config["chain.model"] == "reflexion":
            from src.chain.reflexion.main import MainChain

            chain_class = MainChain
        elif wandb.config["chain.model"] == "multi_turn":
            from src.chain.multi_turn.main import MainChain

            chain_class = MainChain
        else:
            raise NotImplementedError

        predictions = run_parallel(
            chain_class=chain_class,
            chain_kwargs={
                "llm_kwargs": wandb.config["llm"],
                "examples": examples,
                "dataset_type": wandb.config["target.dataset_type"],
                "verbose": verbose,
                **wandb.config.get("chain.kwargs", {}),
            },
            problems=problems.to_dicts(),
            caches=caches,
            thread_count=thread_count,
        )

        # log the results
        Logger().run(
            problems=problems,
            predictions=predictions,
            path=self.result_dir / "result.json",
        )


def parallel_target(
    chain_class,
    chain_kwargs,
    problems,
    caches,
    results,
    idx,
    pbar,
):
    predictions = []
    for i, problem in enumerate(problems):
        cache = {k: v[i] for k, v in caches.items()}
        chain = chain_class(**chain_kwargs, cache=cache)
        prediction = chain._call(problem)

        predictions.append(prediction)
        pbar.update(1)

    results[idx] = predictions


def run_parallel(
    chain_class,
    chain_kwargs,
    problems,
    caches,
    thread_count,
):
    threads = []
    print(thread_count)
    results = [[{} for _ in range(len(problems))] for _ in range(thread_count)]
    with tqdm(total=len(problems)) as pbar:
        for i in range(thread_count):
            l = len(problems) // thread_count
            if i + 1 < thread_count:
                r = l * i, l * (i + 1)
            else:
                r = l * i, len(problems)

            cur_caches = {k: v[r[0] : r[1]] for k, v in caches.items()}
            t = threading.Thread(
                target=parallel_target,
                args=(
                    chain_class,
                    chain_kwargs,
                    problems[r[0] : r[1]],
                    cur_caches,
                    results,
                    i,
                    pbar,
                ),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    return [item for sublist in results for item in sublist]
