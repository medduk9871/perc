from itertools import repeat
import json
import multiprocessing
import os
from pathlib import Path

from evaluate import load
from pydantic import BaseModel


class Evaluator(BaseModel):
    disable_tqdm: bool = False

    def __init__(self, **data):
        super().__init__(**data)

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    def run(
        self,
        run_name,
        sub_dir=None,
        id=None,
        ref_path=None,
        chain_model="multi_turn",
        dataset_type="openai_humaneval",
        k=[1, 2, 5, 10],
        ref_key="test",
        pred_key="code",
        timeout=3,
        early_stop=False,
        ignore_assertion_errors=False,
    ):
        result_root = Path("results") / run_name
        json_path = result_root / "result.json"
        if sub_dir is not None:
            result_root = result_root / sub_dir
        if not result_root.exists():
            result_root.mkdir(parents=True)

        # load the json
        with open(json_path, "r") as f:
            predictions = json.load(f)
        if id is not None:
            predictions = [p for p in predictions if p["id"] == id]
        predictions = sorted(predictions, key=lambda x: x["id"])

        if ref_path is not None:
            with open(ref_path, "r") as f:
                references = json.load(f)
        else:
            references = predictions
        if id is not None:
            references = [r for r in references if r["id"] == id]
        references = sorted(references, key=lambda x: x["id"])

        assert all(p["id"] == r["id"] for p, r in zip(predictions, references))
        ids = [p["id"] for p in predictions]
        predictions = [p[pred_key] for p in predictions]
        references = [r[ref_key] for r in references]
        references = [[r] if isinstance(r, str) else r for r in references]

        if chain_model == "multi_turn":
            if dataset_type == "openai_humaneval":
                import src.chain.multi_turn.templates.humaneval as DATA_TYPE

                pred_template = DATA_TYPE.PRED_TEMPLATE
                ref_template = DATA_TYPE.REF_TEMPLATE
            elif dataset_type == "deepmind/code_contests":
                import src.chain.multi_turn.templates.codecontests as DATA_TYPE

                pred_template = DATA_TYPE.PRED_TEMPLATE
                ref_template = DATA_TYPE.REF_TEMPLATE
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # evaluate the response
        code_eval = load("jjkim0807/code_eval")
        pass_at_k, results = code_eval.compute(
            ids=ids,
            predictions=predictions,
            pred_template=pred_template,
            references=references,
            ref_template=ref_template,
            k=k,
            num_workers=multiprocessing.cpu_count(),
            timeout=timeout,
            early_stop=early_stop,
            disable_tqdm=self.disable_tqdm,
            ignore_assertion_errors=ignore_assertion_errors,
        )

        print(pass_at_k)
        pass_at_k_path = result_root / "pass@k.json"
        with open(pass_at_k_path, "w") as f:
            json.dump(pass_at_k, f, indent=4)

        # save the result
        eval_path = result_root / "eval.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=4)

    def pass_rate(self, eval_results_path):
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

        pass_rate = 0
        for _, r in eval_results.items():
            r = r[0][1]
            ratio = r["result"].count("passed") / len(r["result"])
            pass_rate += ratio
        pass_rate /= len(eval_results)

        return pass_rate

    def run_one(
        self,
        reference,
        candidate,
        timeout=32,
        early_stop=False,
        num_workers=1,
    ):
        code_eval = load("jjkim0807/code_eval")
        _, results = code_eval.compute(
            references=[reference],
            predictions=[[candidate]],
            k=[1],
            num_workers=num_workers,
            timeout=timeout,
            early_stop=early_stop,
            disable_tqdm=self.disable_tqdm,
        )

        result = results[0][0][1]
        passed = result["passed"]
        result = result["result"]
        return passed, result
