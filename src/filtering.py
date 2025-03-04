# python run.py filtering --run_name humaneval-ours --sub_dir all run

import json
import multiprocessing
import os
from math import comb
from pathlib import Path
from typing import List

from evaluate import load
from pydantic import BaseModel


class Filtering(BaseModel):
    run_name: str
    sub_dir: str
    gt_eval_path: str = "results/humaneval-ours/eval.json"
    gen_tc_path: str = "results/humaneval-ours-testcase/result.json"
    parsed_gen_tc_key: str = "parsed_gen_tc"
    code_path: str = "results/humaneval-ours/result.json"
    chain_model: str = "multi_turn"
    dataset_type: str = "openai_humaneval"
    timeout: int = 3
    raw_scores_path: str = "results/humaneval-ours/raw_scores.json"
    k: List[int] = [1, 2, 5, 10]
    ratio: List[float] = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    categories: List[str] = [
        "general",
        "edge",
        "performance",
        "robustness",
        "reliability",
        "maintainability",
    ]

    gen_tc: List[dict] = None
    codes: List[dict] = None
    gt_eval: List[dict] = None
    pred_template: str = None
    ref_template: str = None
    code_eval: object = None
    raw_scores: List[dict] = None
    result_root: Path = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        **data,
    ):
        super().__init__(**data)

        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        self.result_root = Path("results") / self.run_name / self.sub_dir

        with open(self.gen_tc_path, "r") as f:
            self.gen_tc = json.load(f)

        with open(self.code_path, "r") as f:
            self.codes = json.load(f)

        with open(self.gt_eval_path, "r") as f:
            self.gt_eval = json.load(f)

        if self.chain_model == "multi_turn":
            if self.dataset_type == "openai_humaneval":
                import src.chain.multi_turn.templates.humaneval as DATA_TYPE

                self.pred_template = DATA_TYPE.PRED_TEMPLATE
                self.ref_template = DATA_TYPE.REF_TEMPLATE
            elif self.dataset_type == "deepmind/code_contests":
                import src.chain.multi_turn.templates.codecontests as DATA_TYPE

                self.pred_template = DATA_TYPE.PRED_TEMPLATE
                self.ref_template = DATA_TYPE.REF_TEMPLATE
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.code_eval = load("jjkim0807/code_eval")

        if Path(self.raw_scores_path).exists():
            with open(self.raw_scores_path, "r") as f:
                self.raw_scores = json.load(f)
        else:
            self.raw_scores = None

    def run(self):
        """score each candidate codes and calculate unbiased pass@k"""
        weighted_scores = self.score()
        pass_at_k = self.pass_at_k(weighted_scores=weighted_scores)

        with open(self.result_root / "pass_at_k.json", "w") as f:
            json.dump(pass_at_k, f, indent=4)

        return pass_at_k

    def score(self):
        """score each category of requirements and calculate the weighted score"""
        if self.raw_scores is None:
            self.raw_scores = self.score_raw()

        # calculate weighted score
        weighted_scores = []
        for raw_score in self.raw_scores:
            id = raw_score["id"]
            scores = raw_score["scores"]
            weighted_candidate_score = []
            for candidate_scores in zip(*scores.values()):
                weighted_score = sum(
                    [s * r for s, r in zip(candidate_scores, self.ratio)]
                )
                weighted_candidate_score.append(weighted_score)
            weighted_scores.append({"id": id, "score": weighted_candidate_score})

        return weighted_scores

    def score_raw(self):
        raw_scores = []
        for tc_d, code_d in zip(self.gen_tc, self.codes):
            assert tc_d["id"] == code_d["id"]
            id = tc_d["id"]
            parsed_gen_tc = tc_d[self.parsed_gen_tc_key]
            codes = code_d["code"]
            category_scores = {}
            for category in set(self.categories) - {"reliability"}:
                tcs = parsed_gen_tc[0][category]
                _, eval_result = self.code_eval.compute(
                    ids=[id],
                    predictions=[codes],
                    pred_template=self.pred_template,
                    references=[tcs],
                    ref_template=self.ref_template,
                    num_workers=multiprocessing.cpu_count(),
                    timeout=3,
                    k=[1],
                    disable_tqdm=True,
                )
                code_scores = []
                for e in eval_result[id]:
                    eval_d = e[1]
                    score = sum([r == "passed" for r in eval_d["result"]]) / len(
                        eval_d["result"]
                    )
                    code_scores.append(score)

                category_scores[category] = code_scores

            # reliability
            all_tcs = [tc for tcs in parsed_gen_tc[0].values() for tc in tcs]
            _, eval_result = self.code_eval.compute(
                ids=[id],
                predictions=[codes],
                pred_template=self.pred_template,
                references=[all_tcs],
                ref_template=self.ref_template,
                num_workers=multiprocessing.cpu_count(),
                timeout=3,
                k=1,
                disable_tqdm=True,
                ignore_assertion_errors=True,
            )
            code_scores = [int(l[1]["passed"]) for l in eval_result[id]]
            category_scores["reliability"] = code_scores

            raw_scores.append({"id": id, "scores": category_scores})

        with open(self.raw_scores_path, "w") as f:
            json.dump(raw_scores, f, indent=4)

        return raw_scores

    def pass_at_k(self, weighted_scores):
        # calculate unbiased pass@k
        pass_at_k = {}
        for k in self.k:
            probs = []
            for d in weighted_scores:
                id = d["id"]
                weighted_score = d["score"]
                ranked = sorted(
                    zip(weighted_score, range(len(weighted_score))),
                    key=lambda x: x[0],
                    reverse=True,
                )

                top_ranked = []
                tie_ranked = []
                tie_score = ranked[0][0]
                for i in range(len(ranked)):
                    score, idx = ranked[i]
                    if score == tie_score:
                        tie_ranked.append(idx)
                    else:
                        if len(top_ranked) + len(tie_ranked) >= k:
                            break
                        else:
                            top_ranked.extend(tie_ranked)
                            tie_ranked = [idx]
                            tie_score = score

                # count the number of passed in top_ranked
                top_passed = 0
                for idx in top_ranked:
                    if self.gt_eval[id][idx][1]["passed"]:
                        top_passed += 1

                # count the number of passed in tie_ranked
                tie_passed = 0
                for idx in tie_ranked:
                    if self.gt_eval[id][idx][1]["passed"]:
                        tie_passed += 1

                # calculate unbiased pass@k
                # when selecting (k - len(top_ranked)) number of code from tie_ranked,
                # probability that over (k - top_passed) number of passed code is selected
                n = k - len(top_ranked)
                x = k - top_passed
                prob = 1 - comb(n-k, x) / comb(n, x)
                
                probs.append(prob)
            
            pass_at_k[f"pass@{k}"] = sum(probs) / len(probs)
        
        return pass_at_k
    