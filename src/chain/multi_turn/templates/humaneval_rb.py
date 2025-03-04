from typing import Dict, List
from langchain.prompts.chat import HumanMessagePromptTemplate as HMPT
from langchain.prompts.chat import AIMessagePromptTemplate as AMPT
from langchain.prompts.chat import SystemMessagePromptTemplate as SMPT

from src.dataset_types.human_eval import PRED_TEMPLATE, REF_TEMPLATE


class PROMPT:
    human = HMPT.from_template("{prompt}")


class DRAFT_PLAN:
    human = HMPT.from_template("Write a plan for the problem.")
    ai = AMPT.from_template("{draft_plan}")


class FINAL_PLAN:
    human = HMPT.from_template("Write the final plan for the problem.")
    ai = AMPT.from_template("{final_plan}")


class REQUIREMENTS:
    human = HMPT.from_template("Write code for the problem in a referenceable programming language.")
    ai = AMPT.from_template("{requirements}")

class CODE:
    human = HMPT.from_template("Complete Ruby code without ```ruby templates for the prompt following the plan.")
    ai = AMPT.from_template("{code}")


PRED_TEMPLATE = PRED_TEMPLATE

class TESTCASE:
    human = HMPT.from_template(
        """\
Write multiple test cases for the function as format below.
assert function_name(input) == output"""
    )
    ai = AMPT.from_template("{gen_tc}")
    landmarks = {
        "fr": "# Test Cases Regarding Functional Requirements",
        "general": "## General Cases",
        "edge": "## Edge Cases",
        "nfr": "# Test Cases Regarding Non-functional Requirements",
        "performance": "## Performance Requirements",
        "sqr": "## Specific Quality Requirements",
        "robustness": "### Robustness",
        "reliability": "### Reliability",
        "maintainability": "### Maintainability",
    }

    @classmethod
    def parse(cls, text: str) -> Dict[str, List[str]]:
        parsed = {}
        for part in cls.landmarks.keys():
            parsed[part] = []
        current_part = list(cls.landmarks.keys())[0]
        cur_tc = ""
        for line in text.split("\n"):
            if line in cls.landmarks.values():
                if cur_tc != "":
                    parsed[current_part].append(cur_tc)
                    cur_tc = ""
                current_part = list(cls.landmarks.keys())[
                    list(cls.landmarks.values()).index(line)
                ]

            cur_tc += line + "\n"

            # if assertion in line
            if "assert " in line:
                parsed[current_part].append(cur_tc)
                cur_tc = ""

        if cur_tc != "":
            parsed[current_part].append(cur_tc)

        return parsed

REF_TEMPLATE = REF_TEMPLATE
