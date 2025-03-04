import re
from typing import Dict, List

from langchain.prompts.chat import HumanMessagePromptTemplate as HMPT
from langchain.prompts.chat import AIMessagePromptTemplate as AMPT

from src.dataset_types.code_contests import PRED_TEMPLATE, REF_TEMPLATE


class PROMPT:
    human = HMPT.from_template("{prompt}")


class DRAFT_PLAN:
    human = HMPT.from_template("Write logic for the code.")
    ai = AMPT.from_template("{draft_plan}")


class FINAL_PLAN:
    human = HMPT.from_template("Write one of C++ and Java code for the problem.")
    ai = AMPT.from_template("{final_plan}")


class REQUIREMENTS:
    human = HMPT.from_template("Write requirements for the problem.")
    ai = AMPT.from_template("{requirements}")
    landmarks = {
        "agnostic": "# Problem Agnostic Requirements",
        "fr": "# Functional Requirements",
        "io": "## Input-output Conditions",
        "expected": "## Expected Behavior",
        "edge": "## Edge Cases",
        "nfr": "# Non-functional Requirements",
        "performance": "## Performance",
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

        if cur_tc != "":
            parsed[current_part].append(cur_tc)

        return parsed


class CODE:
    human = HMPT.from_template("Write Python3 code for the problem.")
    ai = AMPT.from_template("{code}")


PRED_TEMPLATE = PRED_TEMPLATE


class TESTCASE:
    human = HMPT.from_template("Write test cases for the problem.")
    ai = AMPT.from_template("{gen_tc}")
    landmarks = {
        "functional": "# Test Cases Regarding Functional Requirements",
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
        # 각 라인을 읽으면서 각 파트의 시작점과 끝점을 찾는다.
        se_d = {}
        prev_part = None
        for i, line in enumerate(text.split("\n")):
            if line in cls.landmarks.values():
                part = list(cls.landmarks.keys())[
                    list(cls.landmarks.values()).index(line)
                ]
                se_d[part] = [i]
                if prev_part is not None:
                    se_d[prev_part].append(i)
                prev_part = part
        se_d[prev_part].append(len(text.split("\n")))

        # 각 파트의 시작점과 끝점을 바탕으로 그 사이의 내용을 저장한다.
        parsed = {}
        for part, se in se_d.items():
            parsed[part] = "\n".join(text.split("\n")[se[0] + 1 : se[1]])

        # 각 파트의 내용을 파싱한다.
        def parse_codeblocks(text: str) -> List[str]:
            # ```python\n...\n``` 형태의 코드 블럭 여러 개를 파싱한다.
            parsed = []
            for block in re.findall(r"``` python\n(.*?)\n```", text, re.DOTALL):
                parsed.append(block)
            return parsed

        for part, text in parsed.items():
            parsed[part] = parse_codeblocks(text)

        return parsed


REF_TEMPLATE = REF_TEMPLATE
