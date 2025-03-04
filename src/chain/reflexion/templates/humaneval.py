from collections import defaultdict
import json
import ast
import re
from typing import Any, List

from langchain.prompts.chat import AIMessagePromptTemplate as AMPT
from langchain.prompts.chat import HumanMessagePromptTemplate as HMPT


class PROMPT:
    human = HMPT.from_template("{prompt}")


class TESTCASE:
    human = HMPT.from_template("Write unique, diverse, and intuitive unit tests.")
    ai = AMPT.from_template("{gen_tc}")

    def parse(tests: str) -> List[str]:
        pattern = r"assert .*?(?=assert|$)"
        results = re.findall(pattern, tests, re.S)

        def is_syntax_valid(code: str) -> bool:
            try:
                ast.parse(code)
                return True
            except Exception:
                return False

        results = [r for r in results if is_syntax_valid(r)]

        return results


TEST_EXCEPTION_TEMPLATE = """\
try:
    {code}
    raise Exception("No exception raised.")
except {exception} as e:
    pass
except Exception as e:
    raise Exception("Wrong exception raised.")\
"""

TEST_OUTPUT_TEMPLATE = """\
result = {code}
assert result == {output}, f"Wrong output: {{result}} != {output}"\
"""


class CURRICULUM_TESTCASE:
    human = HMPT.from_template(
        """\
Write unique, diverse, and intuitive unit tests.
The unit tests should be written in following three parts:
# 1. Valid General Cases
# 2. Valid Edge Cases
# 3. Invalid Cases
Formats of the unit tests are as follows:
{{"code": "YOUR CODE", "output": "YOUR OUTPUT"}}
or
{{"code": "YOUR CODE", "exception": "YOUR EXCEPTION"}}
"""
    )
    ai = TESTCASE.ai

    def parse(
        tests: str,
        curriculum={"general", "edge", "invalid"},
    ) -> List[str]:
        results = defaultdict(list)
        for line in tests.splitlines():
            if line.startswith("#"):
                if line.startswith("# 1."):
                    key = "general"
                elif line.startswith("# 2."):
                    key = "edge"
                elif line.startswith("# 3."):
                    key = "invalid"
                else:
                    raise ValueError("Invalid test case: {}".format(line))
                continue

            # find json string in the line
            try:
                pattern = r"{.*?}"
                line = re.findall(pattern, line, re.S)[0]
                d = json.loads(line)
            except Exception:
                continue
            
            if "exception" in d:
                r = TEST_EXCEPTION_TEMPLATE.format(
                    code=d["code"],
                    exception=d["exception"],
                )
            elif "output" in d:
                r = TEST_OUTPUT_TEMPLATE.format(
                    code=d["code"],
                    output=d["output"],
                )
            else:
                raise ValueError("Invalid test case: {}".format(d))

            results[key].append(r)

        results = [r for key in curriculum for r in results[key]]

        return results


class FEEDBACK:
    human = HMPT.from_template("Write execution results of the unit tests.")
    ai = AMPT.from_template("{feedback}")


class REFLECTION:
    human = HMPT.from_template(
        "Write a few sentences to explain why your implementation is wrong as indicated by the tests. Only provide the few sentence description in your answer, not the implementation."
    )
    ai = AMPT.from_template("{reflection}")


class PREV_CODE:
    human = HMPT.from_template(
        "Write your full implementation (restate the function signature). Only respond with python code, NOT ENGLISH."
    )
    ai = AMPT.from_template("{prev_code}")

    def parse(code: str) -> str:
        return code


class CODE:
    human = HMPT.from_template(
        "Write your full implementation (restate the function signature). Only respond with python code, NOT ENGLISH."
    )
    ai = AMPT.from_template("{code}")

    def parse(code: str) -> str:
        return code
