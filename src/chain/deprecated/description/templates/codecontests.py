from src.utils.indent import AIIndMessagePromptTemplate as AIMPT
from src.utils.indent import HumanIndMessagePromptTemplate as HIMPT
from langchain.schema import BaseOutputParser

PROMPT = {
    "human": HIMPT.from_template(
        """\
${prompt}
"""
    ),
}

PLAN = {
    "human": HIMPT.from_template(
        """\
# Let's think step by step"""
    ),
    "ai": AIMPT.from_template(
        """\
# START
${plan}
# END"""
    ),
}

IC = {
    "human": HIMPT.from_template(
        """\
# Let's write comments as detailed as possible"""
    ),
    "ai": AIMPT.from_template(
        """\
# START
${ic}
# END"""
    ),
}

CONST = {
    "human": HIMPT.from_template(
        """\
# Think about constraints"""
    ),
    "ai": AIMPT.from_template(
        """\
# START
${constraint}
# END"""
    ),
}

DESC = {
    "human": HIMPT.from_template(
        """\
# summarize the problem in one or two sentences"""
    ),
    "ai": AIMPT.from_template(
        """\
# START
${description}
# END"""
    ),
}

CODE = {
    "human": HIMPT.from_template(
        """\
# Let's write code"""
    ),
    "ai": AIMPT.from_template(
        """\
# START
${code}
# END"""
    ),
}

IMPORTS = """\
import math
import re
import numpy
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Set
"""


class PARSER(BaseOutputParser):
    def parse_code(self, head: str, text: str) -> str:
        code = IMPORTS + "\n" + text

        return code

    def parse(self, text: str) -> str:
        try:
            return text.split("# START\n")[1].split("# END")[0].rstrip()
        except:
            return text
