from src.utils.indent import AIIndMessagePromptTemplate as AIMPT
from src.utils.indent import HumanIndMessagePromptTemplate as HIMPT
from langchain.schema import BaseOutputParser

PLAN = {
    "human": HIMPT.from_template(
        """\
${head}
    ${prompt}

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
${head}
    ${prompt}

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
${head}
    ${prompt}

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
${head}
    ${prompt}

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
${head}
    ${prompt}

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
        start = head.rfind("def ")
        end = head[start:].find("(")
        head_front = head[start : start + end]

        body = self.parse(text)
        if head_front in body:
            code = body
        else:
            code = head + "\n" + body

        code = IMPORTS + "\n" + code

        return code

    def parse(self, text: str) -> str:
        try:
            return text.split("# START\n")[1].split("# END")[0].rstrip()
        except:
            return text


HUMANEVAL_EXAMPLES = [
    {
        "id": 1,
        "head": "def encrypt(s):",
        "prompt": """\
Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
For example:
encrypt('hi') returns 'lm'
encrypt('asdfghjkl') returns 'ewhjklnop'
encrypt('gf') returns 'kj'
encrypt('et') returns 'ix'""",
        "plan": """\
# 1. Create a variable to store the alphabet.
# 2. Create a variable to store the encrypted string.
# 3. Loop through the input string.
# 4. For each character in the input string, find the index of the character in the alphabet.
# 5. Add two multiplied by two to the index of the character in the alphabet.
# 6. Add the character at the new index to the encrypted string.
# 7. Return the encrypted string.""",
        "ic": """\
# loop through the numbers and get their index and value
    # loop through the numbers and get their index and value
        # ensure the two elements being compared are not the same
            # calculate the absolute difference between the two elements
            # check if the difference is less than the threshold
                # return True if difference is less than threshold
# return False if no elements are closer than the threshold""",
        "description": "# Create a function that takes a string as an argument and returns an encrypted version of the string. The encryption process involves rotating the alphabet by shifting each letter down by a distance equal to two multiplied by two places.",
        "constraint": """\
# The input string consists of lowercase alphabetic characters only.
# The input string can have one or more characters.
# The output string should consist of lowercase alphabetic characters only.
# The length of the output string will be the same as the length of the input string.
# The encryption is performed by rotating the alphabet, so the letters are shifted down by two multiplied by two places.""",
    }
]
