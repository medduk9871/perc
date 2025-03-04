from langchain.prompts.chat import HumanMessagePromptTemplate as HMPT
from langchain.prompts.chat import AIMessagePromptTemplate as AMPT
from langchain.prompts.chat import SystemMessagePromptTemplate as SMPT


class PLAN:
    system = SMPT.from_template(
        "You are an AI coding assistant that can write an accurate plan for functions given the signature and docstring. Think step by step."
    )
    human = HMPT.from_template("{prompt}")
    ai = AMPT.from_template("{plan}")


class CONTRACT:
    system = SMPT.from_template(
        """\
You are an AI coding assistant that can write flawless and detailed program contract for functions given the signature, docstring and plan.
Contract format is as follows:
# 1. Acceptable and unacceptable input values or types, and their meanings
#     Acceptable input values:
#     - ... (as many as you want)
#     Unacceptable input values:
#     - ... (as many as you want)
# 2. Return values or types, and their meanings
#     ... (as long as you want)
# 3. Invariants
#     ... (as long as you want)"""
    )
    human = HMPT.from_template(
        """\
[signature & docstring]
{prompt}
                               
[plan]
{plan}"""
    )
    ai = AMPT.from_template("{contract}")


class CODE:
    system = SMPT.from_template(
        "You are an AI coding assistant that can write an accurate code for functions given the signature, docstring, contract and plan."
    )
    human = HMPT.from_template(
        """\
[signature & docstring]
{prompt}

[contract]
{contract}

[plan]
{plan}"""
    )
    ai = AMPT.from_template("{code}")


class TESTCASE:
    system = SMPT.from_template(
        """\
You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature, docstring and contract.
Unit test format is as follows:
assert function_name(*args, **kwargs) == expected_output"""
    )
    human = HMPT.from_template(
        """\
[signature & docstring]
{prompt}

[contract]
{contract}"""
    )
    ai = AMPT.from_template("{gen_tc}")


IMPORTS = """\
import math
import re
import numpy
import numpy as np
from typing import *
"""
