import copy
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from string import Formatter, Template
from typing import Any, Dict, List, Mapping, Union

from langchain import PromptTemplate
from langchain.prompts.base import StringPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


class IndTemplate(Template):
    def __init__(self, template):
        super().__init__(template)
        self._get_indentation()

    def _get_indentation(self):
        self.indentation = {}
        # self.pattern is the regular expression Template uses to find the substitution patterns
        # self.template is the template string given in the constructor
        for match in self.pattern.finditer(self.template):
            symbol = match.group()
            # search whitespace between the start of a line and the current substitution pattern
            # '^' matches start of line only with flag re.MULTILINE
            pattern = r"^(\s*)" + re.escape(symbol)
            indent = re.search(pattern, self.template, re.MULTILINE)
            if indent:
                self.indentation[symbol] = indent.group(1)

    def substitute(self, __mapping: Mapping[str, object] = ..., **kwds: object) -> str:
        if isinstance(__mapping, dict):
            for k, v in __mapping.items():
                kwds[k] = v

        for k, v in kwds.items():
            k_symbol = "${" + k + "}"
            if k_symbol in self.indentation:
                v = textwrap.dedent(v)
                v = v.replace("\n", "\n" + self.indentation[k_symbol], -1)
                kwds[k] = v
        return super().substitute(**kwds)

    def safe_substitute(
        self, __mapping: Mapping[str, object] = ..., **kwds: object
    ) -> str:
        if isinstance(__mapping, dict):
            for k, v in __mapping.items():
                kwds[k] = v

        for k, v in kwds.items():
            k_symbol = "${" + k + "}"
            if k is None or v is None:
                continue

            if k_symbol in self.indentation:
                v = reindent(v, 4)
                v = v.replace("\n", "\n" + self.indentation[k_symbol], -1)
                kwds[k] = v
        return super().safe_substitute(**kwds)


def reindent(s, numSpaces):
    """Reindent the given string."""
    s = textwrap.dedent(s)

    # find the minimum indentation of any non-blank lines after the first line
    indnt_regex = re.compile(r"^(\s+)")
    indnt_sizes = [
        len(indnt_regex.match(line).group(1))
        for line in s.split("\n")
        if indnt_regex.match(line)
    ]
    if not indnt_sizes:
        return s
    indnt_size = min(indnt_sizes)
    if indnt_size == 0:
        indnt_sizes = set(indnt_sizes)
        indnt_sizes = sorted(indnt_sizes)
        indnt_size = indnt_sizes[1]

    # Replace indnt_size-space indentation with numSpaces spaces using regular expression
    lines = s.split("\n")
    new_lines = []
    for line in lines:
        prev_line = line
        line = line.lstrip()
        line = (len(prev_line) - len(line)) // indnt_size * numSpaces * " " + line
        new_lines.append(line)
    s = "\n".join(new_lines)

    return s


class IndPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs: Any) -> str:
        return IndTemplate(self.template).substitute(**kwargs)

    @classmethod
    def from_file(
        cls, template_file: Union[str, Path], input_variables: List[str], **kwargs: Any
    ) -> PromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            input_variables: A list of variable names the final prompt template
                will expect.
        Returns:
            The prompt loaded from the file.
        """
        with open(str(template_file), "r") as f:
            template = f.read()
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> PromptTemplate:
        """Load a prompt template from a template."""
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }

        return cls(
            input_variables=list(sorted(input_variables)), template=template, **kwargs
        )


class BaseIndMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    prompt: IndPromptTemplate

    @classmethod
    def from_template(cls, template: str, **kwargs: Any):
        prompt = IndPromptTemplate.from_template(template)
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(
        cls,
        template_file: Union[str, Path],
        input_variables: List[str],
        **kwargs: Any,
    ):
        prompt = IndPromptTemplate.from_file(template_file, input_variables)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """To a BaseMessage."""

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [self.format(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        return self.prompt.input_variables


class HumanIndMessagePromptTemplate(BaseIndMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return HumanMessage(content=text)


class AIIndMessagePromptTemplate(BaseIndMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return AIMessage(content=text)


class SystemIndMessagePromptTemplate(BaseIndMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return SystemMessage(content=text)


class IndChatPromptTemplate(ChatPromptTemplate):
    messages: List[BaseIndMessagePromptTemplate]


class FewShotIndChatPromptTemplate(IndChatPromptTemplate):
    examples: List[Dict[str, str]]
    example_prompt: List[BaseIndMessagePromptTemplate]
    prefix: List[BaseIndMessagePromptTemplate] = []

    def format_messages(self, **kwargs: Any) -> str:
        examples = []
        for e in self.examples:
            example = {}
            for p in self.example_prompt:
                for k in p.input_variables:
                    example[k] = e[k]
            examples.append(example)

        example_messages = []
        for example in examples:
            for p in self.example_prompt:
                example_messages.append(p.format(**example))

        original_messages = copy.deepcopy(self.messages)
        self.messages = self.prefix + example_messages + self.messages
        result = super().format_messages(**kwargs)

        self.messages = original_messages

        return result
