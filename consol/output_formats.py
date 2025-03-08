import abc
import pydantic
import typing

class AbstractOutput(pydantic.BaseModel, abc.ABC):
    answer: str

class ReasonedMixin(abc.ABC):
    reasons: typing.List[str]

class FloatOutput(AbstractOutput):
    answer: float

class ABCDEOutput(AbstractOutput):
    answer: typing.Literal["A", "B", "C", "D", "E"]
