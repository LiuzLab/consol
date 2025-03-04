import abc
import pydantic
import typing

class AbstractOutput(pydantic.BaseModel, abc.ABC):
    pass

class FloatOutput(AbstractOutput):
    answer: float

class ReasonedFloatOutput(AbstractOutput):
    reasons: typing.List[str]
    answer: float
