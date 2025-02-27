import pydantic
import typing

class Output(pydantic.BaseModel):
    answer: float

class ReasonedOutput(pydantic.BaseModel):
    reasons: typing.List[str]
    answer: float
