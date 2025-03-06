import typing
import abc

import numpy as np
import scipy.stats
import scipy.special
import pydantic

class AbstractConfidenceModel(abc.ABC):
    @abc.abstractmethod
    def test(self, first: int, second: int) -> bool:
        pass

class SprtConfidenceModelConfig(pydantic.BaseModel):
    alpha: float
    beta: float
    p0: float
    p1: float

class SprtConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        alpha = 0.05,
        beta = 0.1,
        p0 = 0.5,
        p1 = 0.6,
    ):
        self.config = SprtConfidenceModelConfig(
            alpha = alpha,
            beta = beta,
            p0 = p0,
            p1 = p1,
        )

    def test(self, first, second) -> bool:
        alpha, beta, p0, p1 = self.config.alpha, self.config.beta, self.config.p0, self.config.p1

        logA = np.log((1 - beta) / alpha)
        logB = np.log(beta / (1 - alpha))
        logLR = first * np.log(p1 / p0) + second * np.log((1 - p1) / (1 - p0))
        if logLR >= logA or logLR <= logB:
            return True
        return False

class PvalueConfidenceModelConfig(pydantic.BaseModel):
    pvalue_threshold: float

class PValueConfidenceModel(AbstractConfidenceModel):
    def __init__(self, pvalue_threshold = 0.05):
        self.config = PvalueConfidenceModelConfig(pvalue_threshold=pvalue_threshold)

    def test(self, first, second) -> bool:
        pvalue_threshold = self.config.pvalue_threshold
        pvalue = scipy.stats.binomtest(first, first+second, p=0.5, alternative='greater').pvalue
        if pvalue <= pvalue_threshold:
            return True
        return False


class BayesianConfidenceModelConfig(pydantic.BaseModel):
    confidence_threshold: float
    priori: typing.Literal["jeffreys", "uniform"]

class BayesianConfidenceModel(AbstractConfidenceModel):
    def __init__(self, confidence_threshold = 0.95, priori = "uniform"):
        self.config = BayesianConfidenceModelConfig(
            confidence_threshold = confidence_threshold,
            priori = priori,
        )
    def test(self, first, second) -> bool:
        confidence_threshold, priori = self.config.confidence_threshold, self.config.priori

        if priori == "uniform":
            confidence = 1 - scipy.special.betainc(first + 1, second + 1, 0.5)
        elif priori == "jeffreys":
            confidence = 1 - scipy.special.betainc(first + 0.5, second + 0.5, 0.5)
        else:
            raise ValueError("Invalid priori")

        if confidence >= confidence_threshold:
            return True
        return False

class VoteConfidenceModel(AbstractConfidenceModel):
    def test(self, first, second) -> bool:
        _ = first, second  # Avoid linting error for unused arguments
        return False
