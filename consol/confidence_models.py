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
    p1: float
    alpha: float
    beta: float

class SprtConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        p1 = 0.7,
        alpha = 0.05,
        beta = 0.1,
    ):
        self.config = SprtConfidenceModelConfig(
            p1 = p1,
            alpha = alpha,
            beta = beta,
        )

    def test(self, first, second) -> bool:
        p0 = 0.5
        p1, alpha, beta = self.config.p1, self.config.alpha, self.config.beta

        logA = np.log((1 - beta) / alpha)
        logB = np.log(beta / (1 - alpha))
        logLR = first * np.log(p1 / p0) + second * np.log((1 - p1) / (1 - p0))
        if logLR >= logA or logLR <= logB:
            return True
        return False


class SbftConfidenceModelConfig(pydantic.BaseModel):
    evidence_strength: typing.Literal["substantial", "strong", "decisive"]
    priori: typing.Literal["jeffreys", "uniform"]

class SbftConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        evidence_strength = "decisive",
        priori = "uniform",
    ):
        self.config = SbftConfidenceModelConfig(
            evidence_strength = evidence_strength,
            priori = priori,
        )

    def test(self, first, second) -> bool:
        """
        Under the alternative hypothesis (H1), we assume p > 0.5.
        The alternative uses a truncated Beta prior on p in the interval (0.5, 1).

        1. The full Beta(a, b) prior is defined on [0,1]. Its density is:
              prior(p) = p^(a-1) * (1-p)^(b-1) / B(a, b)

        2. When we restrict p to (0.5, 1), we must renormalize the density.
           The normalization constant is:
              norm = 1 - I_{0.5}(a, b)
           where I_x(a, b) is the regularized incomplete Beta function.

        3. The marginal likelihood under H1 (i.e. the probability of the data)
           is then given by integrating over p from 0.5 to 1:

              P(data|H1) = ∫[0.5,1] p^(first) * (1-p)^(second) * [prior(p)] dp

           Because our prior already has the p^(a-1)*(1-p)^(b-1) term, the
           closed-form solution for the integral is:

              P(data|H1) = [ B(first+a, second+b) * (1 - I_{0.5}(first+a, second+b)) ]
                           / [ B(a, b) * (1 - I_{0.5}(a, b)) ]
        """
        evidence_strength, priori = self.config.evidence_strength, self.config.priori

        if priori == "uniform":
            alpha, beta = 1, 1
        elif priori == "jeffreys":
            alpha, beta = 0.5, 0.5
        else:
            raise ValueError("Invalid priori")

        if evidence_strength == "substantial":
            K_threshold = 3.2
        elif evidence_strength == "strong":
            K_threshold = 10
        elif evidence_strength == "decisive":
            K_threshold = 100
        else:
            raise ValueError("Invalid evidence_strength")

        # Under the null hypothesis (H0), assume p = 0.5 for all trials.
        likelihood_H0 = 0.5 ** (first + second)

        # Compute the normalization constant for the truncated prior.
        norm = 1 - scipy.special.betainc(alpha, beta, 0.5)

        # Compute the integral (in closed form) using the Beta function and the
        # regularized incomplete Beta function.
        numerator = scipy.special.beta(first + alpha, second + beta) * \
                    (1 - scipy.special.betainc(first + alpha, second + beta, 0.5))

        likelihood_H1 = numerator / (scipy.special.beta(alpha, beta) * norm)

        # The Bayes factor K is the ratio of the likelihoods:
        #   K = P(data|H1) / P(data|H0)
        K = likelihood_H1 / likelihood_H0

        if K >= K_threshold:
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
