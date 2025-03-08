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
        p1 = 0.6,
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
        if logLR >= logA:
            return True
        elif logLR <= logB:
            # Accept H0: The top two are not different.
            return True
        return False


class MsprtConfidenceModelConfig(pydantic.BaseModel):
    priori: typing.Literal["jeffreys", "uniform"]
    alpha: float
    beta: float

class MsprtConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        priori = "uniform",
        alpha = 0.05,
        beta = 0.1,
    ):
        self.config = MsprtConfidenceModelConfig(
            priori = priori,
            alpha = alpha,
            beta = beta,
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
        priori, alpha, beta = self.config.priori, self.config.alpha, self.config.beta

        if priori == "uniform":
            priori_alpha, priori_beta = 1, 1
        elif priori == "jeffreys":
            priori_alpha, priori_beta = 0.5, 0.5
        else:
            raise ValueError("Invalid priori")


        logA = np.log((1 - beta) / alpha)
        logB = np.log(beta / (1 - alpha))

        # Under the null hypothesis (H0), assume p = 0.5 for all trials.
        likelihood_H0 = 0.5 ** (first + second)

        # Compute the normalization constant for the truncated prior.
        norm = 1 - scipy.special.betainc(priori_alpha, priori_beta, 0.5)

        # Compute the integral (in closed form) using the Beta function and the
        # regularized incomplete Beta function.
        numerator = scipy.special.beta(first + priori_alpha, second + priori_beta) * \
                    (1 - scipy.special.betainc(first + priori_alpha, second + priori_beta, 0.5))

        likelihood_H1 = numerator / (scipy.special.beta(priori_alpha, priori_beta) * norm)

        # The Bayes factor K is the ratio of the likelihoods:
        #   K = P(data|H1) / P(data|H0)
        logLR = np.log(likelihood_H1 / likelihood_H0)

        if logLR >= logA:
            return True
        elif logLR <= logB:
            # Accept H0: The top two are not different.
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


class BayesianPosteriorConfidenceModelConfig(pydantic.BaseModel):
    confidence_threshold: float
    priori: typing.Literal["jeffreys", "uniform"]

class BayesianPosteriorConfidenceModel(AbstractConfidenceModel):
    def __init__(self, confidence_threshold = 0.95, priori = "uniform"):
        self.config = BayesianPosteriorConfidenceModelConfig(
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
