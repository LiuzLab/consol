import typing
import abc

import numpy as np
import scipy.stats
import scipy.special
import pydantic

class AbstractConfidenceModelConfig(abc.ABC, pydantic.BaseModel):
    max_trials: int

class AbstractConfidenceModel(abc.ABC):
    @abc.abstractmethod
    def test(self, first: int, second: int) -> bool:
        pass

class SprtConfidenceModelConfig(AbstractConfidenceModelConfig):
    p1: float
    alpha: float
    beta: float

class SprtConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        max_trials=256,
        p1 = 0.5001,
        alpha = 0.05,
        # beta = 1-.05-.000030,
        beta = 1-.05-.000024,
        # beta = 1-.05-.000018,
    ):
        self.config = SprtConfidenceModelConfig(
            max_trials = max_trials,
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


class MsprtConfidenceModelConfig(AbstractConfidenceModelConfig):
    priori_alpha: float
    priori_beta: float
    alpha: float
    beta: float

class MsprtConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        max_trials=256,
        priori_alpha=1e6,
        priori_beta=1e6,
        alpha = 0.05,
        # beta = 1-.05-.00009,
        beta = 1-.05-.00006,
        # beta = 1-.05-.00003,
    ):
        self.config = MsprtConfidenceModelConfig(
            max_trials = max_trials,
            priori_alpha = priori_alpha,
            priori_beta = priori_beta,
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
        priori_alpha, priori_beta, alpha, beta = self.config.priori_alpha, self.config.priori_beta, self.config.alpha, self.config.beta

        logA = np.log((1 - beta) / alpha)
        logB = np.log(beta / (1 - alpha))

        # Under the null hypothesis (H0), assume p = 0.5 for all trials.
        log_likelihood_H0 = np.log(0.5) * (first + second)

        # Compute the normalization constant for the truncated prior.
        norm = 1 - scipy.special.betainc(priori_alpha, priori_beta, 0.5)

        # Compute the integral (in closed form) using the Beta function and the
        # regularized incomplete Beta function.
        log_numerator = scipy.special.betaln(first + priori_alpha, second + priori_beta) + \
                    np.log(1 - scipy.special.betainc(first + priori_alpha, second + priori_beta, 0.5))

        log_likelihood_H1 = log_numerator - scipy.special.betaln(priori_alpha, priori_beta) - np.log(norm)

        # The Bayes factor K is the ratio of the likelihoods:
        #   K = P(data|H1) / P(data|H0)
        logLR = log_likelihood_H1 - log_likelihood_H0

        if logLR >= logA:
            return True
        elif logLR <= logB:
            # Accept H0: The top two are not different.
            return True
        return False


class PvalueConfidenceModelConfig(AbstractConfidenceModelConfig):
    pvalue_threshold: float

class PValueConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        max_trials=40,
        pvalue_threshold = 0.05
    ):
        self.config = PvalueConfidenceModelConfig(
            max_trials=max_trials,
            pvalue_threshold=pvalue_threshold
        )

    def test(self, first, second) -> bool:
        pvalue_threshold = self.config.pvalue_threshold
        pvalue = scipy.stats.binomtest(first, first+second, p=0.5, alternative='greater').pvalue
        if pvalue <= pvalue_threshold:
            return True
        return False


class BayesianPosteriorConfidenceModelConfig(AbstractConfidenceModelConfig):
    confidence_threshold: float
    priori: typing.Literal["jeffreys", "uniform"]

class BayesianPosteriorConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        max_trials=40,
        confidence_threshold = 0.95,
        priori = "uniform"
    ):
        self.config = BayesianPosteriorConfidenceModelConfig(
            max_trials=max_trials,
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

class VoteConfidenceModelConfig(AbstractConfidenceModelConfig):
    pass

class VoteConfidenceModel(AbstractConfidenceModel):
    def __init__(
        self,
        max_trials=40,
    ):
        self.config = VoteConfidenceModelConfig(
            max_trials=max_trials
        )
    def test(self, first, second) -> bool:
        _ = first, second  # Avoid linting error for unused arguments
        return False
