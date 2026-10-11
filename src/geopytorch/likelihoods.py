"""Conditional distributions for spatial generalized linear models."""

from math import log, pi
from typing import ClassVar

import gpytorch
import torch
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

GLM_FAMILIES = ("poisson", "nbinomial", "binomial", "beta", "Gamma", "inverse.gaussian")


class _Binomial(torch.distributions.Binomial):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # This form also has the correct second derivative at zero logits.
        return (
            torch.lgamma(self.total_count + 1)
            - torch.lgamma(value + 1)
            - torch.lgamma(self.total_count - value + 1)
            + value * torch.nn.functional.logsigmoid(self.logits)
            + (self.total_count - value) * torch.nn.functional.logsigmoid(-self.logits)
        )


class _InverseGaussian(torch.distributions.Distribution):
    arg_constraints: ClassVar = {
        "loc": constraints.positive,
        "concentration": constraints.positive,
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        super().__init__(self.loc.shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.loc.pow(3) / self.concentration

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (self.concentration.log() - log(2 * pi) - 3 * value.log()) - (
            self.concentration * (value / self.loc - 1).square() / (2 * value)
        )

    @torch.no_grad()
    def sample(self, sample_shape=()):
        shape = self._extended_shape(torch.Size(sample_shape))
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device).square()
        a = self.loc * z / (2 * self.concentration)
        x = self.loc / (1 + a + torch.sqrt(a * (2 + a)))
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return torch.where(u <= self.loc / (self.loc + x), x, self.loc.square() / x)


class SpatialGLMLikelihood(_OneDimensionalLikelihood):
    """GLM likelihoods with log or logit links and a mean/precision parameterization.

    ``dispersion`` is fixed at one for Poisson and binomial. Otherwise it is
    negative-binomial size, beta precision, Gamma shape, or inverse-Gaussian
    shape divided by its mean. The inverse-Gaussian variance is mu**2/dispersion.
    Binomial samples are success counts; ``trials`` defaults to one.
    """

    def __init__(self, family, dispersion=1.0):
        super().__init__()
        if family not in GLM_FAMILIES:
            raise ValueError(f"family must be one of {GLM_FAMILIES}")
        self.family = family
        self.has_dispersion = family not in ("poisson", "binomial")
        self.register_parameter("raw_dispersion", torch.nn.Parameter(torch.zeros(())))
        self.register_constraint(
            "raw_dispersion",
            gpytorch.constraints.Positive(transform=torch.exp, inv_transform=torch.log),
        )
        self.dispersion = dispersion
        self.raw_dispersion.requires_grad_(self.has_dispersion)

    @property
    def dispersion(self):
        return self.raw_dispersion_constraint.transform(self.raw_dispersion)

    @dispersion.setter
    def dispersion(self, value):
        value = torch.as_tensor(
            value, dtype=self.raw_dispersion.dtype, device=self.raw_dispersion.device
        )
        if value.numel() != 1 or not bool(torch.isfinite(value)) or value <= 0:
            raise ValueError("dispersion must be a finite positive scalar")
        if not self.has_dispersion and value != 1:
            raise ValueError("dispersion must be one for Poisson and binomial")
        self.initialize(
            raw_dispersion=self.raw_dispersion_constraint.inverse_transform(
                value.reshape(())
            )
        )

    def forward(self, function_samples, *args, trials=1, **kwargs):
        eta, phi = function_samples, self.dispersion
        if self.family == "binomial":
            return _Binomial(total_count=trials, logits=eta)
        if self.family == "beta":
            # Separate sigmoids avoid losing the second shape in the positive tail.
            return torch.distributions.Beta(phi * eta.sigmoid(), phi * (-eta).sigmoid())
        if self.family == "poisson":
            return torch.distributions.Poisson(eta.exp())
        if self.family == "nbinomial":
            return torch.distributions.NegativeBinomial(
                total_count=phi, logits=eta - phi.log()
            )
        if self.family == "Gamma":
            return torch.distributions.Gamma(concentration=phi, rate=phi * (-eta).exp())
        mu = eta.exp()
        return _InverseGaussian(mu, phi * mu)

    def inverse_link(self, eta, trials=1):
        if self.family in ("binomial", "beta"):
            return eta.sigmoid() * (trials if self.family == "binomial" else 1)
        return eta.exp()

    def link_derivative(self, eta, trials=1):
        if self.family in ("binomial", "beta"):
            return (
                eta.sigmoid()
                * (-eta).sigmoid()
                * (trials if self.family == "binomial" else 1)
            )
        return eta.exp()


__all__ = ["GLM_FAMILIES", "SpatialGLMLikelihood"]
