"""Compound Poisson--Gamma density evaluated with PyTorch tensors."""

from math import log
from typing import ClassVar

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class Tweedie(Distribution):
    """Tweedie distribution with mean mu and variance dispersion * mu**power.

    Power lies strictly between one and two. Zero has positive probability;
    positive responses have a continuous density. Series terms stay on the
    input device and retain autograd derivatives. A geometric bound on both
    omitted tails controls truncation. Exceeding max_series_terms raises an
    error instead of returning an unchecked approximation.
    """

    arg_constraints: ClassVar = {
        "mu": constraints.positive,
        "dispersion": constraints.positive,
        "power": constraints.interval(1.0, 2.0),
    }
    support = constraints.nonnegative
    has_rsample = False

    def __init__(
        self,
        mu,
        dispersion,
        power,
        *,
        max_series_terms=16385,
        series_tolerance=None,
        validate_args=None,
    ):
        self.mu, self.dispersion, self.power = broadcast_all(mu, dispersion, power)
        if bool(((self.power <= 1) | (self.power >= 2)).any()):
            raise ValueError("Tweedie power must lie strictly between one and two")
        if (
            isinstance(max_series_terms, bool)
            or not isinstance(max_series_terms, int)
            or max_series_terms < 33
        ):
            raise ValueError("max_series_terms must be an integer of at least 33")
        self.max_series_terms = max_series_terms
        self.series_tolerance = (
            (1e-12 if self.mu.dtype == torch.float64 else 1e-6)
            if series_tolerance is None
            else float(series_tolerance)
        )
        if not 0 < self.series_tolerance < 1:
            raise ValueError("series_tolerance must lie strictly between zero and one")
        super().__init__(self.mu.shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.dispersion * self.mu.pow(self.power)

    @property
    def log_zero_probability(self):
        return -self.mu.pow(2 - self.power) / (self.dispersion * (2 - self.power))

    def _log_series(self, y, phi, power):
        alpha = (2 - power) / (power - 1)
        log_z = (
            alpha * y.log()
            - (1 + alpha) * phi.log()
            - (2 - power).log()
            - alpha * (power - 1).log()
        )
        with torch.no_grad():
            center = (y.pow(2 - power) / (phi * (2 - power))).floor().clamp_min(1)
            if not bool(torch.isfinite(center).all()):
                raise FloatingPointError("Tweedie series center is not finite")
            # Consecutive integers must remain distinguishable at this precision.
            if bool((center * torch.finfo(y.dtype).eps >= 0.125).any()):
                raise FloatingPointError(
                    "Tweedie series needs higher precision for these parameters"
                )

        def term(j):
            return (
                j * log_z[..., None]
                - torch.lgamma(j + 1)
                - torch.lgamma(j * alpha[..., None])
            )

        radius = 16
        while True:
            index = center[..., None] + torch.arange(
                -radius, radius + 1, device=y.device, dtype=y.dtype
            )
            terms = term(index.clamp_min(1)).masked_fill(index < 1, -torch.inf)
            log_sum = torch.logsumexp(terms, dim=-1)
            with torch.no_grad():
                # The positive terms are log-concave in the series index.
                upper = center + radius + 1
                upper_terms = term(torch.stack((upper, upper + 1), dim=-1))
                upper_ratio = upper_terms[..., 1] - upper_terms[..., 0]
                upper_bound = upper_terms[..., 0] - torch.log(
                    -torch.expm1(upper_ratio.clamp_max(-torch.finfo(y.dtype).tiny))
                )
                upper_bound = torch.where(upper_ratio < 0, upper_bound, torch.inf)
                lower = center - radius - 1
                lower_terms = term(
                    torch.stack((lower.clamp_min(1), (lower - 1).clamp_min(1)), dim=-1)
                )
                lower_ratio = lower_terms[..., 1] - lower_terms[..., 0]
                lower_bound = lower_terms[..., 0] - torch.log(
                    -torch.expm1(lower_ratio.clamp_max(-torch.finfo(y.dtype).tiny))
                )
                lower_bound = torch.where(lower_ratio < 0, lower_bound, torch.inf)
                lower_bound = torch.where(lower == 1, lower_terms[..., 0], lower_bound)
                lower_bound = torch.where(lower < 1, -torch.inf, lower_bound)
                omitted = torch.logaddexp(lower_bound, upper_bound)
                converged = omitted <= log_sum.detach() + log(self.series_tolerance)
            if bool(converged.all()):
                return log_sum
            new_radius = min(2 * radius, (self.max_series_terms - 1) // 2)
            if new_radius <= radius:
                raise FloatingPointError(
                    "Tweedie series did not meet its tail tolerance; increase max_series_terms"
                )
            radius = new_radius

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y, mu, phi, power = broadcast_all(value, self.mu, self.dispersion, self.power)
        result = -mu.pow(2 - power) / (phi * (2 - power))
        positive = y > 0
        if bool(positive.any()):
            yp, mup, phip, pp = (
                y[positive],
                mu[positive],
                phi[positive],
                power[positive],
            )
            log_series = self._log_series(yp, phip, pp)
            result = result.clone()
            result[positive] = (
                result[positive]
                - yp / (phip * (pp - 1) * mup.pow(pp - 1))
                + log_series
                - yp.log()
            )
        return result

    @torch.no_grad()
    def sample(self, sample_shape=()):
        shape = self._extended_shape(torch.Size(sample_shape))
        rate = self.mu.pow(2 - self.power) / (self.dispersion * (2 - self.power))
        count = torch.poisson(rate.expand(shape))
        alpha = (2 - self.power) / (self.power - 1)
        scale = self.dispersion * (self.power - 1) * self.mu.pow(self.power - 1)
        positive = torch.distributions.Gamma(
            count.clamp_min(1) * alpha, 1 / scale
        ).sample()
        return torch.where(count > 0, positive, 0)


__all__ = ["Tweedie"]
