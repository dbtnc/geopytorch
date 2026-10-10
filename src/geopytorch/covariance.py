# Expected inputs: Coordinate tensors and positive range parameters.
# Expected outputs: GPyTorch kernels for supported stationary covariance models.

"""Covariance kernels built on GPyTorch."""

from __future__ import annotations

from math import pi
from typing import Literal

import gpytorch
import torch
from torch import Tensor

type CovarianceModel = Literal[
    "gaussian", "exponential", "matern", "spherical", "wave"
]
COVARIANCE_MODELS: tuple[CovarianceModel, ...] = (
    "gaussian",
    "exponential",
    "matern",
    "spherical",
    "wave",
)


class DistanceKernel(gpytorch.kernels.Kernel):
    r"""Base for stationary kernels expressed through scaled distance.

    ``has_lengthscale`` asks GPyTorch to provide a positive trainable range,
    including batching and optional per-coordinate ARD parameters.  It does
    not, by itself, define how that parameter enters a covariance formula.
    This class supplies one convention shared by all subclasses:

    .. math::
        r = \left\lVert \mathbf{x}_1/\boldsymbol{\phi}
            - \mathbf{x}_2/\boldsymbol{\phi}\right\rVert_2.

    Each subclass therefore receives the same dimensionless distance ``r``
    and states its correlation directly in terms of ``r``.  Centralizing this
    step prevents hidden, model-specific distance factors from changing the
    geostatistical meaning of the public range parameter ``phi`` (stored by
    GPyTorch under its standard ``lengthscale`` property).
    """

    has_lengthscale = True

    def _scaled_distance(
        self, x1: Tensor, x2: Tensor, *, diag: bool, **params: object
    ) -> Tensor:
        return self.covar_dist(
            x1.div(self.lengthscale),
            x2.div(self.lengthscale),
            diag=diag,
            **params,
        )


class GaussianKernel(DistanceKernel):
    r"""Gaussian correlation :math:`\exp[-(h/\phi)^2]`."""

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object
    ) -> Tensor:
        distance = self._scaled_distance(x1, x2, diag=diag, **params)
        return torch.exp(-distance.square())


class ExponentialKernel(DistanceKernel):
    r"""Exponential correlation :math:`\exp(-h/\phi)`."""

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object
    ) -> Tensor:
        distance = self._scaled_distance(x1, x2, diag=diag, **params)
        return torch.exp(-distance)


class MaternKernel(DistanceKernel):
    r"""Matérn correlation with Bessel argument :math:`h/\phi`.

    The supported smoothness values match GPyTorch's exact Matérn kernel:
    ``0.5``, ``1.5``, and ``2.5``. Unlike GPyTorch's native parameterization,
    ``lengthscale`` here is the geostatistical range ``phi`` without a
    :math:`\sqrt{2\nu}` factor.
    """

    def __init__(self, nu: float = 1.5, **kwargs: object) -> None:
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be one of 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.nu = float(nu)

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object
    ) -> Tensor:
        distance = self._scaled_distance(x1, x2, diag=diag, **params)
        exponential = torch.exp(-distance)
        if self.nu == 0.5:
            polynomial: Tensor | float = 1.0
        elif self.nu == 1.5:
            polynomial = 1.0 + distance
        else:
            polynomial = 1.0 + distance + distance.square() / 3.0
        return polynomial * exponential


class SphericalKernel(DistanceKernel):
    r"""Spherical correlation with compact-support range ``phi``."""

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object
    ) -> Tensor:
        _require_at_most_three_dimensions(x1, "SphericalKernel")
        distance = self._scaled_distance(x1, x2, diag=diag, **params)
        within_range = 1.0 - 1.5 * distance + 0.5 * distance.pow(3)
        return torch.where(
            distance < 1.0, within_range, torch.zeros_like(distance)
        )


class WaveKernel(DistanceKernel):
    r"""Wave correlation :math:`(\phi/h)\sin(h/\phi)`."""

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, **params: object
    ) -> Tensor:
        _require_at_most_three_dimensions(x1, "WaveKernel")
        distance = self._scaled_distance(x1, x2, diag=diag, **params)
        return torch.sinc(distance / pi)


def covariance_kernel(
    covariance_model: CovarianceModel = "matern",
    *,
    matern_nu: float = 1.5,
    ard_num_dims: int | None = None,
) -> gpytorch.kernels.ScaleKernel:
    """Return a scaled kernel using partial sill ``sigma^2`` and range ``phi``.

    The returned ``ScaleKernel.outputscale`` is the partial sill ``sigma^2``.
    Its base kernel's ``lengthscale`` is the range ``phi`` used directly in
    the selected geostatistical correlation formula.
    """

    if covariance_model not in COVARIANCE_MODELS:
        choices = ", ".join(COVARIANCE_MODELS)
        raise ValueError(f"covariance_model must be one of: {choices}")
    kernel_options = {"ard_num_dims": ard_num_dims}
    if covariance_model == "gaussian":
        base_kernel = GaussianKernel(**kernel_options)
    elif covariance_model == "exponential":
        base_kernel = ExponentialKernel(**kernel_options)
    elif covariance_model == "matern":
        base_kernel = MaternKernel(nu=matern_nu, **kernel_options)
    elif covariance_model == "spherical":
        base_kernel = SphericalKernel(**kernel_options)
    else:
        base_kernel = WaveKernel(**kernel_options)
    return gpytorch.kernels.ScaleKernel(base_kernel)


def _require_at_most_three_dimensions(x: Tensor, kernel_name: str) -> None:
    if x.shape[-1] > 3:
        raise ValueError(
            f"{kernel_name} is positive definite only for up to 3 spatial dimensions"
        )


__all__ = [
    "COVARIANCE_MODELS",
    "CovarianceModel",
    "ExponentialKernel",
    "GaussianKernel",
    "MaternKernel",
    "SphericalKernel",
    "WaveKernel",
    "covariance_kernel",
]
