"""Exact Simple and Ordinary Kriging built on GPyTorch covariance modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log, pi
from typing import Literal

import gpytorch
import torch
from torch import Tensor

from geopytorch.covariance import (
    COVARIANCE_MODELS,
    CovarianceModel,
    covariance_kernel,
)

type Locations = Tensor | Sequence[Sequence[float]]
type Values = Tensor | Sequence[float]
type LikelihoodMethod = Literal["ml", "reml"]
type OptimizerName = Literal["adam", "lbfgs"]


@dataclass(frozen=True, slots=True)
class KrigingPrediction:
    """Mean and joint covariance of a Kriging prediction."""

    mean: Tensor
    covariance_matrix: Tensor

    @property
    def variance(self) -> Tensor:
        """Marginal prediction variance at every target location."""

        return self.covariance_matrix.diagonal(dim1=-2, dim2=-1)

    @property
    def standard_deviation(self) -> Tensor:
        """Marginal prediction standard deviation at every target location."""

        return self.variance.clamp_min(0).sqrt()

    def confidence_region(self, num_std: float = 2.0) -> tuple[Tensor, Tensor]:
        """Return lower and upper pointwise bounds around the mean."""

        if not isfinite(num_std) or num_std < 0:
            raise ValueError("num_std must be a finite, non-negative number")
        radius = num_std * self.standard_deviation
        return self.mean - radius, self.mean + radius


class _ExactKrigingGP(gpytorch.models.ExactGP):
    """GPyTorch model used to estimate Kriging covariance parameters."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        mean_module: gpytorch.means.Mean,
        covariance_module: gpytorch.kernels.Kernel,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covariance_module

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class _BaseKriging:
    """Shared fitting and covariance-system machinery."""

    def __init__(
        self,
        *,
        kernel: gpytorch.kernels.Kernel | None = None,
        likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None,
        covariance_model: CovarianceModel = "matern",
        matern_nu: float = 1.5,
        ard: bool = False,
    ) -> None:
        if covariance_model not in COVARIANCE_MODELS:
            choices = ", ".join(COVARIANCE_MODELS)
            raise ValueError(f"covariance_model must be one of: {choices}")
        if matern_nu not in (0.5, 1.5, 2.5):
            raise ValueError("matern_nu must be one of 0.5, 1.5, or 2.5")
        if kernel is not None and not isinstance(kernel, gpytorch.kernels.Kernel):
            raise TypeError("kernel must be a gpytorch.kernels.Kernel")
        if likelihood is not None and not isinstance(
            likelihood, gpytorch.likelihoods.GaussianLikelihood
        ):
            raise TypeError("likelihood must be a GaussianLikelihood")

        self._kernel_template = kernel
        self._likelihood_template = likelihood
        self.covariance_model = covariance_model
        self.matern_nu = matern_nu
        self.ard = ard
        self.training_losses_: list[float] = []

    def fit(
        self,
        train_x: Locations,
        train_y: Values,
        *,
        training_iterations: int = 100,
        learning_rate: float = 0.1,
        likelihood_method: LikelihoodMethod = "ml",
        optimizer: OptimizerName = "adam",
    ) -> _BaseKriging:
        """Condition on spatial data and optionally estimate GP hyperparameters.

        Set ``training_iterations=0`` to retain supplied or initialized
        covariance and nugget parameters. Restricted maximum likelihood is
        available for estimators with unknown fixed effects, such as Ordinary
        Kriging. ``lbfgs`` is useful for deterministic dense likelihood fits.
        """

        if isinstance(training_iterations, bool) or not isinstance(
            training_iterations, int
        ):
            raise TypeError("training_iterations must be an integer")
        if training_iterations < 0:
            raise ValueError("training_iterations must be non-negative")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        if likelihood_method not in ("ml", "reml"):
            raise ValueError("likelihood_method must be 'ml' or 'reml'")
        if likelihood_method == "reml" and not self._supports_reml:
            raise ValueError(
                "REML requires unknown fixed effects and is not defined for "
                f"{type(self).__name__}"
            )
        if optimizer not in ("adam", "lbfgs"):
            raise ValueError("optimizer must be 'adam' or 'lbfgs'")

        locations = _as_locations(train_x)
        targets = _as_targets(train_y, locations)
        if locations.shape[0] != targets.shape[0]:
            raise ValueError("train_x and train_y must contain the same number of rows")
        if locations.shape[0] < 2:
            raise ValueError("at least two observations are required")

        kernel = self._make_kernel(locations, targets)
        likelihood = self._make_likelihood(locations, targets)
        mean_module = self._make_mean_module(locations, targets)
        model = _ExactKrigingGP(
            locations, targets, likelihood, mean_module, kernel
        )

        self.train_x_ = locations
        self.train_y_ = targets
        self.input_dimension_ = locations.shape[1]
        self.kernel_ = kernel
        self.likelihood_ = likelihood
        self.model_ = model
        self.training_losses_ = []
        self.likelihood_method_ = likelihood_method
        self.optimizer_ = optimizer

        if training_iterations:
            if likelihood_method == "reml":
                parameters = [
                    parameter
                    for module in (kernel, likelihood)
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
            else:
                parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
            if not parameters:
                raise ValueError("the model has no trainable parameters")

            model.train()
            likelihood.train()
            marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood, model
            )

            def objective() -> Tensor:
                if likelihood_method == "reml":
                    return self._negative_restricted_log_likelihood() / len(targets)
                return -marginal_log_likelihood(model(locations), targets)

            def checked_objective() -> Tensor:
                loss = objective()
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(
                        "non-finite loss encountered while fitting Kriging model"
                    )
                return loss

            if optimizer == "adam":
                torch_optimizer = torch.optim.Adam(parameters, lr=learning_rate)
                for _ in range(training_iterations):
                    torch_optimizer.zero_grad()
                    loss = checked_objective()
                    loss.backward()
                    torch_optimizer.step()
                    self.training_losses_.append(float(loss.detach().item()))
            else:
                torch_optimizer = torch.optim.LBFGS(
                    parameters,
                    lr=learning_rate,
                    max_iter=training_iterations,
                    line_search_fn="strong_wolfe",
                )

                def closure() -> Tensor:
                    torch_optimizer.zero_grad()
                    loss = checked_objective()
                    loss.backward()
                    self.training_losses_.append(float(loss.detach().item()))
                    return loss

                torch_optimizer.step(closure)

        if likelihood_method == "reml":
            mean_module.raw_constant.requires_grad_(False)
            mean_module.initialize(constant=self.estimated_mean_.detach())

        model.eval()
        likelihood.eval()
        return self

    @property
    def _supports_reml(self) -> bool:
        return False

    def _negative_restricted_log_likelihood(self) -> Tensor:
        raise RuntimeError("REML is unavailable for this estimator")

    def _make_mean_module(
        self, train_x: Tensor, train_y: Tensor
    ) -> gpytorch.means.Mean:
        raise NotImplementedError

    def _make_kernel(self, train_x: Tensor, train_y: Tensor) -> gpytorch.kernels.Kernel:
        if self._kernel_template is not None:
            return self._kernel_template.to(device=train_x.device, dtype=train_x.dtype)

        ard_num_dims = train_x.shape[1] if self.ard else None
        kernel = covariance_kernel(
            self.covariance_model,
            matern_nu=self.matern_nu,
            ard_num_dims=ard_num_dims,
        ).to(
            device=train_x.device, dtype=train_x.dtype
        )
        base_kernel = kernel.base_kernel

        target_variance = train_y.var(unbiased=False).clamp_min(1e-3)
        nonzero_distances = torch.pdist(train_x).detach()
        nonzero_distances = nonzero_distances[nonzero_distances > 0]
        lengthscale = (
            nonzero_distances.median()
            if nonzero_distances.numel()
            else train_x.new_tensor(1.0)
        )
        kernel.initialize(outputscale=target_variance)
        base_kernel.initialize(lengthscale=lengthscale.clamp_min(1e-3))
        return kernel

    def _make_likelihood(
        self, train_x: Tensor, train_y: Tensor
    ) -> gpytorch.likelihoods.GaussianLikelihood:
        if self._likelihood_template is not None:
            return self._likelihood_template.to(device=train_x.device, dtype=train_x.dtype)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=train_x.device, dtype=train_x.dtype
        )
        initial_noise = (0.05 * train_y.var(unbiased=False)).clamp_min(1e-3)
        likelihood.initialize(noise=initial_noise)
        return likelihood

    def _prediction_terms(
        self, locations: Locations
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        target_x = self._prediction_locations(locations)
        with torch.no_grad():
            train_covariance = self.kernel_(self.train_x_).to_dense()
            train_covariance = train_covariance.clone()
            train_covariance.diagonal().add_(self._noise())
            cholesky = _stable_cholesky(train_covariance)

            cross_covariance = self.kernel_(self.train_x_, target_x).to_dense()
            solved_cross_covariance = torch.cholesky_solve(
                cross_covariance, cholesky
            )
            target_covariance = self.kernel_(target_x).to_dense()
        return (
            target_x,
            cross_covariance,
            target_covariance,
            solved_cross_covariance,
            cholesky,
        )

    def _prediction_locations(self, locations: Locations) -> Tensor:
        self._require_fitted()
        target_x = _as_locations(
            locations, dtype=self.train_x_.dtype, device=self.train_x_.device
        )
        if target_x.shape[1] != self.input_dimension_:
            raise ValueError(
                f"prediction locations must have {self.input_dimension_} columns"
            )
        return target_x

    def _noise(self) -> Tensor:
        noise = self.likelihood_.noise.reshape(-1)
        if noise.numel() != 1:
            raise RuntimeError("only homoskedastic scalar Gaussian noise is supported")
        return noise[0]

    def _finalize_covariance(
        self, covariance: Tensor, *, observation_noise: bool
    ) -> Tensor:
        covariance = 0.5 * (covariance + covariance.mT)
        if observation_noise:
            covariance = covariance.clone()
            covariance.diagonal().add_(self._noise())
        covariance = covariance.clone()
        covariance.diagonal().clamp_min_(0)
        return covariance.detach()

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")


class SimpleKriging(_BaseKriging):
    """Simple Kriging with a known constant mean."""

    def __init__(
        self,
        mean: float,
        *,
        kernel: gpytorch.kernels.Kernel | None = None,
        likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None,
        covariance_model: CovarianceModel = "matern",
        matern_nu: float = 1.5,
        ard: bool = False,
    ) -> None:
        if not isfinite(mean):
            raise ValueError("mean must be finite")
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            covariance_model=covariance_model,
            matern_nu=matern_nu,
            ard=ard,
        )
        self.mean = float(mean)

    def _make_mean_module(
        self, train_x: Tensor, train_y: Tensor
    ) -> gpytorch.means.ConstantMean:
        mean_module = gpytorch.means.ConstantMean().to(
            device=train_x.device, dtype=train_x.dtype
        )
        mean_module.initialize(constant=self.mean)
        mean_module.raw_constant.requires_grad_(False)
        return mean_module

    def weights(self, locations: Locations) -> Tensor:
        """Return Simple-Kriging weights with shape ``(n_targets, n_train)``."""

        _, _, _, solved_cross_covariance, _ = self._prediction_terms(locations)
        return solved_cross_covariance.mT

    def predict(
        self, locations: Locations, *, observation_noise: bool = False
    ) -> KrigingPrediction:
        """Predict the latent signal, optionally including future observation noise."""

        (
            _,
            cross_covariance,
            target_covariance,
            solved_cross_covariance,
            _,
        ) = self._prediction_terms(locations)
        weights = solved_cross_covariance.mT
        known_mean = self.train_y_.new_tensor(self.mean)
        mean = known_mean + weights @ (self.train_y_ - known_mean)
        covariance = target_covariance - cross_covariance.mT @ solved_cross_covariance
        covariance = self._finalize_covariance(
            covariance, observation_noise=observation_noise
        )
        return KrigingPrediction(mean=mean, covariance_matrix=covariance)


class OrdinaryKriging(_BaseKriging):
    """Ordinary Kriging with an unknown constant mean and unit-sum weights."""

    @property
    def _supports_reml(self) -> bool:
        return True

    def _make_mean_module(
        self, train_x: Tensor, train_y: Tensor
    ) -> gpytorch.means.ConstantMean:
        mean_module = gpytorch.means.ConstantMean().to(
            device=train_x.device, dtype=train_x.dtype
        )
        mean_module.initialize(constant=train_y.mean())
        return mean_module

    def _ordinary_terms(
        self, locations: Locations
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            _,
            cross_covariance,
            target_covariance,
            solved_cross_covariance,
            cholesky,
        ) = self._prediction_terms(locations)
        ones = self.train_y_.new_ones((self.train_y_.shape[0], 1))
        solved_ones = torch.cholesky_solve(ones, cholesky)
        denominator = (ones.mT @ solved_ones).squeeze()
        if not bool(torch.isfinite(denominator).item()) or denominator <= 0:
            raise RuntimeError("Ordinary Kriging mean system is not positive definite")

        residual = 1 - ones.mT @ solved_cross_covariance
        corrected_solve = solved_cross_covariance + solved_ones @ (
            residual / denominator
        )
        return (
            cross_covariance,
            target_covariance,
            solved_cross_covariance,
            corrected_solve,
            residual.mT @ (residual / denominator),
        )

    def _negative_restricted_log_likelihood(self) -> Tensor:
        """Return the dense Gaussian REML objective for an unknown constant."""

        train_covariance = self.kernel_(self.train_x_).to_dense().clone()
        train_covariance.diagonal().add_(self._noise())
        cholesky = _stable_cholesky(train_covariance)
        ones = self.train_y_.new_ones((self.train_y_.shape[0], 1))
        targets = self.train_y_.unsqueeze(-1)
        solved_ones = torch.cholesky_solve(ones, cholesky)
        solved_targets = torch.cholesky_solve(targets, cholesky)
        fixed_effect_information = (ones.mT @ solved_ones).squeeze()
        if (
            not bool(torch.isfinite(fixed_effect_information).item())
            or fixed_effect_information <= 0
        ):
            raise RuntimeError("REML fixed-effect system is not positive definite")

        estimated_mean = (ones.mT @ solved_targets) / fixed_effect_information
        residual = targets - ones * estimated_mean
        solved_residual = torch.cholesky_solve(residual, cholesky)
        quadratic_form = (residual.mT @ solved_residual).squeeze()
        log_determinant = 2 * torch.log(cholesky.diagonal()).sum()
        fixed_effect_log_determinant = torch.log(fixed_effect_information)
        degrees_of_freedom = self.train_y_.shape[0] - 1
        return 0.5 * (
            degrees_of_freedom * log(2 * pi)
            + log_determinant
            + fixed_effect_log_determinant
            + quadratic_form
        )

    @property
    def restricted_log_likelihood_(self) -> Tensor:
        """Restricted log likelihood at the fitted covariance parameters."""

        self._require_fitted()
        with torch.no_grad():
            return -self._negative_restricted_log_likelihood().detach()

    @property
    def estimated_mean_(self) -> Tensor:
        """Generalized least-squares estimate of the unknown constant mean."""

        self._require_fitted()
        with torch.no_grad():
            train_covariance = self.kernel_(self.train_x_).to_dense().clone()
            train_covariance.diagonal().add_(self._noise())
            cholesky = _stable_cholesky(train_covariance)
            ones = self.train_y_.new_ones((self.train_y_.shape[0], 1))
            solved_ones = torch.cholesky_solve(ones, cholesky)
            solved_targets = torch.cholesky_solve(
                self.train_y_.unsqueeze(-1), cholesky
            )
            return ((ones.mT @ solved_targets) / (ones.mT @ solved_ones)).squeeze()

    def weights(self, locations: Locations) -> Tensor:
        """Return unit-sum Ordinary-Kriging weights, targets by observations."""

        _, _, _, corrected_solve, _ = self._ordinary_terms(locations)
        return corrected_solve.mT

    def predict(
        self, locations: Locations, *, observation_noise: bool = False
    ) -> KrigingPrediction:
        """Predict with exact Ordinary-Kriging mean and covariance corrections."""

        (
            cross_covariance,
            target_covariance,
            solved_cross_covariance,
            corrected_solve,
            mean_uncertainty,
        ) = self._ordinary_terms(locations)
        weights = corrected_solve.mT
        mean = weights @ self.train_y_

        covariance = (
            target_covariance
            - cross_covariance.mT @ solved_cross_covariance
            + mean_uncertainty
        )
        covariance = self._finalize_covariance(
            covariance, observation_noise=observation_noise
        )
        return KrigingPrediction(mean=mean, covariance_matrix=covariance)


def _as_locations(
    values: Locations,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    locations = torch.as_tensor(values, dtype=dtype, device=device)
    if not locations.is_floating_point():
        locations = locations.to(dtype=torch.get_default_dtype())
    if locations.dtype not in (torch.float32, torch.float64):
        locations = locations.to(dtype=torch.get_default_dtype())
    if locations.ndim != 2:
        raise ValueError("locations must have shape (n_samples, n_dimensions)")
    if locations.shape[0] == 0 or locations.shape[1] == 0:
        raise ValueError("locations must have at least one row and one column")
    if not bool(torch.isfinite(locations).all().item()):
        raise ValueError("locations must contain only finite values")
    return locations


def _as_targets(values: Values, locations: Tensor) -> Tensor:
    targets = torch.as_tensor(values, dtype=locations.dtype, device=locations.device)
    if targets.ndim != 1:
        raise ValueError("train_y must have shape (n_samples,)")
    if not bool(torch.isfinite(targets).all().item()):
        raise ValueError("train_y must contain only finite values")
    return targets


def _stable_cholesky(matrix: Tensor) -> Tensor:
    matrix = 0.5 * (matrix + matrix.mT)
    try:
        return torch.linalg.cholesky(matrix)
    except RuntimeError as original_error:
        diagonal_scale = matrix.diagonal().abs().mean().clamp_min(1.0)
        base_jitter = (1e-6 if matrix.dtype == torch.float32 else 1e-10) * diagonal_scale
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        for multiplier in (1.0, 10.0, 100.0, 1000.0):
            try:
                return torch.linalg.cholesky(matrix + multiplier * base_jitter * identity)
            except RuntimeError:
                continue
        raise RuntimeError("Kriging covariance matrix is not positive definite") from original_error
