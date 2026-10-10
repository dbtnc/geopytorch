"""Gaussian spatial linear model and universal kriging using GPyTorch.

Coordinates and regression covariates are separate inputs.
Design matrices include their own intercept column.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from math import log, pi, sqrt

import gpytorch
import numpy as np
import torch
from linear_operator.utils.errors import NanError, NotPSDError
from torch import Tensor

from geopytorch.covariance import (
    COVARIANCE_MODELS,
    CovarianceModel,
    SphericalKernel,
    WaveKernel,
)
from geopytorch.kriging import KrigingPrediction, LikelihoodMethod, Locations, Values


@contextmanager
def _exact_settings():
    """Use deterministic exact factorization for this dense reference estimator."""
    with (
        gpytorch.settings.max_cholesky_size(1_000_000),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ),
        gpytorch.settings.fast_pred_var(False),
    ):
        yield


def _array(value, name, ndim, *, dtype=torch.float64, device=None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().to(dtype=dtype, device=device).clone()
    else:
        result = torch.tensor(np.asarray(value).copy(), dtype=dtype, device=device)
    if result.ndim != ndim or not result.numel() or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def _theta(value) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or \
            np.any(value[:2] <= 0) or value[2] < 0:
        raise ValueError("initial parameters require positive sill/range and nonnegative nugget")
    return value


class _RegressionGP(gpytorch.models.ExactGP):
    def __init__(self, inputs, targets, likelihood, kernel, spatial_dimensions, columns):
        super().__init__(inputs, targets, likelihood)
        self.spatial_dimensions = spatial_dimensions
        self.mean_module = gpytorch.means.LinearMean(columns, bias=False).to(inputs)
        self.mean_module.weights.requires_grad_(False)
        self.covar_module = kernel

    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs[:, self.spatial_dimensions:]), self.covar_module(inputs)
        )


class SpatialLinearModel:
    """Fit ``y = X beta + spatial GP + independent Gaussian noise``.

    Native RBF/Matérn kernels, GaussianLikelihood, LinearMean, ExactGP and
    ExactMarginalLogLikelihood supply GP calculations. GLS profiles beta out
    of ML/REML; universal kriging adds its estimation-uncertainty correction.
    Spherical and wave models reuse geopytorch's existing custom kernels.
    Inference is dense, with O(n^3) factorization cost and O(n^2) storage.
    """

    def __init__(self, covariance_model: CovarianceModel = "exponential", *, matern_nu: float = 1.5):
        if covariance_model not in COVARIANCE_MODELS:
            raise ValueError(f"covariance_model must be one of {COVARIANCE_MODELS}")
        if matern_nu not in (0.5, 1.5, 2.5):
            raise ValueError("matern_nu must be one of 0.5, 1.5, or 2.5")
        self.covariance_model = covariance_model
        self.matern_nu = float(matern_nu)
        self._fitted = False

    def fit(
        self,
        train_x: Locations,
        train_y: Values,
        design_matrix: Locations,
        *,
        likelihood_method: LikelihoodMethod = "reml",
        initial=None,
        training_iterations: int = 600,
        starts=None,
        parameter_bounds=None,
    ) -> SpatialLinearModel:
        """Estimate covariance by torch.optim.LBFGS with autograd and GLS profiling.

        ``initial`` is (partial sill, phi, nugget). By default use OLS residual
        variance, one third of the median nonzero distance, and 10% noise.
        ``training_iterations=0`` retains initial covariance parameters.
        ``starts`` optionally supplies several positive triples. ``parameter_bounds``
        optionally supplies three (lower, upper) pairs in physical units; None
        endpoints are unbounded, subject to positivity. Optimization starts must
        lie strictly inside the bounds. GPyTorch constraints transform the raw
        parameters; finite bounds use sigmoid transforms, so boundary optima are
        approached from the interior. Fixed parameters can have zero nugget and
        can lie on bounds. Smoothness is fixed, covariance is isotropic, and no
        priors are used.

        Tensor coordinates retain their float32/float64 dtype and device; other
        inputs default to CPU float64. Numerical failures retain the best finite
        parameters from that start and do not stop other starts. Failures are
        recorded in ``optimizer_results_``. If the selected result did not
        converge, ``converged_`` is False and a RuntimeWarning is emitted.
        If no start has a finite result, fitting raises RuntimeError. Diagnostics
        report LBFGS gradient/change stopping criteria and iteration/evaluation
        limits; ``gradient_max`` is measured in transformed raw parameters.
        """
        self._fitted = False
        self.converged_ = False
        if likelihood_method not in ("ml", "reml"):
            raise ValueError("likelihood_method must be 'ml' or 'reml'")
        if isinstance(training_iterations, bool) or not isinstance(training_iterations, int):
            raise TypeError("training_iterations must be an integer")
        if training_iterations < 0:
            raise ValueError("training_iterations must be nonnegative")
        dtype = train_x.dtype if isinstance(train_x, Tensor) else torch.float64
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float64
        device = train_x.device if isinstance(train_x, Tensor) else None
        self.train_x_ = _array(train_x, "train_x", 2, dtype=dtype, device=device)
        options = {"dtype": dtype, "device": self.train_x_.device}
        self.train_y_ = _array(train_y, "train_y", 1, **options)
        self.design_matrix_ = _array(design_matrix, "design_matrix", 2, **options)
        self.n_, self.p_ = self.design_matrix_.shape
        if len(self.train_x_) != self.n_ or len(self.train_y_) != self.n_ or self.n_ <= self.p_:
            raise ValueError("inputs must have matching rows and n > p")
        if int(torch.linalg.matrix_rank(self.design_matrix_)) != self.p_:
            raise ValueError("design_matrix must have full column rank")
        self.input_dimension_ = self.train_x_.shape[1]
        if self.covariance_model in ("spherical", "wave") and self.input_dimension_ > 3:
            raise ValueError("spherical and wave support at most 3 spatial dimensions")
        self.likelihood_method_ = likelihood_method
        self._inputs = torch.cat((self.train_x_, self.design_matrix_), dim=1)
        self._parameter_bounds = self._check_bounds(parameter_bounds)
        self._build_model()
        if initial is None:
            beta = torch.linalg.lstsq(self.design_matrix_, self.train_y_).solution
            variance = (self.train_y_ - self.design_matrix_ @ beta).square().mean()
            if variance <= 0:
                variance = self.train_y_.new_tensor(1.0)
            distances = torch.pdist(self.train_x_)
            distances = distances[distances > 0]
            phi = distances.median().item()/3 if distances.numel() else 1.0
            initial = (0.9 * variance.item(), phi, 0.1 * variance.item())
        initial = _theta(initial)
        self.optimizer_results_ = []
        self.training_losses_ = []
        with _exact_settings():
            if training_iterations:
                initial_values = [initial] if starts is None else [_theta(x) for x in starts]
                if not len(initial_values):
                    raise ValueError("starts must contain at least one parameter triple")
                for start in initial_values:
                    if np.any(start <= 0):
                        raise ValueError("optimization starts must be strictly positive")
                    if any(v <= lo or (hi is not None and v >= hi)
                           for v, (lo, hi) in zip(start, self._parameter_bounds)):
                        raise ValueError("optimization starts must be strictly within parameter_bounds")
                    self._set_theta(start)
                    if not all(bool(torch.isfinite(p).all()) for p in self._parameters):
                        raise ValueError("optimization starts must be strictly within parameter_bounds at the model dtype")
                    self.optimizer_results_.append(self._optimize(training_iterations))
                candidates = [r for r in self.optimizer_results_ if np.isfinite(r["objective"])]
                if not candidates:
                    messages = "; ".join(r["message"] for r in self.optimizer_results_)
                    raise RuntimeError(f"no optimization start produced a finite objective: {messages}")
                best = min(candidates, key=lambda r: r["objective"])
                self._set_theta(best["parameters"])
                self.converged_ = best["success"]
                if not self.converged_:
                    warnings.warn(f"spatial linear model did not converge: {best['message']}", RuntimeWarning)
            else:
                if starts is not None:
                    raise ValueError("starts require training_iterations > 0")
                if any(v < lo or (hi is not None and v > hi)
                       for v, (lo, hi) in zip(initial, self._parameter_bounds)):
                    raise ValueError("initial parameters must be within parameter_bounds")
                self._set_theta(initial)
                self.converged_ = True
            with torch.no_grad():
                beta, information_chol, ml, reml = self._profile()
                self.coef_ = beta.detach()
                self.coef_covariance_ = torch.cholesky_solve(
                    torch.eye(self.p_, **options), information_chol
                ).detach()
                self.log_likelihood_ = ml.detach()
                self.restricted_log_likelihood_ = reml.detach()
                self.reml_design_constant_ = 0.5 * torch.linalg.slogdet(
                    self.design_matrix_.mT @ self.design_matrix_
                ).logabsdet.detach()
                self.model_.mean_module.weights.copy_(self.coef_[:, None])
        self.model_.eval()
        self.likelihood_.eval()
        self._fitted = True
        return self

    @staticmethod
    def _check_bounds(bounds):
        if bounds is None:
            return [(0.0, None)] * 3
        if len(bounds) != 3:
            raise ValueError("parameter_bounds must contain three (lower, upper) pairs")
        result = []
        for pair in bounds:
            if len(pair) != 2:
                raise ValueError("each parameter bound must be a (lower, upper) pair")
            lo, hi = pair
            if lo is not None and (not np.isfinite(lo) or lo < 0):
                raise ValueError("lower bounds must be finite and nonnegative, or None")
            if hi is not None and (not np.isfinite(hi) or hi <= 0):
                raise ValueError("upper bounds must be finite and positive, or None")
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError("each lower bound must be smaller than its upper bound")
            result.append((0.0 if lo is None else float(lo), None if hi is None else float(hi)))
        return result

    def _build_model(self):
        def constraint(index, scale=1.0):
            lo, hi = self._parameter_bounds[index]
            # Interval's constructor casts bounds to torch's global default dtype.
            # Set the buffers after .to() to preserve bounds at the model precision.
            if hi is None:
                result = gpytorch.constraints.GreaterThan(0, transform=torch.exp, inv_transform=torch.log)
            else:
                result = gpytorch.constraints.Interval(0, 1)
            result = result.to(self.train_x_)
            result.lower_bound.copy_(self.train_x_.new_tensor(lo * scale))
            if hi is not None:
                result.upper_bound.copy_(self.train_x_.new_tensor(hi * scale))
            if not bool(torch.isfinite(result.lower_bound)) or \
                    (hi is not None and not bool(torch.isfinite(result.upper_bound))) or \
                    not bool(result.lower_bound < result.upper_bound):
                raise ValueError("parameter_bounds must be representable at the model dtype")
            return result

        self._range_factor = 1.0
        if self.covariance_model == "gaussian":
            self._range_factor = 1/sqrt(2)
        elif self.covariance_model in ("exponential", "matern"):
            nu = .5 if self.covariance_model == "exponential" else self.matern_nu
            self._range_factor = sqrt(2*nu)
        options = {
            "active_dims": tuple(range(self.input_dimension_)),
            "lengthscale_constraint": constraint(1, self._range_factor),
        }
        if self.covariance_model == "gaussian":
            base = gpytorch.kernels.RBFKernel(**options)
        elif self.covariance_model in ("exponential", "matern"):
            base = gpytorch.kernels.MaternKernel(nu=nu, **options)
        else:
            base = (SphericalKernel if self.covariance_model == "spherical" else WaveKernel)(**options)
        self.kernel_ = gpytorch.kernels.ScaleKernel(base, outputscale_constraint=constraint(0)).to(self.train_x_)
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint(2)).to(self.train_x_)
        self.model_ = _RegressionGP(self._inputs, self.train_y_, self.likelihood_, self.kernel_,
                                    self.input_dimension_, self.p_)
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)
        self._parameters = [self.kernel_.raw_outputscale, base.raw_lengthscale,
                            self.likelihood_.raw_noise]

    def _set_theta(self, theta):
        theta = _theta(theta)
        self.kernel_.outputscale = self.train_x_.new_tensor(theta[0])
        self.kernel_.base_kernel.lengthscale = self.train_x_.new_tensor(theta[1] * self._range_factor)
        self.likelihood_.noise = self.train_x_.new_tensor(theta[2])

    def _profile(self):
        K = self.kernel_(self._inputs)
        prior = gpytorch.distributions.MultivariateNormal(torch.zeros_like(self.train_y_), K)
        V = self.likelihood_(prior).lazy_covariance_matrix
        X = self.design_matrix_
        solved = V.solve(torch.cat((X, self.train_y_[:, None]), dim=1))
        info_chol = torch.linalg.cholesky(X.mT @ solved[:, :self.p_])
        beta = torch.cholesky_solve((X.mT @ solved[:, self.p_])[:, None], info_chol).squeeze(1)
        fitted_prior = gpytorch.distributions.MultivariateNormal(X @ beta, K)
        ml = self.n_ * self._mll(fitted_prior, self.train_y_)
        reml = ml - torch.log(info_chol.diagonal()).sum() + .5*self.p_*log(2*pi)
        return beta, info_chol, ml, reml

    def _objective(self):
        _, _, ml, reml = self._profile()
        loss = -(ml if self.likelihood_method_ == "ml" else reml)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite spatial linear model likelihood")
        return loss

    def _optimize(self, iterations):
        tolerance_grad = 1e-7 if self.train_x_.dtype == torch.float64 else 1e-4
        tolerance_change = 1e-12 if self.train_x_.dtype == torch.float64 else 1e-7
        max_eval = 25 * iterations + 1
        optimizer = torch.optim.LBFGS(
            self._parameters, lr=1.0, max_iter=iterations, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )
        best_parameters = [p.detach().clone() for p in self._parameters]
        best_loss = float("inf")
        best_gradient = float("inf")
        evaluations = 0

        def closure():
            nonlocal best_parameters, best_loss, best_gradient, evaluations
            evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            loss = self._objective()
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in self._parameters):
                raise FloatingPointError("non-finite spatial linear model gradient")
            theta = self._current_theta()
            if not bool(torch.isfinite(theta).all()) or bool((theta[:2] <= 0).any()) or theta[2] < 0:
                raise FloatingPointError("invalid covariance parameters during optimization")
            self.training_losses_.append(loss.item())
            if loss.item() < best_loss:
                best_parameters = [p.detach().clone() for p in self._parameters]
                best_loss = loss.item()
                best_gradient = max(p.grad.detach().abs().max().item() for p in self._parameters)
            return loss

        try:
            optimizer.step(closure)
            # Evaluate the loss and gradient at the accepted final parameters.
            objective = closure().item()
            gradient_max = max(p.grad.detach().abs().max().item() for p in self._parameters)
            state = optimizer.state[self._parameters[0]]
            if gradient_max <= tolerance_grad:
                success, message = True, "LBFGS reached gradient tolerance"
            elif state["n_iter"] >= iterations:
                success, message = False, "LBFGS reached iteration limit"
            elif state["func_evals"] >= max_eval:
                success, message = False, "LBFGS reached evaluation limit"
            else:
                success, message = True, "LBFGS reached change tolerance"
        except (FloatingPointError, torch.linalg.LinAlgError, NanError, NotPSDError) as error:
            # A failed line-search trial can leave invalid parameters in place.
            with torch.no_grad():
                for parameter, saved in zip(self._parameters, best_parameters):
                    parameter.copy_(saved)
            objective, gradient_max = best_loss, best_gradient
            success = False
            message = f"LBFGS numerical failure ({type(error).__name__}): {error}"
        state = optimizer.state[self._parameters[0]]
        parameters = self._current_theta().cpu().tolist()
        return {"optimizer": "torch.optim.LBFGS", "success": success, "message": message,
                "objective": objective, "iterations": int(state.get("n_iter", 0)),
                "evaluations": evaluations,
                "gradient_max": gradient_max, "parameters": parameters,
                "at_bound": any(np.isclose(v, lo) or (hi is not None and np.isclose(v, hi))
                                for v, (lo, hi) in zip(parameters, self._parameter_bounds))}

    def _current_theta(self):
        return torch.stack((self.kernel_.outputscale.detach().reshape(()),
                            self.kernel_.base_kernel.lengthscale.detach().reshape(()) / self._range_factor,
                            self.likelihood_.noise.detach().reshape(())))

    @property
    def covariance_parameters_(self) -> Tensor:
        """Partial sill, range phi, and nugget, in that order."""
        self._require_fitted()
        return self._current_theta()

    def _prediction_inputs(self, locations, design_matrix):
        self._require_fitted()
        options = {"dtype": self.train_x_.dtype, "device": self.train_x_.device}
        coords = _array(locations, "locations", 2, **options)
        X = _array(design_matrix, "design_matrix", 2, **options)
        if coords.shape[1] != self.input_dimension_ or X.shape != (len(coords), self.p_):
            raise ValueError("prediction coordinates/design must match training column counts and each other's rows")
        return torch.cat((coords, X), dim=1), X

    def _prediction_terms(self, inputs, X):
        K = self.kernel_(self._inputs)
        V = self.likelihood_(gpytorch.distributions.MultivariateNormal(
            torch.zeros_like(self.train_y_), K
        )).lazy_covariance_matrix
        C = self.kernel_(self._inputs, inputs).to_dense()
        solved = V.solve(torch.cat((C, self.design_matrix_), dim=1))
        solved_C = solved[:, :len(X)]
        solved_X = solved[:, len(X):]
        discrepancy = X - solved_C.mT @ self.design_matrix_
        return solved_C, solved_X, discrepancy

    def predict(self, locations: Locations, design_matrix: Locations, *,
                observation_noise: bool = False) -> KrigingPrediction:
        """Universal kriging of the latent process, including coefficient uncertainty.

        observation_noise adds independent future noise, even at training sites.
        Covariance-parameter uncertainty is not included.
        """
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a bool")
        inputs, X = self._prediction_inputs(locations, design_matrix)
        with torch.no_grad(), _exact_settings():
            posterior = self.model_(inputs)
            if observation_noise:
                posterior = self.likelihood_(posterior)
            _, _, discrepancy = self._prediction_terms(inputs, X)
            correction = discrepancy @ self.coef_covariance_ @ discrepancy.mT
            covariance = posterior.covariance_matrix + correction
            covariance = .5 * (covariance + covariance.mT)
            return KrigingPrediction(posterior.mean.detach(), covariance.detach())

    def weights(self, locations: Locations, design_matrix: Locations) -> Tensor:
        """Universal-kriging weights W, with W @ training_design = prediction_design."""
        inputs, X = self._prediction_inputs(locations, design_matrix)
        with torch.no_grad(), _exact_settings():
            solved_C, solved_X, discrepancy = self._prediction_terms(inputs, X)
            weights = solved_C.mT + discrepancy @ self.coef_covariance_ @ solved_X.mT
            return weights.detach()

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit before inference")


__all__ = ["SpatialLinearModel"]
