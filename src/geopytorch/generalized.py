"""Dense spatial GLMs with profiled coefficients and Laplace curvature."""

from __future__ import annotations

import warnings
from math import log, pi

import gpytorch
import numpy as np
import torch
from linear_operator.utils.errors import NanError, NotPSDError

from geopytorch.kriging import KrigingPrediction
from geopytorch.likelihoods import SpatialGLMLikelihood
from geopytorch.regression import SpatialLinearModel, _array, _exact_settings


def _vector(value, n, like, name):
    result = (
        torch.as_tensor(value, dtype=like.dtype, device=like.device).detach().clone()
    )
    if result.ndim == 0:
        result = result.expand(n).clone()
    if result.shape != (n,) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite scalar or a vector of length {n}")
    return result


def _glm_theta(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (3,)
        or not np.isfinite(value).all()
        or value[1] <= 0
        or np.any(value[[0, 2]] < 0)
        or value[0] + value[2] <= 0
    ):
        raise ValueError(
            "initial needs nonnegative sill/nugget, positive phi, and positive total variance"
        )
    return value


class SpatialGeneralizedLinearModel(SpatialLinearModel):
    """Fit a non-Gaussian response with a Gaussian spatial linear predictor.

    The six families use log links, except binomial and beta which use logit.
    Covariance models and range conventions match SpatialLinearModel. The
    independent nugget is on the link scale, separate from response dispersion.
    Computation uses all observations, O(n**2) storage and O(n**3) factorizations.

    Coefficients are profiled by GLS on the latent predictor. For covariance
    Sigma, let A=(X' Sigma^-1 X)^-1 and P=Sigma^-1-Sigma^-1 X A X' Sigma^-1.
    The latent mode maximizes log p(y|w+offset) - w' P w/2. With H the negative
    Hessian at this mode, the reported profile Laplace log likelihood is
    log p(y|w+offset) + log N(w-X beta; 0, Sigma) - log|H|/2.
    REML also adds log|A|/2 + p*log(2*pi)/2. This convention retains the
    Gaussian normalizing constant; comparisons must use the same convention.
    """

    def __init__(self, family, covariance_model="exponential", *, matern_nu=1.5):
        super().__init__(covariance_model, matern_nu=matern_nu)
        self.family = SpatialGLMLikelihood(family).family

    def fit(
        self,
        train_x,
        train_y,
        design_matrix,
        *,
        likelihood_method="reml",
        initial=None,
        dispersion=1.0,
        estimate_dispersion=True,
        fixed=None,
        trials=None,
        offset=None,
        training_iterations=200,
        starts=None,
        parameter_bounds=None,
        mode_iterations=100,
        mode_tolerance=None,
    ):
        """Fit covariance and dispersion by LBFGS, with an inner Newton mode solve.

        ``initial`` and each entry of ``starts`` are (partial_sill, phi, nugget).
        ``fixed`` is a triple of booleans, default all False. A fixed sill or
        nugget can be zero if total variance is positive. Zero training iterations
        fixes all covariance parameters and dispersion, but still estimates the
        latent mode and regression coefficients. Dispersion is estimated only
        for the four families which have a free dispersion parameter.

        ``trials`` supplies positive integer binomial totals; responses remain
        success counts. ``offset`` is added on the link scale. Bounds use the
        same physical units and transformations as SpatialLinearModel.
        Failed starts retain their best finite iterate and report failure.
        """
        self._fitted = self.converged_ = False
        for value, name, minimum in (
            (training_iterations, "training_iterations", 0),
            (mode_iterations, "mode_iterations", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not isinstance(estimate_dispersion, bool):
            raise TypeError("estimate_dispersion must be a bool")
        if likelihood_method not in ("ml", "reml"):
            raise ValueError("likelihood_method must be 'ml' or 'reml'")
        dtype = train_x.dtype if isinstance(train_x, torch.Tensor) else torch.float64
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float64
        device = train_x.device if isinstance(train_x, torch.Tensor) else None
        self.train_x_ = _array(train_x, "train_x", 2, dtype=dtype, device=device)
        options = {"dtype": dtype, "device": self.train_x_.device}
        self.train_y_ = _array(train_y, "train_y", 1, **options)
        self.design_matrix_ = _array(design_matrix, "design_matrix", 2, **options)
        self.n_, self.p_ = self.design_matrix_.shape
        if (
            len(self.train_x_) != self.n_
            or len(self.train_y_) != self.n_
            or self.n_ <= self.p_
            or self.p_ == 0
        ):
            raise ValueError("inputs must have matching rows and n > p > 0")
        if int(torch.linalg.matrix_rank(self.design_matrix_)) != self.p_:
            raise ValueError("design_matrix must have full column rank")
        self.input_dimension_ = self.train_x_.shape[1]
        if self.covariance_model in ("spherical", "wave") and self.input_dimension_ > 3:
            raise ValueError("spherical and wave support at most 3 spatial dimensions")
        if self.family != "binomial" and trials is not None:
            raise ValueError("trials is only supported for binomial responses")
        self.trials_ = _vector(
            1 if trials is None else trials, self.n_, self.train_x_, "trials"
        )
        self.offset_ = _vector(
            0 if offset is None else offset, self.n_, self.train_x_, "offset"
        )
        self._validate_response()
        self.mode_iterations = mode_iterations
        self.mode_tolerance = (
            (1e-7 if dtype == torch.float64 else 2e-4)
            if mode_tolerance is None
            else float(mode_tolerance)
        )
        if not np.isfinite(self.mode_tolerance) or self.mode_tolerance <= 0:
            raise ValueError("mode_tolerance must be finite and positive")
        self.likelihood_method_ = likelihood_method
        self._inputs = torch.cat((self.train_x_, self.design_matrix_), dim=1)
        self._parameter_bounds = self._check_bounds(parameter_bounds)
        self._build_covariance()
        self.latent_likelihood_ = self.likelihood_
        self.likelihood_ = SpatialGLMLikelihood(self.family).to(self.train_x_)
        self.likelihood_.dispersion = dispersion
        self.mean_module_ = gpytorch.means.LinearMean(self.p_, bias=False).to(
            self.train_x_
        )
        self.mean_module_.weights.requires_grad_(False)
        if fixed is None:
            fixed = [False] * 3
        if len(fixed) != 3 or any(not isinstance(x, (bool, np.bool_)) for x in fixed):
            raise ValueError("fixed must contain three booleans")
        self.fixed_ = tuple(bool(x) or training_iterations == 0 for x in fixed)
        self.dispersion_estimated_ = bool(
            estimate_dispersion
            and training_iterations
            and self.likelihood_.has_dispersion
        )
        self._all_parameters = self._parameters + [self.likelihood_.raw_dispersion]
        for parameter, free in zip(
            self._all_parameters,
            [not x for x in self.fixed_] + [self.dispersion_estimated_],
        ):
            parameter.requires_grad_(free)
        self._parameters = [p for p in self._all_parameters if p.requires_grad]
        self.covariance_estimated_ = sum(not x for x in self.fixed_)
        if initial is None:
            distances = torch.pdist(self.train_x_)
            distances = distances[distances > 0]
            phi = distances.median().item() / 3 if distances.numel() else 1.0
            initial = (0.5, phi, 0.1)
        initial = _glm_theta(initial)
        if not training_iterations and starts is not None:
            raise ValueError("starts require training_iterations > 0")
        initial_values = (
            [initial] if starts is None else [_glm_theta(x) for x in starts]
        )
        if not initial_values:
            raise ValueError("starts must contain at least one parameter triple")
        for start in initial_values:
            for value, is_fixed, (lo, hi) in zip(
                start, self.fixed_, self._parameter_bounds
            ):
                if value < lo or (hi is not None and value > hi):
                    raise ValueError(
                        "initial parameters must be within parameter_bounds"
                    )
                if not is_fixed and (value <= lo or (hi is not None and value >= hi)):
                    raise ValueError(
                        "optimization starts must be strictly within parameter_bounds"
                    )
            if any(
                flag and value != first
                for flag, value, first in zip(self.fixed_, start, initial)
            ):
                raise ValueError(
                    "starts must retain the initial values of fixed parameters"
                )
        self.training_losses_, self.optimizer_results_ = [], []
        with _exact_settings(), torch.enable_grad():
            for start in initial_values:
                self._set_theta(start)
                self.likelihood_.dispersion = dispersion
                if self._parameters:
                    self.optimizer_results_.append(
                        self._optimize_glm(training_iterations)
                    )
                else:
                    objective = self._objective().item()
                    self.optimizer_results_.append(
                        {
                            "success": True,
                            "message": "covariance and dispersion fixed",
                            "objective": objective,
                            "parameters": self._current_theta().cpu().tolist(),
                            "dispersion": self.likelihood_.dispersion.item(),
                            "iterations": 0,
                            "evaluations": 1,
                            "gradient_max": 0.0,
                            "optimizer": "none",
                        }
                    )
            candidates = [
                r for r in self.optimizer_results_ if np.isfinite(r["objective"])
            ]
            if not candidates:
                raise RuntimeError(
                    "no optimization start produced a finite objective: "
                    + "; ".join(r["message"] for r in self.optimizer_results_)
                )
            best = min(candidates, key=lambda r: r["objective"])
            self._set_theta(best["parameters"])
            self.likelihood_.dispersion = best["dispersion"]
            self.converged_ = best["success"]
            state = self._profile_glm()
            self._store_fit(state)
        self._fitted = True
        if not self.converged_:
            warnings.warn(
                f"spatial generalized linear model did not converge: {best['message']}",
                RuntimeWarning,
            )
        return self

    def _validate_response(self):
        y = self.train_y_
        if self.family in ("poisson", "nbinomial", "binomial"):
            if bool(((y < 0) | (y != y.round())).any()):
                raise ValueError("count responses must be nonnegative integers")
        elif self.family == "beta":
            if bool(((y <= 0) | (y >= 1)).any()):
                raise ValueError(
                    "beta responses must lie strictly between zero and one"
                )
        elif bool((y <= 0).any()):
            raise ValueError("responses must be strictly positive")
        if self.family == "binomial" and bool(
            (
                (self.trials_ <= 0)
                | (self.trials_ != self.trials_.round())
                | (y > self.trials_)
            ).any()
        ):
            raise ValueError(
                "binomial trials must be positive integers at least as large as each response"
            )

    def _set_theta(self, theta):
        theta = _glm_theta(theta)
        self.kernel_.outputscale = self.train_x_.new_tensor(theta[0])
        self.kernel_.base_kernel.lengthscale = self.train_x_.new_tensor(
            theta[1] * self._range_factor
        )
        self.latent_likelihood_.noise = self.train_x_.new_tensor(theta[2])

    def _current_theta(self):
        return torch.stack(
            (
                self.kernel_.outputscale.detach().reshape(()),
                self.kernel_.base_kernel.lengthscale.detach().reshape(())
                / self._range_factor,
                self.latent_likelihood_.noise.detach().reshape(()),
            )
        )

    def _conditional(self, eta):
        return self.likelihood_(eta, trials=self.trials_).log_prob(self.train_y_)

    def _derivatives(self, eta):
        value = self._conditional(eta)
        score = torch.autograd.grad(value.sum(), eta, create_graph=True)[0]
        curvature = -torch.autograd.grad(score.sum(), eta, create_graph=True)[0]
        return value, score, curvature

    def _mode(self, precision):
        precision = precision.detach()
        if self.family in ("binomial", "beta"):
            w = torch.zeros_like(self.train_y_) - self.offset_
        else:
            w = 0.5 * torch.log1p(self.train_y_) - self.offset_
        identity = torch.eye(self.n_, dtype=w.dtype, device=w.device)
        for iteration in range(self.mode_iterations):
            eta = (w + self.offset_).detach().requires_grad_()
            values, d, curvature = self._derivatives(eta)
            score = d.detach() - precision @ w
            score_max = score.abs().max().item()
            if score_max <= self.mode_tolerance:
                self.mode_diagnostics_ = {
                    "iterations": iteration,
                    "score_max": score_max,
                }
                return w
            H = precision + torch.diag(curvature.detach())
            chol, info = torch.linalg.cholesky_ex(H)
            if info.item():
                shift = (-torch.linalg.eigvalsh(H).min()).clamp_min(0) + 1e-4
                chol = torch.linalg.cholesky(H + shift * identity)
            step = torch.cholesky_solve(score[:, None], chol).squeeze(1)
            old = values.detach().sum() - 0.5 * w @ precision @ w
            slope = score @ step
            scale = 1.0
            accepted = False
            for _ in range(40):
                trial = w + scale * step
                with torch.no_grad():
                    try:
                        objective = (
                            self._conditional(trial + self.offset_).sum()
                            - 0.5 * trial @ precision @ trial
                        )
                    except ValueError:
                        objective = w.new_tensor(-float("inf"))
                roundoff = 64 * torch.finfo(w.dtype).eps * (1 + old.abs())
                improves = bool(torch.isfinite(objective)) and bool(
                    objective >= old + 1e-4 * scale * slope - roundoff
                )
                if (
                    not improves
                    and score_max < 1e-3
                    and bool(torch.isfinite(objective))
                ):
                    # Near the root, density differences can lose precision.
                    candidate = (trial + self.offset_).detach().requires_grad_()
                    trial_score = torch.autograd.grad(
                        self._conditional(candidate).sum(), candidate
                    )[0].detach()
                    improves = bool(
                        (trial_score - precision @ trial).abs().max() < score_max
                    )
                if improves:
                    w = trial.detach()
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                raise FloatingPointError("latent mode line search failed")
        raise FloatingPointError(
            f"latent mode did not converge in {self.mode_iterations} iterations (score {score_max:.3g})"
        )

    def _profile_glm(self):
        X = self.design_matrix_
        prior = gpytorch.distributions.MultivariateNormal(
            torch.zeros_like(self.train_y_), self.kernel_(self._inputs)
        )
        prior = self.latent_likelihood_(prior)
        Sigma = prior.lazy_covariance_matrix
        inverse = Sigma.solve(torch.eye(self.n_, dtype=X.dtype, device=X.device))
        inverse = 0.5 * (inverse + inverse.mT)
        inverse_X = inverse @ X
        info_chol = torch.linalg.cholesky(X.mT @ inverse_X)
        A = torch.cholesky_inverse(info_chol)
        B = A @ inverse_X.mT
        P = inverse - inverse_X @ B
        P = 0.5 * (P + P.mT)
        root = self._mode(P).detach().requires_grad_()
        _, score, curvature = self._derivatives(root + self.offset_)
        H = P + torch.diag(curvature)
        # One Newton correction attaches the implicit derivative of the solved mode.
        w = root + torch.linalg.solve(H, score - P @ root)
        values, _, curvature = self._derivatives(w + self.offset_)
        Hchol = torch.linalg.cholesky(P + torch.diag(curvature))
        beta = B @ w
        residual = w - X @ beta
        ml = values.sum() + prior.log_prob(residual) - Hchol.diagonal().log().sum()
        reml = ml - info_chol.diagonal().log().sum() + 0.5 * self.p_ * log(2 * pi)
        return {
            "beta": beta,
            "w": w,
            "A": A,
            "B": B,
            "inverse": inverse,
            "Hinv": torch.cholesky_inverse(Hchol),
            "ml": ml,
            "reml": reml,
        }

    def _objective(self):
        state = self._profile_glm()
        loss = -state[self.likelihood_method_]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite spatial GLM objective")
        return loss

    def _optimize_glm(self, iterations):
        tolerance_grad = 1e-6 if self.train_x_.dtype == torch.float64 else 1e-3
        tolerance_change = 1e-11 if self.train_x_.dtype == torch.float64 else 1e-6
        optimizer = torch.optim.LBFGS(
            self._parameters,
            max_iter=iterations,
            max_eval=25 * iterations + 1,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )
        saved = [p.detach().clone() for p in self._all_parameters]
        best_loss = best_gradient = float("inf")
        evaluations = 0

        def closure():
            nonlocal evaluations, saved, best_loss, best_gradient
            evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            loss = self._objective()
            loss.backward()
            if not all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in self._parameters
            ):
                raise FloatingPointError("non-finite spatial GLM gradient")
            gradient = max(p.grad.abs().max().item() for p in self._parameters)
            self.training_losses_.append(loss.item())
            if loss.item() < best_loss:
                best_loss, best_gradient = loss.item(), gradient
                saved = [p.detach().clone() for p in self._all_parameters]
            return loss

        try:
            optimizer.step(closure)
            value = closure().item()
            gradient = max(p.grad.abs().max().item() for p in self._parameters)
            state = optimizer.state[self._parameters[0]]
            success = gradient <= tolerance_grad or (
                state.get("n_iter", 0) < iterations
                and state.get("func_evals", 0) < 25 * iterations + 1
            )
            message = (
                "LBFGS reached tolerance"
                if success
                else "LBFGS reached iteration or evaluation limit"
            )
        except (
            ValueError,
            FloatingPointError,
            torch.linalg.LinAlgError,
            NanError,
            NotPSDError,
        ) as error:
            with torch.no_grad():
                for parameter, value in zip(self._all_parameters, saved):
                    parameter.copy_(value)
            value, gradient = best_loss, best_gradient
            success, message = False, f"LBFGS numerical failure: {error}"
        state = optimizer.state[self._parameters[0]]
        return {
            "optimizer": "torch.optim.LBFGS",
            "success": success,
            "message": message,
            "objective": value,
            "gradient_max": gradient,
            "iterations": int(state.get("n_iter", 0)),
            "evaluations": evaluations,
            "parameters": self._current_theta().cpu().tolist(),
            "dispersion": self.likelihood_.dispersion.detach().item(),
        }

    def _store_fit(self, state):
        state = {key: value.detach() for key, value in state.items()}
        self._state = state
        self.coef_ = state["beta"]
        self.coef_covariance_ = state["A"] + state["B"] @ state["Hinv"] @ state["B"].mT
        self.latent_mode_ = state["w"]
        self.latent_covariance_ = state["Hinv"]
        self.fitted_link_ = state["w"] + self.offset_
        self.fitted_values_ = self.likelihood_.inverse_link(
            self.fitted_link_, self.trials_
        ).detach()
        self.log_likelihood_, self.restricted_log_likelihood_ = (
            state["ml"],
            state["reml"],
        )
        self.dispersion_ = self.likelihood_.dispersion.detach().clone()
        self.reml_design_constant_ = (
            0.5
            * torch.linalg.slogdet(
                self.design_matrix_.mT @ self.design_matrix_
            ).logabsdet
        )
        with torch.no_grad():
            self.mean_module_.weights.copy_(self.coef_[:, None])

    def predict(
        self,
        locations,
        design_matrix,
        *,
        type="response",
        offset=None,
        trials=None,
        observation_noise=False,
    ):
        """Predict a new linear predictor or conditional response mean.

        New latent predictors include an independent link-scale nugget, even
        at a training location. The link covariance accounts for coefficient
        estimation and latent-mode uncertainty, conditional on fitted covariance
        and dispersion. Response means are inverse-link plug-in predictions;
        their covariance uses the delta method, not integration over the link
        distribution. ``observation_noise=True`` adds conditional response
        variance on the response scale. Binomial response predictions are counts
        for ``trials`` (one by default). Use ``fitted_values_`` for training fits.
        """
        if type not in ("link", "response"):
            raise ValueError("type must be 'link' or 'response'")
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a bool")
        if observation_noise and type == "link":
            raise ValueError("observation_noise is only defined on the response scale")
        inputs, X = self._prediction_inputs(locations, design_matrix)
        off = _vector(0 if offset is None else offset, len(X), X, "offset")
        if trials is not None and self.family != "binomial":
            raise ValueError("trials is only supported for binomial responses")
        totals = _vector(1 if trials is None else trials, len(X), X, "trials")
        if bool(((totals <= 0) | (totals != totals.round())).any()):
            raise ValueError("trials must be positive integers")
        with torch.no_grad(), _exact_settings():
            state = self._state
            C = self.kernel_(self._inputs, inputs).to_dense()
            solved_C = state["inverse"] @ C
            discrepancy = X - solved_C.mT @ self.design_matrix_
            weights = solved_C.mT + discrepancy @ state["B"]
            mean = (
                self.mean_module_(X)
                + solved_C.mT @ (state["w"] - self.design_matrix_ @ self.coef_)
                + off
            )
            covariance = self.kernel_(inputs).to_dense()
            covariance = covariance + self.latent_likelihood_.noise * torch.eye(
                len(X), dtype=X.dtype, device=X.device
            )
            covariance = (
                covariance - C.mT @ solved_C + discrepancy @ state["A"] @ discrepancy.mT
            )
            covariance = covariance + weights @ state["Hinv"] @ weights.mT
            covariance = 0.5 * (covariance + covariance.mT)
            if type == "response":
                derivative = self.likelihood_.link_derivative(mean, totals)
                covariance = derivative[:, None] * covariance * derivative[None, :]
                if observation_noise:
                    covariance = covariance + torch.diag(
                        self.likelihood_(mean, trials=totals).variance
                    )
                mean = self.likelihood_.inverse_link(mean, totals)
            return KrigingPrediction(mean.detach(), covariance.detach())

    def weights(self, locations, design_matrix):
        """Weights mapping the offset-free fitted latent mode to a new predictor."""
        inputs, X = self._prediction_inputs(locations, design_matrix)
        with torch.no_grad(), _exact_settings():
            C = self.kernel_(self._inputs, inputs).to_dense()
            solved = self._state["inverse"] @ C
            return solved.mT + (X - solved.mT @ self.design_matrix_) @ self._state["B"]


__all__ = ["SpatialGeneralizedLinearModel"]
