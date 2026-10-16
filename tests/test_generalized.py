"""Distribution identities, independent R references, gradients, and fit diagnostics."""

import json
import warnings
from math import log, pi
from pathlib import Path

import gpytorch
import numpy as np
import pytest
import torch

from geopytorch import (
    GLM_FAMILIES,
    SpatialGeneralizedLinearModel,
    SpatialGLMLikelihood,
    SpatialLinearModel,
)
from geopytorch.regression import _exact_settings

torch.set_num_threads(1)


@pytest.mark.parametrize("family", GLM_FAMILIES)
def test_conditional_mean_variance_and_differentiable_density(family):
    phi = 1.0 if family in ("poisson", "binomial") else 3.5
    likelihood = SpatialGLMLikelihood(family).double()
    likelihood.dispersion = phi
    eta = torch.tensor([-0.8, 0.1, 0.7], dtype=torch.float64, requires_grad=True)
    distribution = likelihood(eta, trials=8)
    mu = eta.sigmoid() if family in ("binomial", "beta") else eta.exp()
    expected_mean = 8 * mu if family == "binomial" else mu
    expected_variance = {
        "poisson": mu,
        "nbinomial": mu + mu.square() / phi,
        "binomial": 8 * mu * (1 - mu),
        "beta": mu * (1 - mu) / (1 + phi),
        "Gamma": mu.square() / phi,
        "inverse.gaussian": mu.square() / phi,
        "tweedie": phi * mu.pow(1.5),
    }[family]
    torch.testing.assert_close(distribution.mean, expected_mean)
    torch.testing.assert_close(distribution.variance, expected_variance)
    y = torch.tensor(
        [0.2, 0.6, 0.8] if family == "beta" else [1.0, 2.0, 3.0], dtype=torch.float64
    )
    assert torch.autograd.gradcheck(
        lambda x: likelihood(x, trials=8).log_prob(y), (eta,)
    )
    assert isinstance(likelihood, gpytorch.likelihoods.Likelihood)


REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/spglm_reference.json").read_text()
)
# Marginal Laplace ML targets, using the data of the REML reference above.
ML_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/spglm_ml_reference.json").read_text()
)
FITS = REFERENCE["fits"] + ML_REFERENCE["fits"]


def test_binomial_curvature_at_zero_logits():
    eta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    likelihood = SpatialGLMLikelihood("binomial").double()
    value = likelihood(eta, trials=8).log_prob(torch.tensor([1.0, 3.0, 6.0])).sum()
    score = torch.autograd.grad(value, eta, create_graph=True)[0]
    curvature = -torch.autograd.grad(score.sum(), eta)[0]
    torch.testing.assert_close(curvature, torch.full_like(eta, 2.0))


def fit_reference(case, **kwargs):
    return SpatialGeneralizedLinearModel(case["family"], case["covariance_model"]).fit(
        REFERENCE["coords"],
        case["y"],
        REFERENCE["design"],
        trials=case["trials"],
        offset=REFERENCE["offset"],
        initial=case["initial"],
        dispersion=case["dispersion"],
        likelihood_method=case["method"],
        **kwargs,
    )


@pytest.mark.parametrize(
    "case",
    FITS,
    ids=lambda c: f"{c['family']}-{c['covariance_model']}-{c['method']}",
)
def test_fixed_parameters_against_reference(case):
    model = fit_reference(case, training_iterations=0)
    np.testing.assert_allclose(model.coef_, case["coefficients"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        model.coef_covariance_, case["vcov"], atol=4e-5, rtol=4e-5
    )
    attr = "log_likelihood_" if case["method"] == "ml" else "restricted_log_likelihood_"
    assert getattr(model, attr).item() == pytest.approx(
        case["log_likelihood"], abs=3e-4
    )
    np.testing.assert_allclose(
        model.fitted_link_, case["fitted_link"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        model.fitted_values_, case["fitted_response"], atol=3e-5, rtol=3e-5
    )
    for scale in ("link", "response"):
        prediction = model.predict(
            REFERENCE["newcoords"],
            REFERENCE["newdesign"],
            type=scale,
            offset=REFERENCE["newoffset"],
            trials=REFERENCE["newtrials"] if case["family"] == "binomial" else None,
        )
        np.testing.assert_allclose(
            prediction.mean, case[f"prediction_{scale}"], atol=4e-5, rtol=4e-5
        )
        np.testing.assert_allclose(
            prediction.variance, case[f"variance_{scale}"], atol=4e-5, rtol=4e-5
        )
        assert torch.linalg.eigvalsh(prediction.covariance_matrix).min() > -1e-9


@pytest.mark.parametrize("family", [f for f in GLM_FAMILIES if f != "tweedie"])
@pytest.mark.parametrize("method", ("ml", "reml"))
def test_implicit_laplace_gradient_matches_finite_differences(family, method):
    case = next(c for c in FITS if c["family"] == family and c["method"] == method)
    model = fit_reference(case, training_iterations=0)
    parameters = (
        model._all_parameters
        if model.likelihood_.has_dispersion
        else model._all_parameters[:3]
    )
    for parameter in parameters:
        parameter.requires_grad_(True)
    with _exact_settings():
        loss = model._objective()
        gradient = torch.autograd.grad(loss, parameters)
        for parameter, analytic in zip(parameters, gradient):
            saved = parameter.detach().clone()
            step = 1e-5
            with torch.no_grad():
                parameter.copy_(saved + step)
            upper = model._objective().item()
            with torch.no_grad():
                parameter.copy_(saved - step)
            lower = model._objective().item()
            with torch.no_grad():
                parameter.copy_(saved)
            assert analytic.item() == pytest.approx(
                (upper - lower) / (2 * step), abs=2e-5, rel=1e-4
            )


@pytest.mark.parametrize("family", [f for f in GLM_FAMILIES if f != "tweedie"])
def test_optimization_and_refitting(family):
    case = next(
        c for c in REFERENCE["fits"] if c["family"] == family and c["method"] == "reml"
    )
    baseline = fit_reference(case, training_iterations=0)
    model = fit_reference(
        case,
        training_iterations=150,
        estimate_dispersion=False,
        parameter_bounds=[(0.05, 4), (0.1, 1.5), (0.05, 2)],
    )
    assert model.converged_
    assert (
        model.restricted_log_likelihood_ >= baseline.restricted_log_likelihood_ - 1e-8
    )
    assert model.mode_diagnostics_["score_max"] <= model.mode_tolerance
    assert model.dispersion_ > 0
    expected = (
        model.weights(REFERENCE["newcoords"], REFERENCE["newdesign"])
        @ model.latent_mode_
    )
    actual = model.predict(REFERENCE["newcoords"], REFERENCE["newdesign"], type="link")
    torch.testing.assert_close(actual.mean, expected)
    model.fit(
        REFERENCE["coords"],
        case["y"],
        REFERENCE["design"],
        trials=case["trials"],
        offset=REFERENCE["offset"],
        initial=case["initial"],
        dispersion=case["dispersion"],
        training_iterations=0,
    )
    torch.testing.assert_close(model.coef_, baseline.coef_)


def test_validation_and_fixed_zero_components():
    case = REFERENCE["fits"][0]
    for family, bad_y in (
        ("poisson", [-1.0] * 18),
        ("nbinomial", [0.2] * 18),
        ("beta", [0.0] * 18),
        ("Gamma", [0.0] * 18),
        ("inverse.gaussian", [-1.0] * 18),
    ):
        with pytest.raises(ValueError):
            SpatialGeneralizedLinearModel(family).fit(
                REFERENCE["coords"], bad_y, REFERENCE["design"]
            )
    with pytest.raises(ValueError, match="trials"):
        SpatialGeneralizedLinearModel("binomial").fit(
            REFERENCE["coords"], [2.0] * 18, REFERENCE["design"]
        )
    with pytest.raises(ValueError, match="mode_tolerance"):
        fit_reference(
            case, training_iterations=0, estimate_dispersion=True, mode_tolerance=0
        )
    for theta in ((0.0, 0.3, 0.2), (0.7, 0.3, 0.0)):
        changed = dict(case, initial=theta)
        model = fit_reference(changed, training_iterations=0)
        assert bool(torch.isfinite(model.coef_).all())
    with pytest.raises(RuntimeError, match="call fit"):
        SpatialGeneralizedLinearModel("poisson").predict([[0, 0]], [[1, 0]])


@pytest.mark.parametrize(
    "case", REFERENCE["estimated"], ids=lambda c: f"{c['family']}-{c['method']}"
)
def test_estimated_parameters_against_reference(case):
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        model = fit_reference(case, fixed=[False, True, True], training_iterations=300)
    if not model.converged_:
        assert case["estimated_dispersion"] >= 1e7
        assert any("did not converge" in str(item.message) for item in diagnostics)
    np.testing.assert_allclose(
        model.covariance_parameters_, case["parameters"], atol=3e-4, rtol=3e-4
    )
    np.testing.assert_allclose(model.coef_, case["coefficients"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(
        model.coef_covariance_, case["vcov"], atol=4e-5, rtol=4e-5
    )
    attr = "log_likelihood_" if case["method"] == "ml" else "restricted_log_likelihood_"
    assert getattr(model, attr).item() == pytest.approx(
        case["log_likelihood"], abs=1e-4
    )
    # Reciprocal dispersion stays identified at the infinite-dispersion limit.
    assert 1 / model.dispersion_.item() == pytest.approx(
        1 / case["estimated_dispersion"], abs=1e-5, rel=1e-4
    )


@pytest.mark.parametrize(
    "case",
    REFERENCE["dispersion_fits"] + ML_REFERENCE["dispersion_fits"],
    ids=lambda c: f"{c['family']}-{c['method']}",
)
def test_dispersion_estimates_against_reference(case):
    model = fit_reference(case, fixed=[True, True, True], training_iterations=150)
    assert model.converged_
    if case.get("limit") == "poisson":
        # The ML size diverges; its criterion approaches the Poisson limit.
        assert 1 / model.dispersion_.item() < 1e-4
    else:
        assert model.dispersion_.item() == pytest.approx(
            case["estimated_dispersion"], rel=3e-4
        )
    np.testing.assert_allclose(model.coef_, case["coefficients"], atol=4e-5, rtol=4e-5)
    attr = "log_likelihood_" if case["method"] == "ml" else "restricted_log_likelihood_"
    assert getattr(model, attr).item() == pytest.approx(
        case["log_likelihood"], abs=1e-4
    )


@pytest.mark.parametrize("family", [f for f in GLM_FAMILIES if f != "tweedie"])
@pytest.mark.parametrize("method", ("ml", "reml"))
def test_single_precision_fixed_fits(family, method):
    case = next(c for c in FITS if c["family"] == family and c["method"] == method)
    model = SpatialGeneralizedLinearModel(family).fit(
        torch.tensor(REFERENCE["coords"], dtype=torch.float32),
        case["y"],
        REFERENCE["design"],
        trials=case["trials"],
        offset=REFERENCE["offset"],
        initial=case["initial"],
        dispersion=case["dispersion"],
        likelihood_method=method,
        training_iterations=0,
    )
    np.testing.assert_allclose(model.coef_, case["coefficients"], atol=5e-4, rtol=5e-4)
    assert model.coef_.dtype == torch.float32


def test_multiple_starts_limits_and_failed_refit():
    case = REFERENCE["fits"][1]
    model = fit_reference(
        case,
        fixed=[False, True, True],
        starts=[(0.4, 0.3, 0.2), (1.0, 0.3, 0.2)],
        training_iterations=100,
    )
    assert len(model.optimizer_results_) == 2
    assert -model.restricted_log_likelihood_.item() == pytest.approx(
        min(result["objective"] for result in model.optimizer_results_)
    )
    with pytest.warns(RuntimeWarning, match="iteration or evaluation limit"):
        limited = fit_reference(case, training_iterations=1)
    assert not limited.converged_
    with pytest.raises(FloatingPointError, match="latent mode did not converge"):
        model.fit(
            REFERENCE["coords"],
            case["y"],
            REFERENCE["design"],
            initial=case["initial"],
            training_iterations=0,
            mode_iterations=1,
        )
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(REFERENCE["newcoords"], REFERENCE["newdesign"])


def test_inverse_gaussian_sampling_moments():
    torch.manual_seed(931)
    likelihood = SpatialGLMLikelihood("inverse.gaussian").double()
    likelihood.dispersion = 4.0
    distribution = likelihood(torch.tensor([0.2], dtype=torch.float64))
    samples = distribution.sample((100000,))
    assert samples.mean().item() == pytest.approx(distribution.mean.item(), rel=0.01)
    assert samples.var().item() == pytest.approx(distribution.variance.item(), rel=0.03)


class _GaussianResponse(SpatialGeneralizedLinearModel):
    """Identity-link Gaussian response, for which both Laplace criteria are exact."""

    response_variance = 0.7

    def _conditional(self, eta):
        return torch.distributions.Normal(eta, self.response_variance**0.5).log_prob(
            self.train_y_
        )


@pytest.mark.parametrize("sill", (0.2, 0.6, 1.8))
def test_criteria_are_exact_for_a_gaussian_response(sill):
    y = next(c["y"] for c in FITS if c["family"] == "Gamma")
    args = (REFERENCE["coords"], y, REFERENCE["design"])
    exact = SpatialLinearModel().fit(
        *args, initial=[sill, 0.3, 0.1 + 0.7], training_iterations=0
    )
    constant = 0.5 * len(y) * log(2 * pi)
    for method in ("ml", "reml"):
        # Gamma only validates the positive responses; the density is Gaussian.
        model = _GaussianResponse("Gamma").fit(
            *args,
            initial=[sill, 0.3, 0.1],
            likelihood_method=method,
            training_iterations=0,
        )
        torch.testing.assert_close(model.coef_, exact.coef_, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(
            model.log_likelihood_, exact.log_likelihood_, rtol=1e-11, atol=1e-10
        )
        torch.testing.assert_close(
            model.restricted_log_likelihood_ + constant,
            exact.restricted_log_likelihood_,
            rtol=1e-11,
            atol=1e-10,
        )
        torch.testing.assert_close(
            model.coef_covariance_, exact.coef_covariance_, rtol=1e-8, atol=1e-10
        )


class _PreciseGaussianResponse(_GaussianResponse):
    response_variance = 0.2


def test_ml_estimate_is_stationary_for_exact_gaussian_ml():
    y = next(c["y"] for c in FITS if c["family"] == "Gamma")
    args = (REFERENCE["coords"], y, REFERENCE["design"])
    model = _PreciseGaussianResponse("Gamma").fit(
        *args,
        initial=[0.6, 0.3, 0.0],
        fixed=[False, True, True],
        likelihood_method="ml",
        estimate_dispersion=False,
    )
    assert model.converged_
    sill = model.covariance_parameters_[0].item()
    assert sill > 0.5

    def exact(value):
        return (
            SpatialLinearModel()
            .fit(
                *args,
                initial=[value, 0.3, 0.2],
                likelihood_method="ml",
                training_iterations=0,
            )
            .log_likelihood_.item()
        )

    assert model.log_likelihood_.item() == pytest.approx(exact(sill), abs=1e-9)
    step = 1e-4 * sill
    slope = (exact(sill + step) - exact(sill - step)) / (2 * step)
    assert slope == pytest.approx(0, abs=1e-5)


BOUNDARY_X = torch.linspace(0, 1, 12, dtype=torch.float64)[:, None]
INTERCEPT = torch.ones(12, 1, dtype=torch.float64)
GROUP = torch.cat((torch.zeros(6), torch.ones(6))).double()
SLOPE = torch.cat((INTERCEPT, torch.linspace(-1, 1, 12)[:, None].double()), dim=1)


@pytest.mark.parametrize(
    "family, y, design",
    [
        ("poisson", torch.zeros(12), INTERCEPT),
        ("nbinomial", torch.zeros(12), INTERCEPT),
        ("tweedie", torch.zeros(12), INTERCEPT),
        ("binomial", torch.zeros(12), INTERCEPT),
        ("binomial", torch.ones(12), INTERCEPT),
        ("binomial", GROUP, torch.cat((INTERCEPT, GROUP[:, None]), dim=1)),
        ("binomial", (SLOPE[:, 1] > 0).double(), SLOPE),
        ("poisson", 3 * (1 - GROUP), torch.cat((INTERCEPT, GROUP[:, None]), dim=1)),
    ],
    ids=[
        "zero-poisson",
        "zero-nbinomial",
        "zero-tweedie",
        "zero-binomial",
        "one-binomial",
        "separated-binomial",
        "separated-slope",
        "zero-poisson-group",
    ],
)
@pytest.mark.parametrize("method", ("ml", "reml"))
def test_responses_without_finite_coefficients_are_rejected(family, y, design, method):
    for tolerance in (1e-4, 1e-7, 1e-10):
        with pytest.raises(ValueError, match="no finite coefficient estimate"):
            SpatialGeneralizedLinearModel(family).fit(
                BOUNDARY_X,
                y,
                design,
                initial=(0.5, 0.3, 0.1),
                likelihood_method=method,
                training_iterations=0,
                mode_tolerance=tolerance,
            )


@pytest.mark.parametrize("method", ("ml", "reml"))
def test_boundary_responses_with_finite_coefficients_still_fit(method):
    overlap = GROUP.clone()
    overlap[[0, 6]] = 1 - overlap[[0, 6]]
    group = torch.cat((INTERCEPT, GROUP[:, None]), dim=1)
    counts = torch.tensor([0, 2, 1, 0, 3, 1, 0, 0, 2, 1, 0, 4], dtype=torch.float64)
    for family, y, design in (
        ("binomial", overlap, group),
        ("poisson", counts, group),
        ("tweedie", counts / 2, SLOPE),
    ):
        model = SpatialGeneralizedLinearModel(family).fit(
            BOUNDARY_X,
            y,
            design,
            initial=(0.5, 0.3, 0.1),
            likelihood_method=method,
            training_iterations=0,
        )
        assert model.converged_
        assert bool((model.coef_.abs() < 10).all())


def test_ml_fits_have_no_latent_mode_weights():
    case = next(c for c in ML_REFERENCE["fits"] if c["family"] == "poisson")
    model = fit_reference(case, training_iterations=0)
    with pytest.raises(ValueError, match="REML"):
        model.weights(REFERENCE["newcoords"], REFERENCE["newdesign"])
