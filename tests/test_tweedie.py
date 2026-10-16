"""Tweedie density references, gradients, estimation, and device preservation."""

import json
from math import log, pi
from pathlib import Path

import numpy as np
import pytest
import torch

from geopytorch import COVARIANCE_MODELS, SpatialGeneralizedLinearModel, Tweedie
from geopytorch.regression import _exact_settings


def tensor(x, **options):
    return torch.as_tensor(x, dtype=torch.float64, **options)


@pytest.mark.parametrize("power", [1.01, 1.1, 1.5, 1.9, 1.999])
def test_density_autograd_and_latent_derivative_identities(power):
    y = tensor([0.0, 0.02, 0.5, 2.0, 8.0])
    eta = tensor([-0.4, 0.2, 0.4, 0.7, 1.2]).requires_grad_()
    phi = tensor(0.8).requires_grad_()
    p = tensor(power).requires_grad_()
    density = lambda a, b, c: Tweedie(a.exp(), b, c).log_prob(y)
    assert torch.autograd.gradcheck(density, (eta, phi, p), atol=2e-5, rtol=2e-4)
    # Near power two, a smaller step loses precision when differencing scores.
    torch.manual_seed(731)
    assert torch.autograd.gradgradcheck(
        density, (eta, phi, p), eps=1e-5 if power > 1.99 else 1e-6, atol=3e-5, rtol=3e-4
    )
    score = torch.autograd.grad(density(eta, phi, p).sum(), eta, create_graph=True)[0]
    curvature = -torch.autograd.grad(score.sum(), eta)[0]
    first, second = y * eta.exp().pow(1 - p), eta.exp().pow(2 - p)
    torch.testing.assert_close(score, (first - second) / phi)
    torch.testing.assert_close(curvature, ((p - 1) * first + (2 - p) * second) / phi)


@pytest.mark.parametrize("power", [1.2, 1.5, 1.9])
def test_sample_moments_and_zero_probability(power):
    torch.manual_seed(391)
    distribution = Tweedie(tensor([0.8, 2.0]), tensor(1.4), tensor(power))
    samples = distribution.sample((150000,))
    torch.testing.assert_close(
        samples.mean(0), distribution.mean, rtol=0.015, atol=0.005
    )
    torch.testing.assert_close(
        samples.var(0), distribution.variance, rtol=0.035, atol=0.01
    )
    torch.testing.assert_close(
        (samples == 0).double().mean(0),
        distribution.log_zero_probability.exp(),
        rtol=0.03,
        atol=0.001,
    )
    assert not samples.requires_grad


def test_zero_density_broadcasting_and_tail_failure():
    mu = tensor([[0.2], [2.0]]).requires_grad_()
    distribution = Tweedie(mu, tensor([0.3, 1.0, 3.0]), tensor(1.5))
    zero = distribution.log_prob(torch.zeros(3, 2, 3, dtype=mu.dtype))
    torch.testing.assert_close(zero, distribution.log_zero_probability.expand(3, 2, 3))
    assert bool(torch.isfinite(torch.autograd.grad(zero.sum(), mu)[0]).all())
    with pytest.raises(FloatingPointError, match="tail tolerance"):
        Tweedie(tensor(1.0), tensor(0.001), tensor(1.9), max_series_terms=33).log_prob(
            tensor(1.0)
        )
    for bad in [1.0, 2.0, 0.5, 3.0]:
        with pytest.raises(ValueError, match="power"):
            Tweedie(tensor(1.0), tensor(1.0), tensor(bad))
    with pytest.raises(ValueError):
        Tweedie(tensor(1.0), tensor(-1.0), tensor(1.5))


REFERENCE_PATH = Path(__file__).parent / "fixtures/tweedie_reference.json"


@pytest.fixture
def reference():
    return json.loads(REFERENCE_PATH.read_text())


def fit_reference(reference, **options):
    return SpatialGeneralizedLinearModel(
        "tweedie", options.pop("covariance_model", "exponential")
    ).fit(
        reference["coords"],
        reference["y"],
        reference["design"],
        offset=reference["offset"],
        initial=reference["initial"],
        dispersion=reference["dispersion"],
        power=reference["power"],
        **options,
    )


def test_density_against_r_tweedie(reference):
    cases = reference["density"]
    y, mu, phi, power = [
        tensor([case[key] for case in cases])
        for key in ("y", "mu", "dispersion", "power")
    ]
    expected = tensor([case["density"] for case in cases])
    log_prob = Tweedie(mu, phi, power).log_prob(y)
    nonzero = expected > 0
    torch.testing.assert_close(
        log_prob[nonzero], expected[nonzero].log(), rtol=1e-8, atol=1e-8
    )
    torch.testing.assert_close(
        log_prob, tensor([case["log_density"] for case in cases]), rtol=1e-8, atol=1e-8
    )
    assert bool(torch.isfinite(log_prob).all())
    assert bool((log_prob[~nonzero] < -700).all())


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_profile_gradient(reference, method):
    model = fit_reference(reference, training_iterations=0, likelihood_method=method)
    parameters = [
        model._all_parameters[0],
        model._all_parameters[1],
        model.likelihood_.raw_dispersion,
        model.likelihood_.raw_power,
    ]
    for parameter in parameters:
        parameter.requires_grad_(True)
    with _exact_settings():
        gradient = torch.autograd.grad(model._objective(), parameters)
        for parameter, analytic in zip(parameters, gradient):
            saved = parameter.detach().clone()
            with torch.no_grad():
                parameter.copy_(saved + 1e-5)
            upper = model._objective().item()
            with torch.no_grad():
                parameter.copy_(saved - 1e-5)
            lower = model._objective().item()
            with torch.no_grad():
                parameter.copy_(saved)
            assert analytic.item() == pytest.approx(
                (upper - lower) / 2e-5, abs=1e-5, rel=1e-4
            )


@pytest.mark.parametrize("covariance_model", COVARIANCE_MODELS)
def test_existing_kernels_and_response_uncertainty(reference, covariance_model):
    model = fit_reference(
        dict(reference, initial=[0.3, 0.3, 0.1]),
        training_iterations=0,
        covariance_model=covariance_model,
    )
    assert model.power_ == reference["power"]
    assert not model.power_estimated_
    expected = (
        model.weights(reference["coords"], reference["design"]) @ model.latent_mode_
    )
    link = model.predict(reference["coords"], reference["design"], type="link")
    response = model.predict(reference["coords"], reference["design"])
    observed = model.predict(
        reference["coords"], reference["design"], observation_noise=True
    )
    torch.testing.assert_close(link.mean, expected)
    torch.testing.assert_close(response.mean, link.mean.exp())
    torch.testing.assert_close(
        observed.variance - response.variance,
        model.dispersion_ * response.mean.pow(model.power_),
    )


@pytest.mark.parametrize("estimate_power", [False, True])
def test_estimation_and_refitting(reference, estimate_power):
    baseline = fit_reference(reference, training_iterations=0)
    model = fit_reference(
        reference,
        fixed=[True, True, True],
        estimate_power=estimate_power,
        training_iterations=150,
    )
    assert model.converged_
    assert model.restricted_log_likelihood_ > baseline.restricted_log_likelihood_
    assert model.power_estimated_ == estimate_power
    assert 1 < model.power_ < 2
    if not estimate_power:
        assert model.power_ == reference["power"]
    model.fit(
        reference["coords"],
        reference["y"],
        reference["design"],
        initial=reference["initial"],
        offset=reference["offset"],
        dispersion=reference["dispersion"],
        power=1.2,
        training_iterations=0,
    )
    assert model.power_ == pytest.approx(1.2)
    assert not model.power_estimated_


def test_response_and_power_validation(reference):
    with pytest.raises(ValueError, match="no finite coefficient"):
        SpatialGeneralizedLinearModel("tweedie").fit(
            reference["coords"], [0] * len(reference["y"]), reference["design"]
        )
    with pytest.raises(ValueError, match="nonnegative"):
        SpatialGeneralizedLinearModel("tweedie").fit(
            reference["coords"], [-1] * len(reference["y"]), reference["design"]
        )
    with pytest.raises(ValueError, match="max_series_terms"):
        fit_reference(reference, max_series_terms=10)
    with pytest.raises(ValueError, match="power_bounds"):
        fit_reference(reference, power_bounds=(0.5, 2.1))
    with pytest.raises(ValueError, match="power"):
        changed = dict(reference, power=1.0)
        fit_reference(changed, training_iterations=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_density_and_fit_preserve_dtype(reference, dtype):
    device = "cpu"
    eta = torch.tensor([-0.5, 0.3, 0.8], device=device, dtype=dtype, requires_grad=True)
    phi = torch.tensor(1.2, device=device, dtype=dtype, requires_grad=True)
    power = torch.tensor(1.5, device=device, dtype=dtype, requires_grad=True)
    y = torch.tensor([0.0, 0.3, 2.0], device=device, dtype=dtype)
    density = Tweedie(eta.exp(), phi, power).log_prob(y)
    gradients = torch.autograd.grad(density.sum(), (eta, phi, power))
    assert density.device.type == device
    assert all(
        g.device.type == device and bool(torch.isfinite(g).all()) for g in gradients
    )
    model = SpatialGeneralizedLinearModel("tweedie").fit(
        torch.tensor(reference["coords"], device=device, dtype=dtype),
        reference["y"],
        reference["design"],
        offset=reference["offset"],
        initial=reference["initial"],
        dispersion=reference["dispersion"],
        power=1.5,
        training_iterations=0,
    )
    prediction = model.predict(reference["coords"][:3], reference["design"][:3])
    assert prediction.mean.device.type == device
    assert prediction.covariance_matrix.device.type == device
    assert prediction.mean.dtype == dtype
    assert bool(torch.isfinite(prediction.mean).all())


@pytest.mark.parametrize("method", ["PQL/L", "PQL"])
def test_fixed_fit_against_reference_joint_mode(reference, method):
    model = fit_reference(reference, training_iterations=0)
    case = reference["fixed_fits"][method]
    torch.testing.assert_close(
        model.coef_, tensor(case["coefficients"]), atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        model.fitted_values_, tensor(case["fitted"]), atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        model.coef_covariance_, tensor(case["vcov"]), atol=1e-8, rtol=1e-8
    )


def test_fixed_ml_fit_against_reference_marginal_laplace(reference):
    model = fit_reference(reference, training_iterations=0, likelihood_method="ml")
    case = reference["fixed_fits"]["ML"]
    torch.testing.assert_close(
        model.coef_, tensor(case["coefficients"]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        model.fitted_values_, tensor(case["fitted"]), atol=1e-6, rtol=1e-6
    )
    assert model.log_likelihood_.item() == pytest.approx(
        case["log_likelihood"], abs=1e-9
    )


def test_power_bounds_and_fixed_dispersion(reference):
    bounded = fit_reference(
        reference,
        fixed=[True] * 3,
        estimate_dispersion=False,
        power_bounds=(1.4, 1.7),
        training_iterations=100,
    )
    assert bounded.converged_
    assert 1.4 <= bounded.power_ <= 1.7
    torch.testing.assert_close(bounded.dispersion_, tensor(reference["dispersion"]))
    assert bounded.power_estimated_ and not bounded.dispersion_estimated_
    fixed = fit_reference(
        dict(reference, power=1.05),
        fixed=[True] * 3,
        estimate_power=False,
        training_iterations=100,
    )
    assert fixed.converged_ and not fixed.power_estimated_
    assert fixed.power_ == pytest.approx(1.05)


SPATIAL_PATH = Path(__file__).parent / "fixtures/tweedie_spatial_reference.json"


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_joint_spatial_estimation_against_reference(method):
    reference = json.loads(SPATIAL_PATH.read_text())
    case = reference["fits"][method]
    model = SpatialGeneralizedLinearModel("tweedie").fit(
        reference["coords"],
        reference["y"],
        reference["design"],
        likelihood_method=method,
        starts=[(0.3, 0.2, 0.2), (1.0, 0.5, 0.05)],
        training_iterations=300,
    )
    assert model.converged_ and model.power_estimated_
    # Every start reaches the same stationary interior optimum.
    objectives = [result["objective"] for result in model.optimizer_results_]
    assert max(objectives) - min(objectives) < 1e-6
    assert all(r["gradient_max"] < 1e-4 for r in model.optimizer_results_)
    if method == "ml":
        criterion = model.log_likelihood_.item()
    else:
        # The reference's joint Laplace integral keeps n*log(2*pi)/2.
        n = len(reference["y"])
        criterion = model.restricted_log_likelihood_.item() + 0.5 * n * log(2 * pi)
    assert criterion == pytest.approx(case["log_likelihood"], abs=1e-6)
    np.testing.assert_allclose(
        model.covariance_parameters_, case["parameters"], rtol=2e-3
    )
    assert model.dispersion_.item() == pytest.approx(case["dispersion"], rel=2e-3)
    assert model.power_.item() == pytest.approx(case["power"], abs=2e-4)
    np.testing.assert_allclose(model.coef_, case["coefficients"], atol=1e-4)
