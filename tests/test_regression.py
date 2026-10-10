"""Independent likelihood, universal-kriging and integration checks."""

import gpytorch
import numpy as np
import pytest
import torch
from linear_operator.utils.errors import NanError, NotPSDError
from torch.utils._python_dispatch import TorchDispatchMode

from geopytorch import OrdinaryKriging, SpatialLinearModel, covariance_kernel
from geopytorch.regression import _exact_settings


@pytest.fixture
def data():
    rng = np.random.default_rng(710)
    coords = rng.uniform(size=(16, 2))
    coords[1] = coords[0]
    X = np.column_stack([np.ones(16), rng.normal(size=16), rng.normal(size=16)])
    y = X @ np.array([1., .8, -.3]) + rng.normal(size=16)
    newcoords = np.vstack([coords[0], rng.uniform(size=(3, 2))])
    newX = np.column_stack([np.ones(4), rng.normal(size=4), rng.normal(size=4)])
    return coords, y, X, newcoords, newX


@pytest.mark.parametrize("kind,nu", [("exponential", .5), ("gaussian", .5),
                                     ("matern", 1.5), ("matern", 2.5),
                                     ("spherical", .5), ("wave", .5)])
@pytest.mark.parametrize("method", ["ml", "reml"])
def test_likelihood_and_universal_kriging_match_independent_algebra(data, kind, nu, method):
    coords, y, X, newcoords, newX = data
    theta = np.array([1.2, .22, .18])

    def kernel(a, b, parameters=None):
        parameters = theta if parameters is None else parameters
        r = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)/parameters[1]
        if kind == "gaussian":
            rho = np.exp(-r*r)
        elif kind == "spherical":
            rho = np.where(r < 1, 1-1.5*r+.5*r**3, 0)
        elif kind == "wave":
            rho = np.sinc(r/np.pi)
        else:
            poly = 1 if kind == "exponential" else 1+r if nu == 1.5 else 1+r+r*r/3
            rho = poly*np.exp(-r)
        return parameters[0]*rho

    V = kernel(coords, coords) + theta[2]*np.eye(len(y))
    C = kernel(coords, newcoords)
    information = X.T @ np.linalg.solve(V, X)
    beta = np.linalg.solve(information, X.T @ np.linalg.solve(V, y))
    residual = y-X@beta
    ml = -.5*(len(y)*np.log(2*np.pi) + np.linalg.slogdet(V)[1] + residual@np.linalg.solve(V, residual))
    reml = ml-.5*np.linalg.slogdet(information)[1]+.5*X.shape[1]*np.log(2*np.pi)
    D = newX-C.T@np.linalg.solve(V, X)
    covariance = kernel(newcoords, newcoords)-C.T@np.linalg.solve(V, C)+D@np.linalg.solve(information, D.T)
    model = SpatialLinearModel(kind, matern_nu=nu).fit(
        coords, y, X, initial=theta, training_iterations=0, likelihood_method=method)
    pred = model.predict(newcoords, newX)
    np.testing.assert_allclose(model.coef_, beta, atol=1e-10)
    np.testing.assert_allclose(model.coef_covariance_, np.linalg.solve(information, np.eye(3)), atol=1e-10)
    assert model.log_likelihood_.item() == pytest.approx(ml, abs=1e-10)
    assert model.restricted_log_likelihood_.item() == pytest.approx(reml, abs=1e-10)
    np.testing.assert_allclose(pred.mean, newX@beta+C.T@np.linalg.solve(V, residual), atol=1e-10)
    np.testing.assert_allclose(pred.covariance_matrix, covariance, atol=1e-10)
    W = model.weights(newcoords, newX).numpy()
    np.testing.assert_allclose(W@X, newX, atol=1e-10)
    np.testing.assert_allclose(W@y, pred.mean, atol=1e-10)
    observed = model.predict(newcoords, newX, observation_noise=True)
    np.testing.assert_allclose(observed.covariance_matrix-pred.covariance_matrix,
                               theta[2]*np.eye(len(newX)), atol=1e-10)
    assert np.linalg.eigvalsh(pred.covariance_matrix).min() > -1e-10
    # Finite differences use the independent covariance formula. At duplicated
    # sites, differencing native pairwise-distance evaluations can amplify
    # cancellation at distance zero for nonsmooth-in-distance kernels.
    def dense_loss(parameters):
        V = kernel(coords, coords, parameters)+parameters[2]*np.eye(len(y))
        A = X.T@np.linalg.solve(V, X)
        beta = np.linalg.solve(A, X.T@np.linalg.solve(V, y))
        r = y-X@beta
        loss = .5*(len(y)*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+r@np.linalg.solve(V,r))
        if method == "reml":
            loss += .5*np.linalg.slogdet(A)[1]-.5*X.shape[1]*np.log(2*np.pi)
        return loss

    with _exact_settings():
        model.model_.zero_grad(set_to_none=True)
        model._objective().backward()
        analytic = np.array([p.grad.item() for p in model._parameters])
    finite = []
    for j in range(3):
        step = np.eye(3)[j]*1e-5
        finite.append((dense_loss(theta*np.exp(step))-dense_loss(theta*np.exp(-step)))/2e-5)
    np.testing.assert_allclose(analytic, finite, atol=3e-6)


def test_intercept_only_reduces_to_ordinary_kriging(data):
    coords, y, _, newcoords, _ = data
    kernel = covariance_kernel("exponential").double()
    kernel.initialize(outputscale=torch.tensor(1.2, dtype=torch.float64))
    kernel.base_kernel.initialize(lengthscale=torch.tensor(.22, dtype=torch.float64))
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.initialize(noise=torch.tensor(.18, dtype=torch.float64))
    ordinary = OrdinaryKriging(kernel=kernel, likelihood=likelihood).fit(
        torch.tensor(coords), torch.tensor(y), training_iterations=0, likelihood_method="reml")
    regression = SpatialLinearModel().fit(coords, y, np.ones((16, 1)),
        initial=(1.2, .22, .18), training_iterations=0)
    actual = regression.predict(newcoords, np.ones((4, 1)))
    expected = ordinary.predict(newcoords)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.covariance_matrix, expected.covariance_matrix)
    torch.testing.assert_close(regression.restricted_log_likelihood_, ordinary.restricted_log_likelihood_)


def test_design_basis_and_coordinate_unit_changes(data):
    coords, y, X, newcoords, newX = data
    transform = np.array([[1., 2., 0.], [0., 3., .1], [0., 0., .5]])
    base = SpatialLinearModel().fit(coords, y, X, initial=(1.2,.22,.18), training_iterations=0)
    changed = SpatialLinearModel().fit(coords*1000, y, X@transform,
        initial=(1.2,220,.18), training_iterations=0)
    torch.testing.assert_close(base.predict(newcoords, newX).mean,
                               changed.predict(newcoords*1000, newX@transform).mean)
    torch.testing.assert_close(base.restricted_log_likelihood_+base.reml_design_constant_,
                               changed.restricted_log_likelihood_+changed.reml_design_constant_)


def test_zero_nugget_and_no_intercept(data):
    coords, y, X, _, _ = data
    model = SpatialLinearModel().fit(coords[2:], y[2:], X[2:, 1:],
                                         initial=(1.2,.22,0), training_iterations=0)
    with pytest.warns(gpytorch.utils.warnings.GPInputWarning):
        pred = model.predict(coords[2:], X[2:, 1:])
    np.testing.assert_allclose(pred.mean, y[2:], atol=1e-9)
    np.testing.assert_allclose(pred.variance, 0, atol=1e-9)


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_multistart_optimization_and_refitting(data, method):
    coords, y, X, newcoords, newX = data
    fixed = SpatialLinearModel().fit(coords, y, X, initial=(1.2,.22,.18), training_iterations=0)
    model = SpatialLinearModel().fit(coords, y, X, likelihood_method=method,
        starts=[(1.2,.22,.18), (.7,.1,.3)],
        parameter_bounds=[(.01,10), (.03,3), (.001,10)])
    assert model.converged_
    assert len(model.optimizer_results_) == 2
    assert all(r["optimizer"] == "torch.optim.LBFGS" for r in model.optimizer_results_)
    attr = "log_likelihood_" if method == "ml" else "restricted_log_likelihood_"
    assert getattr(model, attr) >= getattr(fixed, attr)
    assert -getattr(model, attr).item() == pytest.approx(
        min(r["objective"] for r in model.optimizer_results_), abs=1e-9)
    model.predict(newcoords, newX)  # populate ExactGP prediction cache before refitting
    model.fit(coords, y+2, X, initial=(1.2,.22,.18), training_iterations=0)
    torch.testing.assert_close(model.predict(newcoords, newX).mean,
                               fixed.predict(newcoords, newX).mean+2)


def test_validation_and_float32(data):
    coords, y, X, newcoords, newX = data
    for bad_X in (np.ones((16,2)), X[:3], X*np.nan):
        with pytest.raises(ValueError):
            SpatialLinearModel().fit(coords,y,bad_X,training_iterations=0)
    with pytest.raises(ValueError, match="n > p"):
        SpatialLinearModel().fit(coords[:3],y[:3],X[:3],training_iterations=0)
    with pytest.raises(ValueError, match="strictly positive"):
        SpatialLinearModel().fit(coords,y,X,initial=(1,.2,0))
    with pytest.raises(ValueError, match="starts require"):
        SpatialLinearModel().fit(coords,y,X,training_iterations=0,starts=[(1,.2,.1)])
    with pytest.raises(ValueError, match="within parameter_bounds"):
        SpatialLinearModel().fit(coords,y,X,initial=(1,.2,.1),parameter_bounds=[(.1,.5),(0,2),(0,2)])
    for bounds in ([(1,0)]*3, [(-1,1)]*3, [(0,float("inf"))]*3):
        with pytest.raises(ValueError):
            SpatialLinearModel().fit(coords,y,X,parameter_bounds=bounds)
    with pytest.raises(TypeError, match="integer"):
        SpatialLinearModel().fit(coords,y,X,training_iterations=True)
    model = SpatialLinearModel().fit(torch.tensor(coords,dtype=torch.float32),y,X,
                                          initial=(1.2,.22,.18),training_iterations=0)
    assert model.predict(newcoords,newX).mean.dtype == torch.float32
    with pytest.raises(ValueError, match="column counts"):
        model.predict(newcoords,newX[:,1:])
    with pytest.raises(RuntimeError, match="call fit"):
        SpatialLinearModel().predict(newcoords,newX)


@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_constrained_optimizer_matches_known_boundary_solution(boundary):
    # Distances exceed the spherical support, so V=(sill+nugget)*I.
    # The data force both variance parameters to the chosen boundary.
    coords = np.arange(8.)[:, None]
    X = np.ones((8, 1))
    if boundary == "upper":
        y = np.tile([-3., 3.], 4)
        initial, bounds = (.5, .25, .1), [(None, 1), (None, .5), (None, .2)]
    else:
        y = np.zeros(8)
        initial, bounds = (2., .25, .4), [(1, None), (None, .5), (.2, None)]
    model = SpatialLinearModel("spherical").fit(
        coords, y, X, initial=initial, parameter_bounds=bounds, likelihood_method="ml")
    expected = -.5 * (8*np.log(2*np.pi*1.2) + y@y/1.2)
    assert model.converged_
    assert model.optimizer_results_[0]["at_bound"]
    assert model.log_likelihood_.item() == pytest.approx(expected, abs=2e-6)
    np.testing.assert_allclose(model.covariance_parameters_[[0, 2]], [1, .2], atol=2e-6)
    for value, (lo, hi) in zip(model.covariance_parameters_, bounds):
        assert lo is None or value >= lo
        assert hi is None or value <= hi


@pytest.mark.parametrize("kind", ["gaussian", "matern"])
def test_bound_precision_and_native_range_conversion(data, kind):
    coords, y, X, newcoords, newX = data
    # These endpoints would collapse if first rounded to float32.
    lo, hi = .123456789012345, .123456789012355
    theta = (1.2, (lo+hi)/2, .18)
    bounded = SpatialLinearModel(kind).fit(coords, y, X, initial=theta,
        parameter_bounds=[(None, None), (lo, hi), (0, None)], training_iterations=0)
    unbounded = SpatialLinearModel(kind).fit(coords, y, X, initial=theta, training_iterations=0)
    torch.testing.assert_close(bounded.covariance_parameters_, unbounded.covariance_parameters_, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(bounded.predict(newcoords, newX).mean, unbounded.predict(newcoords, newX).mean)


def test_iteration_limit_reports_failure_at_final_parameters(data):
    coords, y, X, _, _ = data
    with pytest.warns(RuntimeWarning, match="iteration limit"):
        model = SpatialLinearModel().fit(coords, y, X, initial=(1.2, .22, .18), training_iterations=1)
    result = model.optimizer_results_[0]
    assert not model.converged_ and not result["success"]
    assert result["iterations"] == 1
    assert result["evaluations"] == len(model.training_losses_)
    assert result["objective"] == pytest.approx(-model.restricted_log_likelihood_.item(), abs=1e-10)
    with _exact_settings():
        model.model_.zero_grad(set_to_none=True)
        model._objective().backward()
    assert result["gradient_max"] == pytest.approx(max(p.grad.abs().max().item() for p in model._parameters))


def test_bound_endpoints_are_only_valid_for_fixed_parameters(data):
    coords, y, X, _, _ = data
    bounds = [(1, 2), (.1, .3), (0, .5)]
    for initial in ((1, .2, .1), (2, .2, .1), (1.5, .1, .1), (1.5, .3, .1)):
        with pytest.raises(ValueError, match="strictly within"):
            SpatialLinearModel().fit(coords, y, X, initial=initial, parameter_bounds=bounds)
        fixed = SpatialLinearModel().fit(coords, y, X, initial=initial,
            parameter_bounds=bounds, training_iterations=0)
        np.testing.assert_allclose(fixed.covariance_parameters_, initial, atol=1e-15)
    with pytest.raises(ValueError, match="within parameter_bounds"):
        SpatialLinearModel().fit(coords, y, X, initial=(3, .2, .1),
            parameter_bounds=bounds, training_iterations=0)


@pytest.mark.parametrize("failed_start", [0, 1])
@pytest.mark.parametrize("error_type", [FloatingPointError, torch.linalg.LinAlgError,
                                       NanError, NotPSDError])
def test_multistart_survives_a_numerical_failure(data, monkeypatch, failed_start, error_type):
    coords, y, X, newcoords, newX = data
    original_step = torch.optim.LBFGS.step
    calls = 0

    def step(optimizer, closure):
        nonlocal calls
        current = calls
        calls += 1
        if current == failed_start:
            closure()
            with torch.no_grad():
                for parameter in optimizer.param_groups[0]["params"]:
                    parameter.fill_(float("nan"))
            raise error_type("failed line-search trial")
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    model = SpatialLinearModel().fit(coords, y, X, starts=[(1.2, .22, .18)] * 2,
        parameter_bounds=[(.01, 10), (.03, 3), (.001, 10)])
    assert calls == 2
    assert model.converged_
    failed = model.optimizer_results_[failed_start]
    assert not failed["success"] and "numerical failure" in failed["message"]
    assert failed["evaluations"] == 1
    assert model.optimizer_results_[1 - failed_start]["success"]
    assert -model.restricted_log_likelihood_.item() == pytest.approx(
        min(r["objective"] for r in model.optimizer_results_), abs=1e-9)
    torch.testing.assert_close(model.weights(newcoords, newX) @ model.train_y_,
                               model.predict(newcoords, newX).mean)


def test_single_start_restores_finite_parameters_after_failure(data, monkeypatch):
    coords, y, X, newcoords, newX = data

    def step(optimizer, closure):
        closure()
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        closure()

    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    with pytest.warns(RuntimeWarning, match="numerical failure"):
        model = SpatialLinearModel().fit(coords, y, X, initial=(1.2, .22, .18))
    result = model.optimizer_results_[0]
    assert not model.converged_ and not result["success"]
    assert result["evaluations"] == 2
    assert np.isfinite(result["gradient_max"])
    np.testing.assert_allclose(model.covariance_parameters_, [1.2, .22, .18], atol=1e-12)
    assert result["objective"] == pytest.approx(-model.restricted_log_likelihood_.item(), abs=1e-9)
    assert bool(torch.isfinite(model.predict(newcoords, newX).mean).all())


def test_all_starts_without_finite_results_report_failure(data, monkeypatch):
    coords, y, X, newcoords, newX = data
    model = SpatialLinearModel()

    def objective():
        raise NanError("no finite covariance")

    monkeypatch.setattr(model, "_objective", objective)
    with pytest.raises(RuntimeError, match="no optimization start produced a finite objective"):
        model.fit(coords, y, X, starts=[(1.2, .22, .18), (.7, .1, .3)])
    assert not model.converged_
    assert len(model.optimizer_results_) == 2
    for result in model.optimizer_results_:
        assert not result["success"]
        assert result["objective"] == float("inf")
        assert result["evaluations"] == 1
        assert "no finite covariance" in result["message"]
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(newcoords, newX)


def test_unexpected_optimizer_errors_propagate(data, monkeypatch):
    coords, y, X, _, _ = data

    def step(optimizer, closure):
        raise RuntimeError("unexpected optimizer error")

    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    with pytest.raises(RuntimeError, match="unexpected optimizer error"):
        SpatialLinearModel().fit(coords, y, X, initial=(1.2, .22, .18))


@pytest.mark.parametrize("reverse", [False, True])
def test_large_range_start_does_not_lose_a_usable_fit(reverse):
    rng = np.random.default_rng(710)
    coords = rng.uniform(size=(16, 2))
    X = np.column_stack((np.ones(16), rng.normal(size=16)))
    y = X @ [1., .8] + rng.normal(size=16)
    starts = [(1., .2, .1), (1., 1e8, 1e-10)]
    reference = SpatialLinearModel().fit(coords, y, X, initial=starts[0])
    model = SpatialLinearModel().fit(coords, y, X, starts=starts[::-1] if reverse else starts)
    assert len(model.optimizer_results_) == 2
    assert model.restricted_log_likelihood_.item() >= reference.restricted_log_likelihood_.item() - 1e-8
    assert bool(torch.isfinite(model.covariance_parameters_).all())


def test_weights_do_not_allocate_a_prediction_covariance(data):
    coords, y, X, _, _ = data
    rng = np.random.default_rng(23)
    newcoords = rng.uniform(size=(128, 2))
    newX = np.column_stack((np.ones(128), rng.normal(size=(128, 2))))
    model = SpatialLinearModel().fit(coords, y, X, initial=(1.2, .22, .18), training_iterations=0)

    class CheckAllocation(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                assert tuple(result.shape) != (len(newcoords), len(newcoords))
            return result

    with CheckAllocation():
        weights = model.weights(newcoords, newX)
    assert weights.shape == (128, 16)
    np.testing.assert_allclose(weights @ model.design_matrix_, newX, atol=1e-10)
