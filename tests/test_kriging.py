import gpytorch
import pytest
import torch

from geopytorch import OrdinaryKriging, SimpleKriging


def covariance_components(
    input_dimension: int,
) -> tuple[gpytorch.kernels.ScaleKernel, gpytorch.likelihoods.GaussianLikelihood]:
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dimension)
    )
    kernel.initialize(outputscale=1.2)
    kernel.base_kernel.initialize(lengthscale=0.7)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.initialize(noise=1e-3)
    return kernel, likelihood


def test_simple_kriging_predicts_in_2d_and_keeps_known_mean() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    train_y = torch.tensor([2.0, 3.0, 1.0, 2.5], dtype=torch.float64)
    kernel, likelihood = covariance_components(input_dimension=2)
    kriging = SimpleKriging(mean=2.0, kernel=kernel, likelihood=likelihood)

    kriging.fit(train_x, train_y, training_iterations=0)
    prediction = kriging.predict([[0.5, 0.5], [100.0, 100.0]])

    assert prediction.mean.shape == (2,)
    assert prediction.covariance_matrix.shape == (2, 2)
    assert prediction.mean[1].item() == pytest.approx(2.0, abs=1e-8)
    assert kriging.model_.mean_module.constant.item() == pytest.approx(2.0)
    assert not kriging.model_.mean_module.raw_constant.requires_grad


def test_observation_prediction_adds_only_nugget_to_diagonal() -> None:
    train_x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    train_y = torch.tensor([0.0, 1.0, -1.0])
    kernel, likelihood = covariance_components(input_dimension=2)
    kriging = SimpleKriging(mean=0.0, kernel=kernel, likelihood=likelihood)
    kriging.fit(train_x, train_y, training_iterations=0)

    latent = kriging.predict([[0.2, 0.3], [0.6, 0.8]])
    observed = kriging.predict(
        [[0.2, 0.3], [0.6, 0.8]], observation_noise=True
    )
    difference = observed.covariance_matrix - latent.covariance_matrix

    assert difference.diagonal().tolist() == pytest.approx(
        [1e-3, 1e-3], abs=2e-8
    )
    assert difference[0, 1].item() == pytest.approx(0.0, abs=1e-7)


def test_ordinary_weights_sum_to_one_and_predictions_shift_with_data() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    train_y = torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64)
    target_x = torch.tensor([[0.25, 0.4], [0.8, 0.6]], dtype=torch.float64)

    kernel_a, likelihood_a = covariance_components(input_dimension=2)
    model_a = OrdinaryKriging(kernel=kernel_a, likelihood=likelihood_a).fit(
        train_x, train_y, training_iterations=0
    )
    weights = model_a.weights(target_x)
    prediction_a = model_a.predict(target_x)

    kernel_b, likelihood_b = covariance_components(input_dimension=2)
    model_b = OrdinaryKriging(kernel=kernel_b, likelihood=likelihood_b).fit(
        train_x, train_y + 7.0, training_iterations=0
    )
    prediction_b = model_b.predict(target_x)

    assert weights.sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-10)
    assert prediction_b.mean.tolist() == pytest.approx(
        (prediction_a.mean + 7.0).tolist(), abs=1e-10
    )
    torch.testing.assert_close(
        prediction_b.covariance_matrix,
        prediction_a.covariance_matrix,
        rtol=0,
        atol=1e-10,
    )


def test_ordinary_variance_matches_augmented_kriging_system() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    train_y = torch.tensor([0.2, 1.0, -0.3, 0.7], dtype=torch.float64)
    target_x = torch.tensor([[0.4, 0.3]], dtype=torch.float64)
    kernel, likelihood = covariance_components(input_dimension=2)
    model = OrdinaryKriging(kernel=kernel, likelihood=likelihood).fit(
        train_x, train_y, training_iterations=0
    )

    prediction = model.predict(target_x)
    train_covariance = model.kernel_(train_x).to_dense()
    train_covariance.diagonal().add_(model.likelihood_.noise.squeeze())
    cross_covariance = model.kernel_(train_x, target_x).to_dense()
    augmented = torch.cat(
        [
            torch.cat([train_covariance, torch.ones(4, 1, dtype=torch.float64)], dim=1),
            torch.cat(
                [torch.ones(1, 4, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)],
                dim=1,
            ),
        ],
        dim=0,
    )
    solution = torch.linalg.solve(
        augmented,
        torch.cat([cross_covariance, torch.ones(1, 1, dtype=torch.float64)]),
    )
    expected_variance = (
        model.kernel_(target_x).to_dense().squeeze()
        - solution[:-1].mT @ cross_covariance
        - solution[-1]
    ).squeeze()

    torch.testing.assert_close(prediction.variance.squeeze(), expected_variance)


@pytest.mark.parametrize("model_kind", ["simple", "ordinary"])
def test_same_api_accepts_3d_locations(model_kind: str) -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [1.0, 1.0, 0.8],
        ]
    )
    train_y = torch.tensor([0.0, 1.0, -1.0, 0.5])
    kernel, likelihood = covariance_components(input_dimension=3)
    if model_kind == "simple":
        model = SimpleKriging(mean=0.0, kernel=kernel, likelihood=likelihood)
    else:
        model = OrdinaryKriging(kernel=kernel, likelihood=likelihood)

    model.fit(train_x, train_y, training_iterations=0)
    prediction = model.predict([[0.5, 0.5, 0.3]])

    assert model.input_dimension_ == 3
    assert prediction.mean.shape == (1,)
    assert prediction.variance.item() >= 0


def test_optional_gpytorch_training_records_losses() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [0.3, 0.2], [0.8, 0.7], [1.0, 1.0]]
    )
    train_y = torch.tensor([0.0, 0.4, 0.9, 1.1])
    model = OrdinaryKriging(ard=True).fit(
        train_x, train_y, training_iterations=2, learning_rate=0.05
    )

    assert len(model.training_losses_) == 2
    assert all(torch.isfinite(torch.tensor(model.training_losses_)))


def test_ordinary_reml_matches_direct_dense_objective() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [0.3, 0.1], [0.7, 0.8], [1.0, 1.0]],
        dtype=torch.float64,
    )
    train_y = torch.tensor([0.1, 0.5, -0.2, 0.9], dtype=torch.float64)
    kernel, likelihood = covariance_components(input_dimension=2)
    model = OrdinaryKriging(kernel=kernel, likelihood=likelihood).fit(
        train_x,
        train_y,
        training_iterations=0,
        likelihood_method="reml",
    )

    covariance = model.kernel_(train_x).to_dense().clone()
    covariance.diagonal().add_(model.likelihood_.noise.squeeze())
    inverse = torch.linalg.inv(covariance)
    ones = torch.ones((len(train_y), 1), dtype=torch.float64)
    targets = train_y.unsqueeze(-1)
    information = (ones.mT @ inverse @ ones).squeeze()
    mean = (ones.mT @ inverse @ targets) / information
    residual = targets - ones * mean
    expected = -0.5 * (
        (len(train_y) - 1) * torch.log(train_y.new_tensor(2 * torch.pi))
        + torch.logdet(covariance)
        + torch.log(information)
        + (residual.mT @ inverse @ residual).squeeze()
    )

    torch.testing.assert_close(model.restricted_log_likelihood_, expected)
    assert model.model_.mean_module.constant.item() == pytest.approx(
        model.estimated_mean_.item()
    )


def test_simple_kriging_rejects_reml_for_known_mean() -> None:
    with pytest.raises(ValueError, match="unknown fixed effects"):
        SimpleKriging(mean=0.0).fit(
            [[0.0, 0.0], [1.0, 1.0]],
            [0.0, 1.0],
            likelihood_method="reml",
        )


def test_ordinary_reml_can_optimize_with_lbfgs() -> None:
    train_x = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.7], [0.8, 0.6], [1.0, 1.0]],
        dtype=torch.float64,
    )
    train_y = torch.tensor([0.0, 0.3, -0.1, 0.8, 0.5], dtype=torch.float64)
    kernel, likelihood = covariance_components(input_dimension=2)
    model = OrdinaryKriging(kernel=kernel, likelihood=likelihood).fit(
        train_x,
        train_y,
        training_iterations=8,
        learning_rate=0.5,
        likelihood_method="reml",
        optimizer="lbfgs",
    )

    assert len(model.training_losses_) >= 2
    assert all(torch.isfinite(torch.tensor(model.training_losses_)))
    assert model.training_losses_[-1] <= model.training_losses_[0]


def test_shape_validation_is_dimension_aware() -> None:
    model = SimpleKriging(mean=0.0).fit(
        [[0.0, 0.0], [1.0, 1.0]], [0.0, 1.0], training_iterations=0
    )

    with pytest.raises(ValueError, match="2 columns"):
        model.predict([[0.5, 0.5, 0.5]])
