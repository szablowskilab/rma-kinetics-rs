import numpy as np
import pytest

from rma_kinetics import models

pytensor = pytest.importorskip("pytensor")
pt = pytest.importorskip("pytensor.tensor")
pymc_ops = pytest.importorskip("rma_kinetics.pymc")


def test_inference_op_predict_and_grad_match_solver():
    obs_time = np.array([1.0, 6.0, 12.0, 12.0, 24.0], dtype=np.float64)
    log_theta = np.log(np.array([0.2, 0.6, 0.007], dtype=np.float64))
    cotangent = np.array([0.25, -0.5, 0.75, 0.5, 1.25], dtype=np.float64)

    solver = models.constitutive.InferenceSolver(obs_time, dt=0.25)
    op = pymc_ops.InferenceOp(solver)

    x = pt.dvector("x")
    mu = op(x)
    predict_fn = pytensor.function([x], mu)

    expected_mu = solver.predict(log_theta)
    np.testing.assert_allclose(predict_fn(log_theta), expected_mu)

    scalar = pt.sum(mu * cotangent)
    grad = pt.grad(scalar, x)
    grad_fn = pytensor.function([x], grad)

    _, expected_grad = solver.predict_and_vjp(log_theta, cotangent)
    np.testing.assert_allclose(grad_fn(log_theta), expected_grad, rtol=1e-6, atol=1e-8)


def test_population_inference_op_predict_and_grad_match_solver():
    mouse_id = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    obs_time = np.array([1.0, 6.0, 1.0, 12.0, 24.0], dtype=np.float64)
    log_prod_mouse = np.log(np.array([0.2, 0.4], dtype=np.float64))
    log_bbb = float(np.log(0.6))
    log_deg = float(np.log(0.007))
    cotangent = np.array([0.25, -0.5, 0.75, 1.25, -0.4], dtype=np.float64)

    solver = models.constitutive.PopulationInferenceSolver(
        mouse_id, obs_time, 2, tf=24.0, dt=0.25
    )
    op = pymc_ops.PopulationInferenceOp(solver)

    prod = pt.dvector("prod")
    bbb = pt.dscalar("bbb")
    deg = pt.dscalar("deg")
    mu = op(prod, bbb, deg)
    predict_fn = pytensor.function([prod, bbb, deg], mu)

    expected_mu = solver.predict(log_prod_mouse, log_bbb, log_deg)
    np.testing.assert_allclose(predict_fn(log_prod_mouse, log_bbb, log_deg), expected_mu)

    scalar = pt.sum(mu * cotangent)
    grad_prod, grad_bbb, grad_deg = pt.grad(scalar, [prod, bbb, deg])
    grad_fn = pytensor.function([prod, bbb, deg], [grad_prod, grad_bbb, grad_deg])

    _, expected_grad_prod, expected_grad_bbb, expected_grad_deg = solver.predict_and_vjp(
        log_prod_mouse, log_bbb, log_deg, cotangent
    )
    actual_grad_prod, actual_grad_bbb, actual_grad_deg = grad_fn(
        log_prod_mouse, log_bbb, log_deg
    )
    np.testing.assert_allclose(actual_grad_prod, expected_grad_prod, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual_grad_bbb, expected_grad_bbb, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual_grad_deg, expected_grad_deg, rtol=1e-6, atol=1e-8)


def test_inference_op_zero_cotangent_gradient_is_zero():
    obs_time = np.array([1.0, 6.0, 12.0, 24.0], dtype=np.float64)
    log_theta = np.log(np.array([0.2, 0.6, 0.007], dtype=np.float64))

    solver = models.constitutive.InferenceSolver(obs_time, dt=0.25)
    op = pymc_ops.InferenceOp(solver)

    x = pt.dvector("x")
    scalar = pt.sum(op(x) * np.zeros_like(obs_time))
    grad_fn = pytensor.function([x], pt.grad(scalar, x))

    np.testing.assert_allclose(grad_fn(log_theta), np.zeros(3), atol=1e-12)
