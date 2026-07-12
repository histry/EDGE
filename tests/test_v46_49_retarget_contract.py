#!/usr/bin/env python3
import numpy as np

from tools.v46_49_gravity_contract import (
    EDGE_DIM,
    GravityThresholds,
    evaluate_gravity_contract,
    gravity_metrics_np,
    identity6d_np,
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
)
from tools.chang_e_edge_retarget import similarity_umeyama


def make_identity_motion(T=60):
    x = np.zeros((T, EDGE_DIM), np.float32)
    x[:, 5] = 1.0
    x[:, 7:151] = identity6d_np((T, 24)).reshape(T, -1)
    return x


def test_rot6d_roundtrip():
    I = np.eye(3, dtype=np.float32)
    r = matrix_to_rot6d_np(I)
    out = rot6d_to_matrix_np(r)
    assert np.allclose(out, I, atol=1e-5)


def test_upright_passes():
    x = make_identity_motion()
    m = gravity_metrics_np(x)
    ok, reasons = evaluate_gravity_contract(m, GravityThresholds())
    assert ok, reasons


def test_sideways_fails():
    x = make_identity_motion()
    a = np.pi / 2
    Rz = np.asarray(
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    x[:, 7:13] = matrix_to_rot6d_np(Rz)
    m = gravity_metrics_np(x)
    ok, _ = evaluate_gravity_contract(m, GravityThresholds())
    assert not ok


def test_similarity():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3))
    a = 0.7
    R = np.asarray(
        [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    )
    s = 0.013
    t = np.asarray([0.2, -0.1, 0.3])
    Y = s * (X @ R.T) + t
    sh, Rh, th = similarity_umeyama(X, Y)
    pred = sh * (X @ Rh.T) + th
    assert np.max(np.abs(pred - Y)) < 1e-6


if __name__ == "__main__":
    test_rot6d_roundtrip()
    test_upright_passes()
    test_sideways_fails()
    test_similarity()
    print("V46.49 contract tests passed")
