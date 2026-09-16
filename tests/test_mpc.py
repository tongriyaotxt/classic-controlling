"""LinearMPC 测试：双积分器调节、与 SLSQP 对拍、热启动、约束检查。"""

import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint

from classic_controlling import LinearMPC

# 双积分器（离散，dt=0.1）
A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q = np.eye(2)
R = np.array([[0.1]])
N = 20
X0 = np.array([3.0, 0.0])


def make_mpc(**kw):
    args = dict(A=A, B=B, Q=Q, R=R, horizon=N, u_min=-1.0, u_max=1.0)
    args.update(kw)
    return LinearMPC(**args)


def test_double_integrator_regulation_with_input_bounds():
    mpc = make_mpc()
    x = X0.copy()
    for _ in range(120):
        u = mpc.solve(x)
        assert abs(u[0]) <= 1.0 + 1e-9
        x = A @ x + B @ u
    assert np.linalg.norm(x) < 0.05, f"终端状态范数 {np.linalg.norm(x):.4f}"


def _reference_qp(mpc: LinearMPC, x0):
    """用 SLSQP 解同一个稠密式 QP，返回最优控制序列。"""
    g = mpc._G @ x0
    l = mpc._l0 + mpc._ol @ x0
    u = mpc._u0 + mpc._ou @ x0
    H = 0.5 * (mpc.H + mpc.H.T)

    def cost(z):
        return 0.5 * z @ H @ z + g @ z

    def grad(z):
        return H @ z + g

    con = LinearConstraint(mpc.C, l, u)
    res = minimize(cost, np.zeros_like(g), jac=grad,
                   constraints=[con], method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    assert res.success, res.message
    return res.x


def test_qp_correctness_vs_slsqp():
    mpc = make_mpc()
    x0 = np.array([1.5, -0.5])
    u0 = mpc.solve(x0, warm_start=False)
    z_ref = _reference_qp(mpc, x0)
    np.testing.assert_allclose(u0, z_ref[:mpc.nu], atol=1e-4)


def test_warm_start_reduces_iterations():
    mpc = make_mpc()
    x = X0.copy()
    u = mpc.solve(x, warm_start=False)
    cold_iters = mpc.last_iters
    x = A @ x + B @ u                      # 真实前进一步
    mpc.solve(x, warm_start=True)
    warm_iters = mpc.last_iters
    assert warm_iters < cold_iters, f"热启动 {warm_iters} 应 < 冷启动 {cold_iters}"


def test_state_and_input_constraints_respected():
    x_max = np.array([4.0, 2.0])
    x_min = -x_max
    mpc = make_mpc(x_min=x_min, x_max=x_max)
    x = X0.copy()
    for _ in range(60):
        u = mpc.solve(x)
        assert -1.0 - 1e-6 <= u[0] <= 1.0 + 1e-6
        x = A @ x + B @ u
        assert np.all(x <= x_max + 1e-6) and np.all(x >= x_min - 1e-6)
    assert np.linalg.norm(x) < 0.1
