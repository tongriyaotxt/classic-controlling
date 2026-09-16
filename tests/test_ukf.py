"""向量化 UKF 测试：CV 跟踪、非线性兼容、与逐点参考实现对拍。"""

import numpy as np
import pytest

from classic_controlling import UKF


# ---------------------------------------------------------------- models
def cv_fx(sigmas, dt):
    """2D 匀速模型，状态 [x, y, vx, vy]，批量输入 (M, 4)。"""
    out = sigmas.copy()
    out[:, 0] += dt * sigmas[:, 2]
    out[:, 1] += dt * sigmas[:, 3]
    return out


def pos_hx(sigmas):
    """位置观测 [x, y]。"""
    return sigmas[:, :2]


def make_cv_ukf(dt=0.1, q=0.1, r_std=1.0, alpha=1e-3):
    Q = np.array([[dt**3 / 3, 0, dt**2 / 2, 0],
                  [0, dt**3 / 3, 0, dt**2 / 2],
                  [dt**2 / 2, 0, dt, 0],
                  [0, dt**2 / 2, 0, dt]]) * q
    R = np.eye(2) * r_std**2
    return UKF(4, 2, dt, cv_fx, pos_hx, Q=Q, R=R, alpha=alpha)


# ---------------------------------------------------------------- tests
def test_cv_tracking_rmse_below_measurement_noise():
    rng = np.random.default_rng(42)
    dt, steps, r_std = 0.1, 200, 1.0
    # 真实轨迹：匀速直线
    x_true = np.zeros((steps, 4))
    x_true[0] = [0, 0, 2.0, 1.0]
    for k in range(1, steps):
        x_true[k] = cv_fx(x_true[k - 1:k], dt)[0]
    zs = x_true[:, :2] + rng.normal(0, r_std, (steps, 2))

    ukf = make_cv_ukf(dt, r_std=r_std)
    ukf.x = np.array([zs[0, 0], zs[0, 1], 0.0, 0.0])
    ukf.P = np.eye(4) * 4.0
    ests = ukf.batch_filter(zs)

    tail = slice(steps // 2, None)  # 收敛后统计
    rmse_ukf = np.sqrt(np.mean((ests[tail, :2] - x_true[tail, :2])**2))
    rmse_raw = np.sqrt(np.mean((zs[tail] - x_true[tail, :2])**2))
    assert rmse_ukf < 0.6 * rmse_raw, f"UKF RMSE {rmse_ukf:.3f} vs 观测 {rmse_raw:.3f}"


def test_nonlinear_transition_compatibility():
    """含 sin 的非线性状态转移 + 非线性观测（距离/方位角）。"""

    def fx(sigmas, dt):
        out = sigmas.copy()
        out[:, 0] = sigmas[:, 0] + dt * np.sin(sigmas[:, 1])
        out[:, 1] = sigmas[:, 1] + dt * np.cos(sigmas[:, 0])
        return out

    def hx(sigmas):
        return np.stack([np.hypot(sigmas[:, 0], sigmas[:, 1]),
                         np.arctan2(sigmas[:, 1], sigmas[:, 0])], axis=1)

    ukf = UKF(2, 2, 0.1, fx, hx, Q=np.eye(2) * 0.01, R=np.diag([0.1, 0.01]))
    ukf.x = np.array([1.0, 0.5])
    ukf.P = np.eye(2) * 0.5
    rng = np.random.default_rng(0)
    for _ in range(50):
        ukf.predict()
        z = hx(ukf.x[None, :])[0] + rng.normal(0, [0.3, 0.1])
        ukf.update(z)
    assert np.all(np.isfinite(ukf.x))
    assert np.all(np.linalg.eigvalsh(ukf.P) > 0)  # P 保持正定


# -------------------------------------------- 逐点循环参考实现（对拍用）
def loop_sigma_points(x, P, c):
    n = len(x)
    S = np.linalg.cholesky(P) * c
    pts = [x]
    for i in range(n):
        pts.append(x + S[:, i])
    for i in range(n):
        pts.append(x - S[:, i])
    return np.array(pts)


def loop_predict_update(ukf: UKF, z):
    """与 UKF.predict/update 数学等价、逐点循环的参考实现。"""
    n, Wm, Wc, c = ukf.dim_x, ukf._Wm, ukf._Wc, ukf._c
    # predict
    pts = loop_sigma_points(ukf.x, ukf.P, c)
    pf = np.array([ukf.fx(p[None, :], ukf.dt)[0] for p in pts])
    x = np.zeros(n)
    for i in range(2 * n + 1):
        x += Wm[i] * pf[i]
    P = ukf.Q.copy()
    for i in range(2 * n + 1):
        d = pf[i] - x
        P += Wc[i] * np.outer(d, d)
    # update
    pts = loop_sigma_points(x, P, c)
    pz = np.array([ukf.hx(p[None, :])[0] for p in pts])
    zp = np.zeros(ukf.dim_z)
    for i in range(2 * n + 1):
        zp += Wm[i] * pz[i]
    S = ukf.R.copy()
    Pxz = np.zeros((n, ukf.dim_z))
    for i in range(2 * n + 1):
        dz = pz[i] - zp
        dx = pts[i] - x
        S += Wc[i] * np.outer(dz, dz)
        Pxz += Wc[i] * np.outer(dx, dz)
    K = Pxz @ np.linalg.inv(S)
    return x + K @ (z - zp), P - K @ S @ K.T


def test_vectorized_matches_loop_reference():
    dt = 0.1
    rng = np.random.default_rng(7)
    # alpha 取 O(1) 避免 Merwe 权值巨大抵消造成的固有浮点误差
    ukf = make_cv_ukf(dt, alpha=1.0)
    # 每次随机取一组 (x, P) 做单步对拍，避免浮点求和顺序差异逐步放大
    for _ in range(20):
        ukf.x = rng.normal(0, 1, 4)
        L = rng.normal(0, 1, (4, 4))
        ukf.P = L @ L.T + np.eye(4)
        z = pos_hx(ukf.x[None, :])[0] + rng.normal(0, 0.5, 2)
        x_ref, P_ref = loop_predict_update(ukf, z)
        ukf.predict()
        ukf.update(z)
        np.testing.assert_allclose(ukf.x, x_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ukf.P, P_ref, rtol=1e-10, atol=1e-12)
