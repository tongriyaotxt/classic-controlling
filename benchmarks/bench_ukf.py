"""UKF 向量化实现 vs 逐点循环实现 的性能对比。

各跑 1000 步 predict+update，维度 dim_x = 2, 8, 32，打印加速比表格。
"""

import time

import numpy as np

from classic_controlling import UKF

STEPS = 1000
DIMS = [2, 8, 32]


def make_problem(dim_x, dt=0.1):
    """构造稳定的随机线性系统与观测矩阵。"""
    rng = np.random.default_rng(0)
    F = np.eye(dim_x) * 0.95 + rng.normal(0, 0.02, (dim_x, dim_x))
    H = rng.normal(0, 1, (dim_x, dim_x))  # 取 dim_z = dim_x

    def fx(sigmas, dt):
        return sigmas @ F.T

    def hx(sigmas):
        return sigmas @ H.T

    return fx, hx, np.eye(dim_x) * 0.01, np.eye(dim_x) * 0.1


# ------------------------------------------------ 逐点循环参考实现
def loop_sigma_points(x, P, c):
    n = len(x)
    S = np.linalg.cholesky(P) * c
    pts = [x]
    for i in range(n):
        pts.append(x + S[:, i])
    for i in range(n):
        pts.append(x - S[:, i])
    return pts


def loop_step(ukf: UKF, z):
    """逐点循环版本的 predict+update（与 UKF 数学等价）。"""
    n, Wm, Wc, c = ukf.dim_x, ukf._Wm, ukf._Wc, ukf._c
    pts = loop_sigma_points(ukf.x, ukf.P, c)
    pf = [ukf.fx(p[None, :], ukf.dt)[0] for p in pts]
    x = np.zeros(n)
    for i in range(2 * n + 1):
        x += Wm[i] * pf[i]
    P = ukf.Q.copy()
    for i in range(2 * n + 1):
        d = pf[i] - x
        P += Wc[i] * np.outer(d, d)
    P = 0.5 * (P + P.T)
    pts = loop_sigma_points(x, P, c)
    pz = [ukf.hx(p[None, :])[0] for p in pts]
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


def bench(dim_x):
    fx, hx, Q, R = make_problem(dim_x)
    rng = np.random.default_rng(1)
    zs = rng.normal(0, 1, (STEPS, dim_x))

    # 向量化实现
    ukf = UKF(dim_x, dim_x, 0.1, fx, hx, Q=Q, R=R)
    t0 = time.perf_counter()
    for z in zs:
        ukf.predict()
        ukf.update(z)
    t_vec = time.perf_counter() - t0

    # 逐点循环参考实现
    x, P = np.zeros(dim_x), np.eye(dim_x)
    t0 = time.perf_counter()
    for z in zs:
        x, P = loop_step(ukf, z)
    t_loop = time.perf_counter() - t0

    return t_vec, t_loop


def main():
    print(f"UKF benchmark: {STEPS} 步 predict+update")
    print(f"{'dim_x':>6} | {'向量化 (s)':>12} | {'逐点循环 (s)':>12} | {'加速比':>8}")
    print("-" * 48)
    for d in DIMS:
        t_vec, t_loop = bench(d)
        print(f"{d:>6} | {t_vec:>12.4f} | {t_loop:>12.4f} | {t_loop / t_vec:>7.1f}x")


if __name__ == "__main__":
    main()
