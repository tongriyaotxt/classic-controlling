"""向量化无迹卡尔曼滤波（UKF，Merwe 比例采样点）。

与 filterpy 等逐点实现不同，本实现的所有 sigma 点操作都用
NumPy 张量广播一次完成：sigma 点生成、批量传播 fx/hx、加权
均值与协方差重建（np.einsum / matmul）均无逐点 Python 循环。

约定
----
- ``fx(sigmas, dt)`` : 状态转移函数，接收形状 ``(2n+1, dim_x)`` 的
  sigma 点矩阵，返回同形状的传播结果（用户函数内部也应向量化）。
- ``hx(sigmas)``     : 观测函数，接收 ``(2n+1, dim_x)``，返回
  ``(2n+1, dim_z)``。
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky


class UKF:
    """向量化无迹卡尔曼滤波器。"""

    def __init__(self, dim_x: int, dim_z: int, dt: float, fx, hx,
                 Q=None, R=None, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        参数
        ----
        dim_x : 状态维数
        dim_z : 观测维数
        dt    : 采样周期（传给 fx）
        fx    : 状态转移函数 fx(sigmas, dt)，支持 (2n+1, dim_x) 批量输入
        hx    : 观测函数 hx(sigmas)，支持 (2n+1, dim_x) 批量输入
        Q     : 过程噪声协方差 (dim_x, dim_x)，缺省为微小对角阵
        R     : 观测噪声协方差 (dim_z, dim_z)，缺省为单位阵
        alpha, beta, kappa : Merwe 比例采样参数
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx

        self.Q = np.eye(dim_x) * 1e-6 if Q is None else np.atleast_2d(Q).astype(float)
        self.R = np.eye(dim_z) if R is None else np.atleast_2d(R).astype(float)

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

        # ---- 预计算 Merwe 权值与 sigma 点生成的固定系数 ----
        n = dim_x
        lam = alpha**2 * (n + kappa) - n
        self._c = np.sqrt(n + lam)          # sigma 点缩放系数
        self._Wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self._Wc = self._Wm.copy()
        self._Wm[0] = lam / (n + lam)
        self._Wc[0] = self._Wm[0] + (1.0 - alpha**2 + beta)
        # 预分配 eye 用于 sigma 点生成
        self._I = np.eye(n)

    # ------------------------------------------------------------------
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """一次生成全部 2n+1 个 sigma 点，形状 (2n+1, dim_x)。"""
        S = cholesky(P) * self._c            # 下三角 Cholesky 因子
        sigmas = np.empty((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = x
        sigmas[1:self.dim_x + 1] = x + S.T   # 广播：每行加一个列向量
        sigmas[self.dim_x + 1:] = x - S.T
        return sigmas

    def _unscented_mean_cov(self, sigmas: np.ndarray, noise: np.ndarray):
        """由 sigma 点矩阵重建加权均值与协方差（全矩阵运算）。"""
        x = self._Wm @ sigmas                # (dim,)
        d = sigmas - x                       # (2n+1, dim)
        P = (d * self._Wc[:, None]).T @ d + noise
        return x, P

    # ------------------------------------------------------------------
    def predict(self):
        """预测步：Cholesky 分解 → 批量传播 → 外积加权重建 x, P。"""
        sigmas = self._sigma_points(self.x, self.P)
        self._sigmas_f = self.fx(sigmas, self.dt)          # (2n+1, dim_x)
        self.x, self.P = self._unscented_mean_cov(self._sigmas_f, self.Q)
        # 数值对称化，避免累积非对称导致 Cholesky 失败
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray):
        """更新步：批量 hx、互协方差矩阵乘法、卡尔曼增益修正。"""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_z = self.hx(sigmas)                         # (2n+1, dim_z)

        z_pred, S = self._unscented_mean_cov(sigmas_z, self.R)
        # 互协方差 Pxz = sum_i Wc[i] * (x_i - x)(z_i - z_pred)'
        dx = sigmas - self.x
        dz = sigmas_z - z_pred
        Pxz = (dx * self._Wc[:, None]).T @ dz              # (dim_x, dim_z)

        K = Pxz @ np.linalg.inv(S)                         # 卡尔曼增益
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    def batch_filter(self, zs) -> np.ndarray:
        """对观测序列批量滤波，返回每步更新后的状态估计 (N, dim_x)。"""
        zs = np.atleast_2d(np.asarray(zs, dtype=float))
        out = np.empty((len(zs), self.dim_x))
        for i, z in enumerate(zs):
            self.predict()
            self.update(z)
            out[i] = self.x
        return out
