"""轻量线性 MPC：稠密式（condensed）QP + 内置 ADMM 求解器 + 热启动。

"编译"一次：把有限时域 LQR 型 MPC 转成单个 QP

    min  0.5 z'H z + g(x0)' z
    s.t. l <= C z <= u

其中 z 为全预测控制序列 (N*nu,)。H 与 C 只算一次；
g、l/u 中依赖 x0 的部分每步线性更新。

QP 由内置 ADMM（OSQP 风格）求解：KKT 矩阵预分解一次
（scipy.linalg.lu_factor），每次迭代 = 一次回代 + 一次投影 + 对偶更新。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve


class LinearMPC:
    """线性时不变系统的约束 MPC 调节器（目标为原点）。"""

    def __init__(self, A, B, Q, R, horizon: int,
                 x_min=None, x_max=None, u_min=None, u_max=None,
                 rho: float = 1.0, sigma: float = 1e-6,
                 max_iter: int = 4000, eps: float = 1e-7):
        """
        参数
        ----
        A, B    : 离散系统 x+ = A x + B u
        Q, R    : 状态/输入权重（终端权重取 Q）
        horizon : 预测时域 N
        x_min/x_max, u_min/u_max : 状态与输入约束（标量或向量，可缺省）
        rho, sigma, max_iter, eps : ADMM 参数
        """
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.N = int(horizon)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.rho, self.sigma = float(rho), float(sigma)
        self.max_iter, self.eps = int(max_iter), float(eps)

        N, nx, nu = self.N, self.nx, self.nu

        # ---- 编译阶段：稠密式预测矩阵  X = Phi x0 + Gamma U ----
        Phi = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, N * nu))
        A_pow = np.eye(nx)
        for k in range(N):
            A_pow = A_pow @ self.A                    # A^(k+1)
            Phi[k * nx:(k + 1) * nx] = A_pow
            row = Gamma[k * nx:(k + 1) * nx]
            Bk = np.eye(nx)
            for j in range(k, -1, -1):                # A^(k-j) B
                row[:, j * nu:(j + 1) * nu] = Bk @ self.B
                Bk = Bk @ self.A
        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.kron(np.eye(N), self.R)

        # H = Gamma' Qbar Gamma + Rbar 只算一次；g(x0) = Gamma' Qbar Phi x0
        self.H = Gamma.T @ Qbar @ Gamma + Rbar
        self._G = Gamma.T @ Qbar @ Phi                # g = G @ x0

        # ---- 约束矩阵 C、以及与 x0 无关/有关的界 ----
        rows, lows, ups = [], [], []
        off_l, off_u = [], []          # 界的 x0 依赖部分（矩阵）：l = l0 + Ol @ x0
        ZU = np.zeros((N * nu, nx))
        if u_min is not None or u_max is not None:
            rows.append(np.eye(N * nu))
            lo = np.full(nu, -np.inf) if u_min is None else np.broadcast_to(np.asarray(u_min, float), (nu,))
            hi = np.full(nu, np.inf) if u_max is None else np.broadcast_to(np.asarray(u_max, float), (nu,))
            lows.append(np.tile(lo, N)); ups.append(np.tile(hi, N))
            off_l.append(ZU); off_u.append(ZU)
        if x_min is not None or x_max is not None:
            rows.append(Gamma)
            lo = np.full(nx, -np.inf) if x_min is None else np.broadcast_to(np.asarray(x_min, float), (nx,))
            hi = np.full(nx, np.inf) if x_max is None else np.broadcast_to(np.asarray(x_max, float), (nx,))
            lows.append(np.tile(lo, N)); ups.append(np.tile(hi, N))
            off_l.append(-Phi); off_u.append(-Phi)    # l - Phi x0 <= Gamma U
        if not rows:
            rows = [np.eye(N * nu)]
            lows = [np.full(N * nu, -np.inf)]; ups = [np.full(N * nu, np.inf)]
            off_l = [ZU]; off_u = [ZU]
        self.C = np.vstack(rows)
        self._l0 = np.concatenate(lows); self._u0 = np.concatenate(ups)
        self._ol = np.vstack(off_l); self._ou = np.vstack(off_u)
        # 约束分块（热启动时移 s/y 用）：(起始行, 行数, 每阶段宽度)
        self._c_blocks = []
        ofs = 0
        for r in rows:
            n_rows = r.shape[0]
            self._c_blocks.append((ofs, n_rows, n_rows // self.N))
            ofs += n_rows

        # ---- ADMM KKT 矩阵预分解一次 ----
        KKT = self.H + self.sigma * np.eye(N * nu) + self.rho * self.C.T @ self.C
        self._lu = lu_factor(KKT)

        # 热启动缓存与诊断
        self._z_prev = np.zeros(N * nu)
        self._s_prev = np.zeros(self.C.shape[0])
        self._y_prev = np.zeros(self.C.shape[0])
        self.last_iters = 0

    @staticmethod
    def _shift_blocks(v: np.ndarray, blocks) -> np.ndarray:
        """按分块把序列时移一个阶段（丢头、末尾重复最后一段），供热启动。"""
        out = np.empty_like(v)
        for start, n_rows, step in blocks:
            seg = v[start:start + n_rows]
            out[start:start + n_rows - step] = seg[step:]
            out[start + n_rows - step:start + n_rows] = seg[n_rows - step:]
        return out

    # ------------------------------------------------------------------
    def _admm(self, g, l, u, z0, s0, y0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """OSQP 风格 ADMM：迭代 = 回代 + 投影 + 对偶更新。

        返回 (z, s, y, 迭代数)；s/y 一并返回以供热启动缓存。"""
        z, s, y = z0.copy(), s0.copy(), y0.copy()
        for it in range(1, self.max_iter + 1):
            z_new = lu_solve(self._lu, self.sigma * z - g
                             + self.rho * self.C.T @ (s - y))
            s_new = np.clip(self.C @ z_new + y, l, u)
            y = y + self.C @ z_new - s_new
            # 收敛判据：原始/对偶残差
            r_prim = np.linalg.norm(self.C @ z_new - s_new, np.inf)
            r_dual = self.rho * np.linalg.norm(self.C.T @ (s_new - s), np.inf)
            z, s = z_new, s_new
            if max(r_prim, r_dual) < self.eps:
                break
        return z, s, y, it

    def solve(self, x0, warm_start: bool = True) -> np.ndarray:
        """求解 MPC，返回第一个控制量 u0。

        参数
        ----
        x0         : 当前状态
        warm_start : 热启动复用上一步解（z/s/y 按约束分块时移一个阶段，
                     末尾重复最后一段；对偶变量一并复用才能真实加速）
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        g = self._G @ x0
        l = self._l0 + self._ol @ x0
        u = self._u0 + self._ou @ x0
        if warm_start:
            z0 = np.roll(self._z_prev, -self.nu)
            z0[-self.nu:] = self._z_prev[-self.nu:]
            s0 = self._shift_blocks(self._s_prev, self._c_blocks)
            y0 = self._shift_blocks(self._y_prev, self._c_blocks)
        else:
            z0 = np.zeros_like(g)
            s0 = np.zeros_like(l)
            y0 = np.zeros_like(l)
        z, s, y, self.last_iters = self._admm(g, l, u, z0, s0, y0)
        self._z_prev = z
        self._s_prev = s
        self._y_prev = y
        return z[:self.nu].copy()
