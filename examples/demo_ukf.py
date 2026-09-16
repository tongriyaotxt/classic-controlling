"""UKF 演示：2D 协调转弯（coordinated turn）目标跟踪。

目标以恒定速率转弯（转弯率 omega 已知），状态 [x, y, vx, vy]，
观测为带噪位置。UKF 用精确积分形式的转弯模型（sigma 点批量传播）。
子图 1：真实轨迹 vs 带噪观测 vs UKF 估计；子图 2：位置误差随时间。

运行::

    python examples/demo_ukf.py

输出 docs/figures/ukf_tracking.png。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _common import L, setup_plot

from classic_controlling import UKF

DT = 0.1
STEPS = 300               # 30 s
OMEGA = 0.15              # 转弯率 (rad/s)
SPEED = 1.0               # 速度大小 (m/s)
OBS_NOISE = 0.5           # 位置观测噪声 std (m)


def make_fx(omega: float):
    """协调转弯模型（精确离散积分），支持 (M, 4) 批量 sigma 点。"""
    phi = omega * DT
    sp, cp = np.sin(phi), np.cos(phi)

    def fx(sigmas, dt):
        x, y, vx, vy = sigmas.T
        x_new = x + (vx * sp - vy * (1.0 - cp)) / omega
        y_new = y + (vx * (1.0 - cp) + vy * sp) / omega
        vx_new = vx * cp - vy * sp
        vy_new = vx * sp + vy * cp
        return np.stack([x_new, y_new, vx_new, vy_new], axis=1)

    return fx


def hx(sigmas):
    """位置观测 (M, 4) -> (M, 2)。"""
    return sigmas[:, :2]


def main():
    plt = setup_plot()
    rng = np.random.default_rng(42)

    # ---- 真实轨迹：先直行 5 s，再转弯 ----
    fx_straight = None
    xs_true = np.empty((STEPS, 4))
    x = np.array([0.0, 0.0, SPEED, 0.0])
    fx_turn = make_fx(OMEGA)
    for k in range(STEPS):
        xs_true[k] = x
        if k < 50:  # 直行
            x = x + np.array([x[2] * DT, x[3] * DT, 0.0, 0.0])
        else:       # 协调转弯
            x = fx_turn(x[None, :], DT)[0]

    zs = xs_true[:, :2] + OBS_NOISE * rng.standard_normal((STEPS, 2))

    # ---- UKF（转弯率已知，全程用转弯模型；直行段是其 omega*dt 小的特例）----
    Q = np.diag([0.01, 0.01, 0.05, 0.05])
    R = np.eye(2) * OBS_NOISE**2
    ukf = UKF(4, 2, DT, fx_turn, hx, Q=Q, R=R, alpha=0.3, beta=2.0)
    ukf.x = np.array([zs[0, 0], zs[0, 1], 0.5, 0.5])  # 故意给偏的初值
    ukf.P = np.diag([1.0, 1.0, 1.0, 1.0])

    ests = ukf.batch_filter(zs)
    err = np.linalg.norm(ests[:, :2] - xs_true[:, :2], axis=1)
    rmse = float(np.sqrt(np.mean(err[50:]**2)))  # 跳过初始收敛段
    print(L(f"位置 RMSE（5s 后）: {rmse:.3f} m（观测噪声 std {OBS_NOISE} m）",
            f"position RMSE (after 5s): {rmse:.3f} m (obs noise std {OBS_NOISE} m)"))

    # ------------------------------------------------ 绘图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw=dict(width_ratios=[1.3, 1.0]))

    ax = axes[0]
    ax.plot(xs_true[:, 0], xs_true[:, 1], "-", color="#2ca02c", lw=2.0,
            label=L("真实轨迹", "true trajectory"))
    ax.plot(zs[:, 0], zs[:, 1], ".", color="gray", ms=2.5, alpha=0.5,
            label=L("带噪观测", "noisy measurements"))
    ax.plot(ests[:, 0], ests[:, 1], "--", color="#1f77b4", lw=1.6,
            label=L("UKF 估计", "UKF estimate"))
    ax.plot(*xs_true[0, :2], "o", color="#2ca02c", ms=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(L("协调转弯目标跟踪（ω = 0.15 rad/s）",
                   "coordinated-turn tracking (omega = 0.15 rad/s)"))
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    ax.plot(DT * np.arange(STEPS), err, "-", color="#1f77b4", lw=1.4,
            label=L("UKF 位置误差", "UKF position error"))
    ax.axhline(OBS_NOISE, color="gray", ls="--", lw=1.0,
               label=L(f"观测噪声 std = {OBS_NOISE} m", f"obs noise std = {OBS_NOISE} m"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_ylabel(L("位置误差 (m)", "position error (m)"))
    ax.set_title(L(f"收敛后 RMSE = {rmse:.3f} m", f"RMSE after convergence = {rmse:.3f} m"))
    ax.legend(fontsize=9)

    fig.suptitle(L("向量化 UKF：2D 转弯目标跟踪", "Vectorized UKF: 2D turning-target tracking"),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures", "ukf_tracking.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved -> {out}")


if __name__ == "__main__":
    main()
