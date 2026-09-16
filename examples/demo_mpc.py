"""LinearMPC 演示：双积分器带输入约束调节 + 热启动加速。

- 双积分器（dt=0.1），|u| <= 1，从 x0=[3, 0] 调节到原点；
- 子图 1：状态相平面轨迹（位置-速度）；
- 子图 2：控制量曲线，画出 ±1 约束线（可见 bang-bang 式的饱和段）；
- 子图 3：热启动 vs 冷启动的 ADMM 迭代次数对比（每步曲线 + 总迭代柱状图）。

运行::

    python examples/demo_mpc.py

输出 docs/figures/mpc_demo.png。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _common import L, setup_plot

from classic_controlling import LinearMPC

# 双积分器（离散，dt=0.1）
A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q = np.eye(2)
R = np.array([[0.1]])
HORIZON = 40
X0 = np.array([3.0, 0.0])
STEPS = 120


def simulate(warm_start: bool):
    mpc = LinearMPC(A, B, Q, R, horizon=HORIZON, u_min=-1.0, u_max=1.0, eps=1e-5)
    x = X0.copy()
    xs = np.empty((STEPS, 2))
    us = np.empty(STEPS)
    iters = np.empty(STEPS, dtype=int)
    for k in range(STEPS):
        u = mpc.solve(x, warm_start=warm_start)
        iters[k] = mpc.last_iters
        xs[k] = x
        us[k] = u[0]
        x = A @ x + B @ u
    return xs, us, iters


def main():
    plt = setup_plot()
    xs, us, iters_warm = simulate(warm_start=True)
    _, _, iters_cold = simulate(warm_start=False)
    t = 0.1 * np.arange(STEPS)

    print(L(f"终端状态范数: {np.linalg.norm(xs[-1]):.4f}（120 步后）",
            f"final state norm: {np.linalg.norm(xs[-1]):.4f} (after 120 steps)"))
    print(L(f"控制约束 |u|<=1 满足: {np.all(np.abs(us) <= 1 + 1e-9)}",
            f"input bound |u|<=1 satisfied: {np.all(np.abs(us) <= 1 + 1e-9)}"))
    print(L(f"ADMM 总迭代数  冷启动: {iters_cold.sum()}  热启动: {iters_warm.sum()} "
            f"（{iters_cold.sum() / iters_warm.sum():.1f}x 减少）",
            f"total ADMM iters  cold: {iters_cold.sum()}  warm: {iters_warm.sum()} "
            f"({iters_cold.sum() / iters_warm.sum():.1f}x fewer)"))
    print(L(f"平均迭代数/步  冷启动: {iters_cold.mean():.1f}  热启动: {iters_warm.mean():.1f}",
            f"mean iters/step  cold: {iters_cold.mean():.1f}  warm: {iters_warm.mean():.1f}"))

    # ------------------------------------------------ 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 相平面
    ax1.plot(xs[:, 0], xs[:, 1], "-", color="#1f77b4", lw=1.8)
    ax1.plot(*xs[0], "o", color="#d62728", ms=8, label=L("起点 (3, 0)", "start (3, 0)"))
    ax1.plot(0, 0, "*", color="#2ca02c", ms=14, label=L("目标 (0, 0)", "target (0, 0)"))
    ax1.set_xlabel(L("位置 x1", "position x1"))
    ax1.set_ylabel(L("速度 x2", "velocity x2"))
    ax1.set_title(L("状态相平面轨迹", "phase-plane trajectory"))
    ax1.legend(fontsize=9)

    # 控制量
    ax2.plot(t, us, "-", color="#1f77b4", lw=1.8, label="u")
    ax2.axhline(1.0, color="#d62728", ls="--", lw=1.2, label="u_max = 1")
    ax2.axhline(-1.0, color="#d62728", ls="--", lw=1.2, label="u_min = -1")
    ax2.set_xlabel(L("时间 (s)", "time (s)"))
    ax2.set_ylabel("u")
    ax2.set_title(L("控制量（约束饱和段可见）", "control input (saturation visible)"))
    ax2.legend(fontsize=9)

    # 迭代次数：每步曲线（对数坐标）
    ax3.plot(t, iters_cold, "-", color="#d62728", lw=1.2,
             label=L(f"冷启动（共 {iters_cold.sum()} 次）", f"cold start (total {iters_cold.sum()})"))
    ax3.plot(t, iters_warm, "-", color="#1f77b4", lw=1.2,
             label=L(f"热启动（共 {iters_warm.sum()} 次）", f"warm start (total {iters_warm.sum()})"))
    ax3.set_yscale("log")
    ax3.set_xlabel(L("时间 (s)", "time (s)"))
    ax3.set_ylabel(L("ADMM 迭代数/步（对数）", "ADMM iters/step (log)"))
    ax3.set_title(L("每步迭代数", "iterations per step"))
    ax3.legend(fontsize=9)

    # 迭代数柱状图：总迭代 + 平均迭代
    labels = [L("总迭代数", "total iters"), L("平均迭代/步", "mean iters/step")]
    cold_v = [iters_cold.sum(), iters_cold.mean()]
    warm_v = [iters_warm.sum(), iters_warm.mean()]
    xpos = np.arange(2)
    b1 = ax4.bar(xpos - 0.18, cold_v, width=0.36, color="#d62728", label=L("冷启动", "cold start"))
    b2 = ax4.bar(xpos + 0.18, warm_v, width=0.36, color="#1f77b4", label=L("热启动", "warm start"))
    ax4.bar_label(b1, fmt="%.0f", fontsize=9)
    ax4.bar_label(b2, fmt="%.0f", fontsize=9)
    ax4.set_xticks(xpos, labels)
    ax4.set_title(L(f"热启动加速比 {iters_cold.sum() / iters_warm.sum():.1f}x",
                    f"warm-start speedup {iters_cold.sum() / iters_warm.sum():.1f}x"))
    ax4.legend(fontsize=9)

    fig.suptitle(L("LinearMPC：双积分器约束调节（稠密 QP + 内置 ADMM）",
                   "LinearMPC: constrained double-integrator regulation (condensed QP + built-in ADMM)"),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures", "mpc_demo.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved -> {out}")


if __name__ == "__main__":
    main()
