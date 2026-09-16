"""演示 ADRC.auto_tune 全过程：阶跃辨识 -> FOPDT 拟合 -> 闭环 -> 在线自适应。

四个子图：
1. 开环阶跃实验数据 + 辨识出的 FOPDT 模型响应叠加（标注 K, T, L）；
2. 闭环阶跃跟踪（r=1）：非自适应 vs 自适应，t=40s 对象增益突变 1.0 -> 0.4
   （模拟执行器老化/工况漂移）；
3. 闭环控制量对比；
4. 在线 RLS 对 b0 的修正轨迹：突变后 b0 被拉向新的有效增益 K/T。

说明：在线 RLS 修正是 experimental 特性，auto_tune 默认关闭
（adaptive=False）；本演示显式开启并加快自适应速率（b0_rate=0.05），
用于展示增益突变场景下 RLS 的价值。良态 LTI 对象上非自适应基线
已经足够快，RLS 仅在增益显著漂移时才有收益。

运行::

    python examples/demo_autotune.py

输出 docs/figures/autotune_process.png。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _common import L, setup_plot

from classic_controlling import ADRC, FOPDTPlant, fit_fopdt

DT = 0.05
N_SETTLE = 200            # 阶跃辨识实验长度（10 s）
U_STEP = 1.0
STEPS = 1600              # 闭环 80 s
IDX_SWITCH = STEPS // 2   # t = 40 s 对象增益突变
GAIN_NEW = 0.4            # 突变后增益（有效增益 K/T: 2.0 -> 0.8）
B0_RATE = 0.05            # 演示用自适应速率（快于默认 0.01，便于在演示时域内看到效果）


def run_closed_loop(adaptive):
    """r=1 跟踪，t=40s 增益突变；返回 (ys, us, ctl)。"""
    plant = FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.0, dt=DT)
    probe = ADRC.auto_tune(plant, dt=DT, n_settle=N_SETTLE)
    ctl = ADRC(b0=probe.b0, omega_c=probe.kp ** 0.5, omega_o=probe.beta3 ** (1 / 3),
               dt=DT, adaptive=adaptive, b0_rate=B0_RATE)
    plant.reset()
    y = 0.0
    ys, us = np.empty(STEPS), np.empty(STEPS)
    for k in range(STEPS):
        u = ctl.step(1.0, y)
        if k == IDX_SWITCH:
            plant.true_gain = GAIN_NEW
        y = plant.step(u)
        ys[k], us[k] = y, u
    return ys, us, ctl


def main():
    plt = setup_plot()
    plant = FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.0, dt=DT)

    # ---- 阶段 1：开环阶跃辨识实验（复现 auto_tune 内部的第一步）----
    plant.reset()
    y_id = np.empty(N_SETTLE)
    for k in range(N_SETTLE):
        y_id[k] = plant.step(U_STEP)
    t_id = DT * (np.arange(N_SETTLE) + 1.0)
    K, T, Ld = fit_fopdt(t_id, y_id, U_STEP)

    # FOPDT 模型响应叠加
    tau = np.clip(t_id - Ld, 0.0, None)
    y_fit = y_id[0] + (y_id[-1] - y_id[0]) * (1.0 - np.exp(-tau / T))

    # ---- 阶段 2：闭环跟踪 + 增益突变，自适应 vs 非自适应 ----
    ys_off, us_off, _ = run_closed_loop(adaptive=False)
    ys_on, us_on, ctl = run_closed_loop(adaptive=True)
    t = DT * np.arange(STEPS)
    t_b0 = DT * np.arange(len(ctl.b0_history))
    b0_init = ctl.b0_initial

    def recovery(ys):
        rec = np.nonzero(np.abs(ys[IDX_SWITCH:] - 1.0) > 0.05)[0]
        return (rec[-1] + 1) * DT if rec.size else 0.0

    print(L(f"辨识结果: K={K:.3f} (真值 1.0), T={T:.3f}s (真值 0.5), L={Ld:.3f}s (真值 0)",
            f"Identified: K={K:.3f} (true 1.0), T={T:.3f}s (true 0.5), L={Ld:.3f}s (true 0)"))
    print(L(f"带宽: omega_c={ctl.kp**0.5:.3f} rad/s, omega_o={(ctl.beta3)**(1/3):.3f} rad/s",
            f"Bandwidths: omega_c={ctl.kp**0.5:.3f} rad/s, omega_o={(ctl.beta3)**(1/3):.3f} rad/s"))
    print(L(f"增益突变 (t=40s, K: 1.0->{GAIN_NEW}) 后恢复到 5% 耗时: "
            f"非自适应 {recovery(ys_off):.1f}s, 自适应 {recovery(ys_on):.1f}s",
            f"Recovery to 5% after gain switch (t=40s, K: 1.0->{GAIN_NEW}): "
            f"non-adaptive {recovery(ys_off):.1f}s, adaptive {recovery(ys_on):.1f}s"))
    print(L(f"b0: 初值 {b0_init:.3f} -> RLS 修正后 {ctl.b0:.3f}"
            f"（突变后有效增益 K/T = {GAIN_NEW / 0.5:.2f}）",
            f"b0: initial {b0_init:.3f} -> RLS adapted {ctl.b0:.3f} "
            f"(effective gain after switch K/T = {GAIN_NEW / 0.5:.2f})"))

    # ------------------------------------------------ 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    ax = axes[0, 0]
    ax.plot(t_id, y_id, ".", ms=2.5, color="#1f77b4",
            label=L("阶跃实验数据", "step-test data"))
    ax.plot(t_id, y_fit, "-", color="#d62728",
            label=L(f"FOPDT 拟合: K={K:.2f}, T={T:.2f}s, L={Ld:.2f}s",
                    f"FOPDT fit: K={K:.2f}, T={T:.2f}s, L={Ld:.2f}s"))
    ax.set_title(L("阶段 1：开环阶跃辨识（10 s）", "Stage 1: open-loop step identification (10 s)"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_ylabel(L("输出 y", "output y"))
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[0, 1]
    ax.plot(t, ys_off, "--", color="#d62728",
            label=L("非自适应（auto_tune 默认）", "non-adaptive (auto_tune default)"))
    ax.plot(t, ys_on, "-", color="#1f77b4",
            label=L("自适应（RLS 修正 b0）", "adaptive (RLS on b0)"))
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label=L("设定值", "setpoint"))
    ax.axvline(t[IDX_SWITCH], color="gray", ls="-.", lw=1.0,
               label=L(f"增益突变 K->{GAIN_NEW}", f"gain switch K->{GAIN_NEW}"))
    ax.set_title(L("阶段 2：闭环跟踪 + 增益突变", "Stage 2: tracking + gain switch"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_ylabel(L("输出 y", "output y"))
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1, 0]
    ax.plot(t, us_off, "--", color="#d62728", label=L("非自适应", "non-adaptive"))
    ax.plot(t, us_on, "-", color="#1f77b4", label=L("自适应", "adaptive"))
    ax.axvline(t[IDX_SWITCH], color="gray", ls="-.", lw=1.0)
    ax.set_title(L("闭环控制量", "closed-loop control signal"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_ylabel("u")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1, 1]
    ax.plot(t_b0, ctl.b0_history, "-", color="#9467bd",
            label=L("b0 在线 RLS 轨迹", "b0 online RLS trajectory"))
    ax.axhline(b0_init, color="#d62728", ls="--", lw=1.2,
               label=L(f"辨识初值 K/T = {b0_init:.2f}", f"identified initial K/T = {b0_init:.2f}"))
    ax.axhline(GAIN_NEW / 0.5, color="#2ca02c", ls=":", lw=1.2,
               label=L(f"突变后有效增益 K/T = {GAIN_NEW / 0.5:.2f}",
                       f"effective gain after switch K/T = {GAIN_NEW / 0.5:.2f}"))
    ax.axvline(t[IDX_SWITCH], color="gray", ls="-.", lw=1.0)
    ax.set_title(L("增益突变后 RLS 把 b0 拉向新的有效增益",
                   "RLS pulls b0 toward the new effective gain"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_ylabel("b0")
    ax.legend(loc="center right", fontsize=9)

    fig.suptitle(L("ADRC.auto_tune 全过程：辨识 -> 整定 -> 跟踪 -> 增益突变下的在线自适应",
                   "ADRC.auto_tune end-to-end: identify -> tune -> track -> adapt to gain change"),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures", "autotune_process.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved -> {out}")


if __name__ == "__main__":
    main()
