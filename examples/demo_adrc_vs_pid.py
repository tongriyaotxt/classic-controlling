"""核心对比演示：模型整定 PID vs ADRC.auto_tune（零人工干预）。

三个被控对象（FOPDT、二阶欠阻尼、非线性）上对比阶跃跟踪与抗负载扰动：

- PID（对照组）：由同一份开环阶跃数据辨识 FOPDT 模型，再用经典
  Ziegler-Nichols 反应曲线公式整定；辨识时滞 L≈0 时 Z-N 不适用，
  按标准做法回退 IMC-PI（实现见 examples/_common.py，不进包）。
- ADRC：``ADRC.auto_tune(plant, dt)`` 一键完成辨识 + 带宽选择
  （默认非自适应；在线 RLS 修正 b0 为 experimental 可选项），全程零人工调参。

仿真中点在对象输入端突加恒定负载扰动，展示两者的抗扰差异。

运行::

    python examples/demo_adrc_vs_pid.py

输出 docs/figures/adrc_vs_pid.png，并在控制台打印指标对比表。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 供直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from _common import L, auto_tune_pid, identify_fopdt, setup_plot, step_metrics

from classic_controlling import ADRC, FOPDTPlant, NonlinearPlant, SecondOrderPlant

DT = 0.05
STEPS = 800                 # 40 s
IDX_DIST = STEPS // 2       # t = 20 s 突加负载扰动

CASES = [
    dict(name="FOPDT",
         title_zh="一阶惯性对象:  0.5·y' + y = u",
         title_en="First-order plant: 0.5·y' + y = u",
         make=lambda: FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.0, dt=DT),
         n_settle=200, dist=0.5, ylim=None),
    dict(name="SecondOrder",
         title_zh="二阶欠阻尼:  K=1, ζ=0.5, ωn=3 rad/s",
         title_en="2nd-order underdamped: K=1, zeta=0.5, wn=3 rad/s",
         make=lambda: SecondOrderPlant(gain=1.0, zeta=0.5, omega_n=3.0, dt=DT),
         n_settle=200, dist=0.3, ylim=None),
    dict(name="Nonlinear",
         title_zh="非线性（开环发散）:  y'' = -y' + u + 0.5y^2",
         title_en="Nonlinear (open-loop unstable): y'' = -y' + u + 0.5y^2",
         make=lambda: NonlinearPlant(gain=1.0, dt=DT),
         n_settle=15, dist=0.2, ylim=(-2.5, 3.5)),
]


def run_loop(ctl_step, plant, dist):
    """通用闭环：r=1 阶跃，t 中点起在输入端加恒定负载扰动 dist。"""
    plant.reset()
    y = 0.0
    ys, us = np.empty(STEPS), np.empty(STEPS)
    for k in range(STEPS):
        u = ctl_step(1.0, y)
        d = dist if k >= IDX_DIST else 0.0
        y = plant.step(u + d)
        ys[k], us[k] = y, u
    return ys, us


def main():
    plt = setup_plot()
    t = DT * np.arange(STEPS)
    results = []

    for case in CASES:
        # ---- 对照组：阶跃辨识 + 模型整定 PID ----
        plant = case["make"]()
        K, T, Ld = identify_fopdt(plant, DT, n_settle=case["n_settle"])
        pid, rule = auto_tune_pid(K, T, Ld, DT)
        y_pid, u_pid = run_loop(pid.step, plant, case["dist"])

        # ---- 实验组：ADRC.auto_tune 零人工干预 ----
        adrc = ADRC.auto_tune(case["make"](), dt=DT, n_settle=case["n_settle"])
        y_adrc, u_adrc = run_loop(adrc.step, case["make"](), case["dist"])

        m_pid = step_metrics(t, y_pid, 1.0, IDX_DIST)
        m_adrc = step_metrics(t, y_adrc, 1.0, IDX_DIST)
        results.append((case, rule, y_pid, u_pid, y_adrc, u_adrc, m_pid, m_adrc))

    # ------------------------------------------------ 控制台指标表
    hdr = (f"{'Plant':<12} {'Controller':<14} {'Overshoot%':>11} "
           f"{'T_settle(s)':>12} {'T_recover(s)':>13} {'IAE':>7}")
    print("\n" + L("阶跃跟踪指标对比（r=1，t=20s 突加恒定负载扰动）",
                   "Step-response metrics (r=1, step load disturbance at t=20s)"))
    print(hdr)
    print("-" * len(hdr))
    for case, rule, *_r, m_pid, m_adrc in results:
        for tag, m in ((f"PID({rule})", m_pid), ("ADRC", m_adrc)):
            tr = f"{m['t_recover']:>13.2f}" if np.isfinite(m["t_recover"]) else f"{'inf':>13}"
            ts = f"{m['t_settle']:>12.2f}" if np.isfinite(m["t_settle"]) else f"{'--':>12}"
            print(f"{case['name']:<12} {tag:<14} {m['overshoot']:>11.1f} "
                  f"{ts} {tr} {m['iae']:>7.2f}")

    # ------------------------------------------------ 绘图
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    c_pid, c_adrc = "#d62728", "#1f77b4"
    for i, (case, rule, y_pid, u_pid, y_adrc, u_adrc, m_pid, m_adrc) in enumerate(results):
        ax, axu = axes[i]
        ax.plot(t, y_pid, "--", color=c_pid,
                label=L(f"PID（{rule} 整定）", f"PID ({rule} tuned)"))
        ax.plot(t, y_adrc, "-", color=c_adrc, label=L("ADRC（auto_tune）", "ADRC (auto_tune)"))
        ax.axhline(1.0, color="k", ls=":", lw=1.0, label=L("设定值", "setpoint"))
        ax.axvline(t[IDX_DIST], color="gray", ls="-.", lw=1.0,
                   label=L("负载扰动", "load disturbance"))
        ax.set_title(case["title_zh"] if L("x", "") == "x" else case["title_en"],
                     fontsize=11)
        ax.set_ylabel(L("输出 y", "output y"))
        if case["ylim"] is not None:
            ax.set_ylim(*case["ylim"])
        ax.annotate(L(f"超调 {m_adrc['overshoot']:.1f}% / Ts {m_adrc['t_settle']:.2f}s (ADRC)",
                      f"OS {m_adrc['overshoot']:.1f}% / Ts {m_adrc['t_settle']:.2f}s (ADRC)"),
                    xy=(0.98, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=9, color=c_adrc)
        ax.annotate(L(f"超调 {m_pid['overshoot']:.1f}% / Ts {m_pid['t_settle']:.2f}s (PID)",
                      f"OS {m_pid['overshoot']:.1f}% / Ts {m_pid['t_settle']:.2f}s (PID)"),
                    xy=(0.98, 0.16), xycoords="axes fraction",
                    ha="right", fontsize=9, color=c_pid)
        if case["ylim"] is not None:
            ax.annotate(L("PID 持续振荡发散（超出坐标范围）",
                          "PID in growing oscillation (off-scale)"),
                        xy=(0.5, 0.92), xycoords="axes fraction",
                        ha="center", fontsize=9, color=c_pid)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        axu.plot(t, u_pid, "--", color=c_pid, label="PID")
        axu.plot(t, u_adrc, "-", color=c_adrc, label="ADRC")
        axu.axvline(t[IDX_DIST], color="gray", ls="-.", lw=1.0)
        axu.set_ylabel(L("控制量 u", "control u"))
        if case["ylim"] is not None:
            axu.set_ylim(-15, 15)
        if i == 0:
            axu.legend(loc="lower right", fontsize=9)
            axu.set_title(L("控制量对比", "control effort"), fontsize=11)
        if i == 2:
            ax.set_xlabel(L("时间 (s)", "time (s)"))
            axu.set_xlabel(L("时间 (s)", "time (s)"))

    fig.suptitle(L("模型整定 PID vs ADRC.auto_tune —— 零人工调参对比",
                   "Model-tuned PID vs ADRC.auto_tune — zero manual tuning"),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures", "adrc_vs_pid.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure saved -> {out}")


if __name__ == "__main__":
    main()
