"""工业 PID 四大特性演示：抗饱和 / 微分先行 / 前馈 / 串级。

四联图输出到 docs/figures/pid_features.png，并打印指标表（README 用）。

运行: python examples/demo_pid_features.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import setup_plot, L  # noqa: E402

from classic_controlling import (  # noqa: E402
    PID, CascadePID, FOPDTPlant, SecondOrderPlant)

DT = 0.01


def run_pid(ctl, plant, r, steps, ff=None, r_func=None):
    """通用闭环仿真，返回 (t, y, u)。r_func 给定时 r 随时间变化。"""
    t = DT * (np.arange(steps) + 1.0)
    ys, us = np.empty(steps), np.empty(steps)
    y = plant.y
    for k in range(steps):
        rk = r_func(t[k]) if r_func else r
        u = ctl.step(rk, y, ff=ff(rk) if ff else 0.0)
        y = plant.step(u)
        ys[k], us[k] = y, u
    return t, ys, us


# ------------------------------------------------------------------ 场景 1
def sim_anti_windup():
    """二阶对象 + 紧限幅 ±1 + 强积分：抗饱和 on/off 的超调对比。"""
    out = {}
    for aw in (True, False):
        plant = SecondOrderPlant(gain=1.0, zeta=0.5, omega_n=3.0, dt=DT)
        ctl = PID(kp=3.0, ti=0.5, td=0.0, dt=DT, u_min=-1.0, u_max=1.0,
                  anti_windup=aw)
        out[aw] = run_pid(ctl, plant, r=1.0, steps=int(8 / DT))
    return out


# ------------------------------------------------------------------ 场景 2
def sim_derivative_kick():
    """设定值阶跃：微分先行 vs 微分作用误差，看控制量 u 的冲击。"""
    out = {}
    for dom in (True, False):
        plant = SecondOrderPlant(gain=1.0, zeta=0.7, omega_n=2.0, dt=DT)
        ctl = PID(kp=1.0, ti=1.0, td=0.5, dt=DT,
                  derivative_on_measurement=dom)
        out[dom] = run_pid(ctl, plant, r=1.0, steps=int(3 / DT))
    return out


# ------------------------------------------------------------------ 场景 3
def sim_feedforward():
    """斜坡参考 r=0.2t：完美模型前馈 (r+T·ṙ)/K vs 纯反馈。"""
    T, K = 1.0, 1.0
    out = {}
    for use_ff in (True, False):
        plant = FOPDTPlant(gain=K, time_constant=T, delay=0.0, dt=DT)
        ctl = PID(kp=0.5, ti=1.0, td=0.0, dt=DT)
        ff = (lambda r: (r + T * 0.2) / K) if use_ff else None
        out[use_ff] = run_pid(ctl, plant, r=None, steps=int(10 / DT),
                              ff=ff, r_func=lambda t: 0.2 * t)
    return out


# ------------------------------------------------------------------ 场景 4
def sim_cascade(t_pre=5.0, t_total=15.0, d=0.3):
    """内环(快,T=0.2)×外环(慢,T=1.0)串联对象，t=5s 内环输入加扰动 d。
    串级(内环快就地吸收) vs 同参数单环。"""
    steps = int(t_total / DT)
    t = DT * (np.arange(steps) + 1.0)
    out = {}
    for casc in (True, False):
        inner = FOPDTPlant(gain=1.0, time_constant=0.2, dt=DT)
        outer = FOPDTPlant(gain=1.0, time_constant=1.0, dt=DT)
        if casc:
            ctl = CascadePID(outer=PID(kp=1.0, ti=1.0, td=0.0, dt=DT),
                             inner=PID(kp=5.0, ti=0.2, td=0.0, dt=DT))
        else:
            ctl = PID(kp=1.0, ti=1.0, td=0.0, dt=DT)
        ys = np.empty(steps)
        y = outer.y
        for k in range(steps):
            dist = d if t[k] >= t_pre else 0.0
            u = ctl.step(1.0, outer.y, inner.y) if casc else ctl.step(1.0, y)
            v = inner.step(u + dist)
            y = outer.step(v)
            ys[k] = y
        out[casc] = ys
    return t, out


def main():
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # 1) 抗饱和
    r1 = sim_anti_windup()
    ax = axes[0, 0]
    for aw, style, name in ((False, "--", "OFF"), (True, "-", "ON")):
        t, y, _ = r1[aw]
        ax.plot(t, y, style, label=L(f"抗饱和 {name}", f"anti-windup {name}"))
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ov_on = (r1[True][1].max() - 1.0) * 100
    ov_off = (r1[False][1].max() - 1.0) * 100
    ax.set_title(L(f"① 抗积分饱和（限幅±1）: 超调 {ov_off:.0f}% → {ov_on:.0f}%",
                   f"Anti-windup (u in ±1): overshoot {ov_off:.0f}% → {ov_on:.0f}%"))
    ax.set_xlabel("t (s)"); ax.set_ylabel("y"); ax.legend()

    # 2) 微分先行
    r2 = sim_derivative_kick()
    ax = axes[0, 1]
    for dom, style, name in ((False, "--", L("微分作用误差", "derivative on error")),
                             (True, "-", L("微分先行", "derivative on measurement"))):
        t, _, u = r2[dom]
        ax.plot(t, u, style, label=name)
    pk_off = np.abs(r2[False][2]).max()
    pk_on = np.abs(r2[True][2]).max()
    ax.set_title(L(f"② 微分先行: 设定值跳变冲击 {pk_off:.1f} → {pk_on:.1f}",
                   f"Derivative kick: u spike {pk_off:.1f} → {pk_on:.1f}"))
    ax.set_xlabel("t (s)"); ax.set_ylabel("u"); ax.set_xlim(0, 1.5); ax.legend()

    # 3) 前馈
    r3 = sim_feedforward()
    ax = axes[1, 0]
    t3 = r3[True][0]
    ax.plot(t3, 0.2 * t3, "k:", lw=1.2, label=L("斜坡参考", "ramp reference"))
    for use_ff, style, name in ((False, "--", L("纯反馈", "feedback only")),
                                (True, "-", L("反馈+前馈", "feedback + feedforward"))):
        t, y, _ = r3[use_ff]
        ax.plot(t, y, style, label=name)
    iae = {k: float(np.sum(np.abs(r3[k][1] - 0.2 * t3)) * DT) for k in r3}
    ax.set_title(L(f"③ 前馈（斜坡跟踪）: IAE {iae[False]:.2f} → {iae[True]:.2f}",
                   f"Feedforward (ramp): IAE {iae[False]:.2f} → {iae[True]:.2f}"))
    ax.set_xlabel("t (s)"); ax.set_ylabel("y"); ax.legend()

    # 4) 串级
    t4, r4 = sim_cascade()
    ax = axes[1, 1]
    for casc, style, name in ((False, "--", L("单环 PID", "single-loop PID")),
                              (True, "-", L("串级 PID", "cascade PID"))):
        ax.plot(t4, r4[casc], style, label=name)
    ax.axvline(5.0, color="gray", lw=0.8, alpha=0.6)
    ax.text(5.1, 0.55, L("内环扰动 +0.3", "inner disturbance +0.3"), fontsize=9)
    post = t4 >= 5.0
    dev_s = np.abs(r4[False][post] - 1.0).max()
    dev_c = np.abs(r4[True][post] - 1.0).max()
    ax.set_title(L(f"④ 串级（内环扰动）: 最大偏差 {dev_s:.2f} → {dev_c:.2f}",
                   f"Cascade (inner disturbance): max dev {dev_s:.2f} → {dev_c:.2f}"))
    ax.set_xlabel("t (s)"); ax.set_ylabel("y"); ax.legend()

    fig.suptitle(L("工业 PID 四大件：抗饱和 / 微分先行 / 前馈 / 串级",
                   "Industrial PID: anti-windup / D-on-measurement / feedforward / cascade"))
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "docs", "figures", "pid_features.png")
    fig.savefig(out)
    print(f"figure -> {os.path.normpath(out)}")

    print("\n指标（README 用）:")
    print(f"  抗饱和超调: OFF {ov_off:.1f}%  ON {ov_on:.1f}%")
    print(f"  微分冲击峰值 u: 作用误差 {pk_off:.2f}  先行 {pk_on:.2f}")
    print(f"  前馈 IAE(斜坡): 纯反馈 {iae[False]:.3f}  带前馈 {iae[True]:.3f}")
    print(f"  串级内环扰动最大偏差: 单环 {dev_s:.3f}  串级 {dev_c:.3f}")


if __name__ == "__main__":
    main()
