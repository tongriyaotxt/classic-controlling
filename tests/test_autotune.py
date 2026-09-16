"""ADRC 自动整定盲测（本功能的质量门槛）。

全程零人工调参：ADRC.auto_tune(plant) 一键完成阶跃辨识 + 带宽选择，
直接闭环跟踪 r=1 阶跃并断言稳态误差。被控对象的真实参数仅用于断言，
不进入整定流程。

回归覆盖（复核发现的两个已修复缺陷，防复发）：
- 问题 1：RLS 回归在一阶 LTI 对象上 mis-specified，b0 曾漂到护栏导致
  自适应反而劣化闭环 —— 现为"限速 + 护栏 + 停滞回退"三层保护，且默认关闭；
- 问题 2：带宽规则曾过保守（1/(3T)），FOPDT T=0.5 对象 20s 仍有 8% 误差
  —— 现为 1/T 量级 + 相位裕量时滞帽，20s 内误差 < 2%。
"""

import numpy as np

from classic_controlling import (
    ADRC,
    FOPDTPlant,
    SecondOrderPlant,
    NonlinearPlant,
    fit_fopdt,
    auto_bandwidth,
)


def track(ctl, plant, r, steps, noise_std=0.0, seed=0):
    """闭环仿真 steps 步，返回输出序列。noise_std 为输出高斯噪声幅值。"""
    plant.reset()
    ctl.reset()
    rng = np.random.default_rng(seed)
    y = 0.0
    ys = np.empty(steps)
    for i in range(steps):
        u = ctl.step(r, y + noise_std * rng.standard_normal())
        y = plant.step(u)
        ys[i] = y
    return ys


def rail_margin(ctl):
    """b0 终值距最近护栏的距离，以护栏区间宽度归一化。"""
    lo, hi = sorted((0.3 * ctl.b0_initial, 3.0 * ctl.b0_initial))
    return min(ctl.b0 - lo, hi - ctl.b0) / (hi - lo)


# ---------------------------------------------------------------- 三对象盲测

def test_blind_fopdt():
    """一阶对象：T*ẏ + y = K*u，零人工参数，300 步内误差 < 5%。"""
    dt = 0.05
    plant = FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.0, dt=dt)
    ctl = ADRC.auto_tune(plant, dt=dt)
    ys = track(ctl, plant, r=1.0, steps=300)
    err = abs(ys[-1] - 1.0)
    assert np.all(np.isfinite(ys))
    assert err < 0.05, f"FOPDT 稳态误差 {err:.4f}"


def test_blind_second_order():
    """二阶欠阻尼对象：对 ADRC 存在阶次失配，300 步内误差 < 5%。"""
    dt = 0.05
    plant = SecondOrderPlant(gain=1.0, zeta=0.5, omega_n=3.0, dt=dt)
    ctl = ADRC.auto_tune(plant, dt=dt)
    ys = track(ctl, plant, r=1.0, steps=300)
    err = abs(ys[-1] - 1.0)
    assert np.all(np.isfinite(ys))
    assert err < 0.05, f"二阶对象稳态误差 {err:.4f}"


def test_blind_nonlinear():
    """非线性对象 ÿ = -ẏ + K*u + 0.5*y²：开环发散，0.5*y² 未建模。

    n_settle 取短（15 步）：对象开环不稳定，阶跃实验必须在响应
    发散前结束；FOPDT 拟合只需抓住早期动态即可，剩下的模型误差
    由 LESO 的总扰动估计吸收。
    """
    dt = 0.05
    plant = NonlinearPlant(gain=1.0, dt=dt)
    ctl = ADRC.auto_tune(plant, dt=dt, n_settle=15)
    ys = track(ctl, plant, r=1.0, steps=300)
    err = abs(ys[-1] - 1.0)
    assert np.all(np.isfinite(ys))
    assert err < 0.05, f"非线性对象稳态误差 {err:.4f}"


# ------------------------------------------------- 回归：问题 2（带宽曾过保守）

def test_autotune_converges_fast_on_fopdt():
    """复核复现配置：FOPDT(K=1, T=0.5, L=0.1, dt=0.05)，非自适应 400 步
    （20 s）稳态误差必须 < 2%（修复前同配置为 8.3%）。"""
    dt = 0.05
    plant = FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.1, dt=dt)
    ctl = ADRC.auto_tune(plant, dt=dt)  # 默认非自适应
    assert not ctl.adaptive
    ys = track(ctl, plant, r=1.0, steps=400)
    err = abs(ys[-1] - 1.0)
    assert np.all(np.isfinite(ys))
    assert err < 0.02, f"FOPDT(L=0.1) 20s 稳态误差 {err * 100:.2f}%，带宽仍过保守"


# ------------------------------------------- 回归：问题 1（RLS mis-specified）

def test_adaptive_never_much_worse_than_baseline():
    """自适应不得显著劣于非自适应基线，且 b0 终值不得贴护栏。

    复核复现配置（FOPDT K=1, T=0.5, L=0.1, dt=0.05）与无时滞变体：
    adaptive=True 的稳态误差与同 horizon 的 adaptive=False 之差 < 3 个百分点，
    b0 终值距上下护栏 > 10% 区间宽度。
    """
    dt = 0.05
    for delay, steps in ((0.1, 400), (0.1, 2000), (0.0, 400)):
        mk = lambda: FOPDTPlant(gain=1.0, time_constant=0.5, delay=delay, dt=dt)
        ys_off = track(ADRC.auto_tune(mk(), dt=dt, adaptive=False), mk(), 1.0, steps)
        ctl_on = ADRC.auto_tune(mk(), dt=dt, adaptive=True)
        ys_on = track(ctl_on, mk(), 1.0, steps)
        err_off = abs(ys_off[-1] - 1.0)
        err_on = abs(ys_on[-1] - 1.0)
        assert np.all(np.isfinite(ys_on)), f"L={delay} n={steps} 自适应闭环发散"
        assert err_on - err_off < 0.03, (
            f"L={delay} n={steps}: 自适应 {err_on * 100:.2f}% 比非自适应 "
            f"{err_off * 100:.2f}% 差超过 3pp"
        )
        m = rail_margin(ctl_on)
        assert m > 0.10, f"L={delay} n={steps}: b0 贴护栏（margin={m:.2f}）"


def test_rail_stagnation_rollback():
    """回退护栏机制（白盒）：b0 持续贴护栏且误差停滞时回退并冻结。

    模拟 mis-specified 失控场景：人为把 b0 钉在上护栏，测量恒为 0
    （误差恒 1.0 停滞），2*rail_patience 拍后必须回退到 b0_initial 并
    冻结（frozen=True）；reset 后解冻。
    """
    ctl = ADRC(b0=1.0, omega_c=1.0, omega_o=3.0, dt=0.05,
               adaptive=True, rail_patience=20)
    froze_at = None
    for k in range(200):
        if not ctl.frozen:
            ctl.b0 = 3.0 * ctl.b0_initial  # 钉在上护栏（模拟回归失控漂移）
        ctl.step(1.0, 0.0)  # 测量恒 0 -> 跟踪误差恒 1.0，停滞
        if ctl.frozen and froze_at is None:
            froze_at = k
    assert ctl.frozen, "贴护栏+误差停滞未触发回退"
    assert froze_at >= 2 * 20 - 1, f"回退过早: k={froze_at}"
    assert ctl.b0 == ctl.b0_initial, "回退后 b0 未恢复初值"
    ctl.reset()
    assert not ctl.frozen and ctl.b0 == ctl.b0_initial, "reset 未解冻"


# ------------------------------------------------------------------ RLS 收敛

def test_rls_b0_converges_toward_true_gain():
    """非线性对象上 adaptive=True：在线 RLS 应把偏差的 b0 拉向真实增益。

    带宽由 auto_tune 盲辨识给出；b0 初值人为设为真实增益的 2.5 倍
    （模拟辨识偏差/工况漂移）。断言闭环稳定跟踪，且最终 b0 比初值
    更接近 plant.true_gain（ÿ = ... + K*u 中 K 的真值）。
    """
    dt = 0.05
    plant = NonlinearPlant(gain=1.0, dt=dt)
    probe = ADRC.auto_tune(plant, dt=dt, n_settle=15)  # 盲辨识带宽
    ctl = ADRC(b0=2.5 * plant.true_gain, omega_c=probe.kp ** 0.5,
               omega_o=probe.beta3 ** (1 / 3), dt=dt, adaptive=True)
    ys = track(ctl, plant, r=1.0, steps=4000)
    assert np.all(np.isfinite(ys)), "自适应闭环必须保持稳定"
    assert abs(ys[-1] - 1.0) < 0.05, "自适应闭环仍应跟踪到位"

    b0_true = plant.true_gain
    d_init = abs(ctl.b0_initial - b0_true)
    d_final = abs(ctl.b0 - b0_true)
    assert len(ctl.b0_history) == 4001  # 初值 + 每拍一条记录
    assert d_final < d_init, (
        f"RLS 未向真值收敛: |b0_init - b*|={d_init:.3f} -> |b0_final - b*|={d_final:.3f}"
    )


# -------------------------------------------------------------- 时滞对象

def test_delay_plant_bandwidth_suppressed():
    """L=0.3s 的 FOPDT：带宽必须被时滞正确压制 —— 不振荡、稳定、误差 < 5%。

    时滞回路的有效收敛远慢于无时滞情形（扰动补偿滞后一个 L），
    故用长时域（2000 步）验证稳态；同时断言响应有界且末段无振荡。
    """
    dt = 0.1
    plant = FOPDTPlant(gain=1.0, time_constant=1.0, delay=0.3, dt=dt)
    ctl = ADRC.auto_tune(plant, dt=dt)
    K, T, L = ctl.identified
    # 辨识出的时滞应接近真实值 0.3s，且带宽确实被压制: omega_c * L 有界
    assert abs(L - plant.true_delay) < 0.15, f"时滞辨识偏差过大: L={L:.3f}"
    omega_c = ctl.kp ** 0.5
    assert omega_c * L < 1.0, f"带宽未被时滞压制: omega_c*L={omega_c * L:.2f}"

    ys = track(ctl, plant, r=1.0, steps=2000)
    assert np.all(np.isfinite(ys)), "时滞闭环发散"
    assert np.max(np.abs(ys)) < 1.5, "时滞闭环出现大幅超调/振荡"
    tail = ys[-400:]
    assert tail.max() - tail.min() < 0.05, "稳态段仍有振荡"
    err = abs(ys[-1] - 1.0)
    assert err < 0.05, f"时滞对象稳态误差 {err:.4f}"


# -------------------------------------------------------------- 噪声鲁棒

def test_noisy_output_still_converges():
    """输出叠加 1% 幅值高斯噪声：多随机种子下稳态误差均 < 8%（放宽阈值）。"""
    dt = 0.05
    for seed in range(5):
        plant = FOPDTPlant(gain=1.0, time_constant=0.5, delay=0.0, dt=dt)
        ctl = ADRC.auto_tune(plant, dt=dt)
        ys = track(ctl, plant, r=1.0, steps=300, noise_std=0.01, seed=seed)
        err = abs(ys[-1] - 1.0)
        assert np.all(np.isfinite(ys))
        assert err < 0.08, f"seed={seed} 噪声下稳态误差 {err:.4f}"


# ----------------------------------------------------- 辨识/带宽单元数值校验

def test_fit_fopdt_recovers_synthetic_params():
    """用无噪声合成 FOPDT 阶跃数据校验拟合精度（数值实现健全性）。"""
    dt, K_true, T_true, L_true = 0.01, 2.0, 0.8, 0.15
    plant = FOPDTPlant(gain=K_true, time_constant=T_true, delay=L_true, dt=dt)
    n = 600
    y = np.array([plant.step(1.0) for _ in range(n)])
    t = dt * (np.arange(n) + 1.0)
    K, T, L = fit_fopdt(t, y, du=1.0)
    assert abs(K - K_true) < 0.05, f"K={K:.3f}"
    assert abs(T - T_true) < 0.1, f"T={T:.3f}"
    assert abs(L - L_true) < 0.06, f"L={L:.3f}"


def test_auto_bandwidth_respects_hard_constraints():
    """带宽选择必须满足硬约束：时滞（2/L 与相位裕量帽）与采样（<= 0.2/dt）。"""
    wc, wo = auto_bandwidth(T=1.0, L=0.3, dt=0.01)
    assert wc <= 2.0 / 0.3 + 1e-12
    assert wc <= 0.2 / 0.01 + 1e-12
    assert wc * 0.3 + np.arctan(wc * 1.0) <= 0.6 + 1e-9  # 相位裕量帽
    assert wo == 3.0 * wc
    # 无时滞：wc 应达到 1/T 量级（问题 2 回归：不再保守到 1/(3T)）
    wc0, _ = auto_bandwidth(T=1.0, L=0.0, dt=0.01)
    assert np.isfinite(wc0) and wc0 > 0
    assert wc0 >= 0.5 / 1.0, f"无时滞带宽 {wc0:.3f} 仍过保守"
    # 时滞主导对象：相位帽必须真正把带宽压下来（实测 2/L 会发散）
    wcD, _ = auto_bandwidth(T=0.5, L=1.0, dt=0.05)
    assert wcD * 1.0 < 0.6, f"时滞主导对象带宽 {wcD:.3f} 过高"
