"""工业 PID（微分先行 / 前馈 / 抗饱和 / 串级）测试。

对拍惯例：整定规则测试与经典公式逐数值核对；功能测试用
"同参数、只切换一个开关"的对照仿真断言该功能确实有效。
"""

import numpy as np
import pytest

from classic_controlling import PID, CascadePID, FOPDTPlant, SecondOrderPlant


def simulate(controller, plant, r, steps):
    ys = np.empty(steps)
    y = plant.y
    for i in range(steps):
        u = controller.step(r, y)
        y = plant.step(u)
        ys[i] = y
    return ys


# ---------------------------------------------------------------- 基本闭环
def test_step_response_converges():
    dt = 0.01
    plant = FOPDTPlant(gain=1.0, time_constant=1.0, delay=0.2, dt=dt)
    ctl = PID.ziegler_nichols(K=1.0, T=1.0, L=0.2, dt=dt)
    ys = simulate(ctl, plant, r=1.0, steps=int(10 / dt))
    assert abs(ys[-1] - 1.0) < 0.02, f"最终误差 {abs(ys[-1]-1.0):.4f}"


def test_output_limits_respected():
    ctl = PID(kp=10.0, ti=0.5, td=0.1, dt=0.01, u_min=-0.5, u_max=0.5)
    for _ in range(100):
        u = ctl.step(1.0, 0.0)
        assert -0.5 <= u <= 0.5


def test_reset_and_retrack():
    dt = 0.01
    plant = FOPDTPlant(gain=1.0, time_constant=1.0, delay=0.0, dt=dt)
    ctl = PID.imc_pi(K=1.0, T=1.0, L=0.0, dt=dt)
    simulate(ctl, plant, r=1.0, steps=int(5 / dt))
    ctl.reset()
    plant.reset()
    ys = simulate(ctl, plant, r=1.0, steps=int(10 / dt))
    assert abs(ys[-1] - 1.0) < 0.02


# ------------------------------------------------------------ 微分先行
def test_derivative_on_measurement_kills_kick():
    """设定值从 0 跳到 1 的第一拍：微分作用误差版有 Kp*Td/dt 量级的
    冲击，微分先行版（y 尚未动）完全没有。"""
    kw = dict(kp=1.0, ti=1.0, td=0.5, dt=0.01)
    u_meas = PID(derivative_on_measurement=True, **kw).step(1.0, 0.0)
    u_err = PID(derivative_on_measurement=False, **kw).step(1.0, 0.0)
    assert u_err > 5.0 * u_meas, \
        f"微分冲击未被抑制: on_error={u_err:.2f} on_meas={u_meas:.4f}"


def test_setpoint_weight_softens_kick():
    """b=0 时比例项也先行，设定值阶跃第一拍只剩积分项（≈0）。"""
    u_b1 = PID(kp=2.0, ti=1.0, td=0.0, dt=0.01, setpoint_weight=1.0).step(1.0, 0.0)
    u_b0 = PID(kp=2.0, ti=1.0, td=0.0, dt=0.01, setpoint_weight=0.0).step(1.0, 0.0)
    assert u_b1 == pytest.approx(2.0, abs=0.05)  # 2.0 比例 + 0.02 首拍积分
    assert abs(u_b0) < 0.05


# ------------------------------------------------------------ 抗积分饱和
def test_anti_windup_reduces_overshoot():
    """紧限幅 + 强积分：关抗饱和会积分累积（windup）导致更大超调。"""
    dt = 0.01
    kw = dict(kp=3.0, ti=0.5, td=0.0, dt=dt, u_min=-1.0, u_max=1.0)
    overshoots = {}
    for aw in (True, False):
        plant = SecondOrderPlant(gain=1.0, zeta=0.5, omega_n=3.0, dt=dt)
        ctl = PID(anti_windup=aw, **kw)
        ys = simulate(ctl, plant, r=1.0, steps=int(8 / dt))
        overshoots[aw] = max(0.0, ys.max() - 1.0)
    assert overshoots[True] < 0.7 * overshoots[False], \
        f"抗饱和未见效: aw={overshoots[True]:.3f} no_aw={overshoots[False]:.3f}"


# ------------------------------------------------------------ 前馈
def test_feedforward_improves_tracking():
    """斜坡参考跟踪：完美模型前馈 u_ff = (r + T·ṙ)/K 消除速度滞后；
    纯 PI 反馈对斜坡存在有限稳态误差（ṙ/Kv），IAE 应显著更大。"""
    dt = 0.01
    T, K = 1.0, 1.0
    steps = int(10 / dt)
    iae = {}
    for use_ff in (True, False):
        plant = FOPDTPlant(gain=K, time_constant=T, delay=0.0, dt=dt)
        ctl = PID(kp=0.5, ti=1.0, td=0.0, dt=dt)
        y = plant.y
        err_sum = 0.0
        for k in range(steps):
            r = 0.2 * (k * dt)  # 斜坡参考，ṙ = 0.2
            ff = (r + T * 0.2) / K if use_ff else 0.0
            y = plant.step(ctl.step(r, y, ff=ff))
            err_sum += abs(y - r) * dt
        iae[use_ff] = err_sum
    assert iae[True] < 0.3 * iae[False], \
        f"前馈未见效: ff={iae[True]:.3f} no_ff={iae[False]:.3f}"


# ------------------------------------------------------------ 串级
def _cascade_sim(cascade: bool, dt=0.01, t_pre=5.0, t_post=5.0, d=0.3):
    """内环对象(快, T=0.2) 串联 外环对象(慢, T=1.0)；t_pre 后向内环
    输入注入常值扰动 d。返回扰动后的最大偏差与末端误差。"""
    inner = FOPDTPlant(gain=1.0, time_constant=0.2, dt=dt)
    outer = FOPDTPlant(gain=1.0, time_constant=1.0, dt=dt)
    if cascade:
        ctl = CascadePID(outer=PID(kp=1.0, ti=1.0, td=0.0, dt=dt),
                         inner=PID(kp=5.0, ti=0.2, td=0.0, dt=dt))
    else:
        ctl = PID(kp=1.0, ti=1.0, td=0.0, dt=dt)
    devs = []
    for k in range(int((t_pre + t_post) / dt)):
        dist = d if k >= int(t_pre / dt) else 0.0
        if cascade:
            u = ctl.step(1.0, outer.y, inner.y)
        else:
            u = ctl.step(1.0, outer.y)
        v = inner.step(u + dist)
        y = outer.step(v)
        if k >= int(t_pre / dt):
            devs.append(abs(y - 1.0))
    return max(devs), devs[-1]


def test_cascade_rejects_inner_disturbance():
    """串级内环快速就地吸收扰动：扰动后最大偏差与稳态误差均显著更小。"""
    max_c, end_c = _cascade_sim(cascade=True)
    max_s, end_s = _cascade_sim(cascade=False)
    assert end_c < 0.05, f"串级未收敛: {end_c:.3f}"
    assert max_c < 0.6 * max_s, f"串级抑扰未见效: cascade={max_c:.3f} single={max_s:.3f}"
    assert end_c < end_s


# ------------------------------------------------------------ 整定规则对拍
def test_ziegler_nichols_matches_reference_formula():
    """与经典 Z-N 反应曲线公式逐数值核对（含 L 兜底与 Kp 限幅）。"""
    dt = 0.01
    K, T, L = 2.0, 3.0, 0.4
    pid = PID.ziegler_nichols(K, T, L, dt)
    L_eff = max(L, 2.0 * dt)
    assert pid.kp == pytest.approx(np.clip(1.2 * T / (K * L_eff), 0.01, 20.0))
    assert pid.ti == pytest.approx(2.0 * L_eff)
    assert pid.td == pytest.approx(0.5 * L_eff)
    # L≈0 退化：L 兜底为 2*dt
    pid0 = PID.ziegler_nichols(K, T, 0.0, dt)
    assert pid0.ti == pytest.approx(4.0 * dt)


def test_imc_pi_matches_reference_formula():
    """与 IMC-PI（MacGregor）公式逐数值核对。"""
    dt = 0.01
    K, T, L = 2.0, 3.0, 0.05
    pid = PID.imc_pi(K, T, L, dt)
    lam = max(0.2 * T, 1.7 * L)
    assert pid.kp == pytest.approx(T / (K * (lam + L)))
    assert pid.ti == pytest.approx(T)
    assert pid.td == 0.0


# ------------------------------------------------------------ 参数校验
def test_invalid_params_raise():
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=0.0, td=0.0, dt=0.01)
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=1.0, td=-0.1, dt=0.01)
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=1.0, td=0.0, dt=0.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=1.0, td=0.0, dt=0.01, u_min=1.0, u_max=0.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=1.0, td=0.0, dt=0.01, setpoint_weight=1.5)
    with pytest.raises(ValueError):
        PID(kp=1.0, ti=1.0, td=0.0, dt=0.01, kaw=0.0)
    with pytest.raises(TypeError):
        CascadePID(outer="not a pid", inner=PID(1.0, 1.0, 0.0, 0.01))
