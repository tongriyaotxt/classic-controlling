"""ADRC 自动整定：阶跃响应 FOPDT 辨识 + 带宽自动选择。

流程：对被控对象做一次开环阶跃实验，把响应拟合为一阶加纯时滞
（FOPDT）模型 (K, T, L)，再据此按物理约束自动选择控制器带宽
``omega_c`` 与观测器带宽 ``omega_o``，全程无需人工调参。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["fit_fopdt", "auto_bandwidth"]

# 相位裕量预算（rad）：时滞回路的带宽帽按 wc*L + atan(wc*T) <= _PM_BUDGET 求解。
# 本离散实现（半隐式 LESO + b0=K/T 增益折算）上实测：wc*L >= ~0.6·L/(L+T) 量级
# 即开始振荡，0.6 rad 预留有实测安全余量（见 tests/test_autotune.py 时滞用例）。
_PM_BUDGET = 0.6


def _first_cross(t: np.ndarray, yn: np.ndarray, level: float, default: float) -> float:
    """归一化响应 yn 首次超过 level 的时刻；未到达时返回 default。"""
    idx = np.nonzero(yn >= level)[0]
    return float(t[idx[0]]) if idx.size else float(default)


def fit_fopdt(t: np.ndarray, y: np.ndarray, du: float) -> tuple[float, float, float]:
    """从阶跃响应数据拟合 FOPDT 模型 (K, T, L)。

    方法：增益 K = Δy/du 直接由首尾稳态差得到；再用
    ``scipy.optimize.curve_fit`` 对 y(t) = y0 + Δy*(1 - exp(-(t-L)/T))
    （t > L 段）做非线性最小二乘拟合。初值用两点法：63.2% 时刻估 T、
    28.3% 时刻估 L；curve_fit 失败时退回两点法 (Ziegler-Nichols)。

    参数
    ----
    t  : 时间向量（秒），单调递增
    y  : 阶跃响应输出，与 t 等长
    du : 阶跃输入幅值

    返回
    ----
    (K, T, L) : 稳态增益、时间常数（秒）、纯时滞（秒，>= 0）
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size or t.size < 5:
        raise ValueError("t 与 y 需等长且至少 5 个采样点")
    if du == 0.0:
        raise ValueError("du 不能为 0")

    y0 = float(y[0])
    n_tail = max(3, y.size // 10)
    yss = float(np.mean(y[-n_tail:]))  # 末段均值作稳态值，抗噪
    dy = yss - y0
    if not np.isfinite(dy) or abs(dy) < 1e-12:
        raise ValueError("阶跃响应无明显变化，无法辨识 FOPDT 模型")
    K = dy / du

    # ---- 两点法初值（Z-N）：t28 / t63 取首次穿越时刻 ----
    yn = (y - y0) / dy  # 归一化到 0 -> 1（对发散/超调响应也只取首次穿越）
    t_end = float(t[-1])
    t28 = _first_cross(t, yn, 0.283, 0.3 * t_end)
    t63 = _first_cross(t, yn, 0.632, 0.6 * t_end)
    T0 = 1.5 * (t63 - t28)
    L0 = 1.5 * t28 - 0.5 * t63
    if T0 <= 1e-9:  # 采样太稀或响应异常时的兜底
        T0 = max(0.2 * t_end, 1e-3)
        L0 = 0.0
    L0 = max(L0, 0.0)

    # ---- curve_fit 精拟合（幅值固定为 dy，只拟合 T 与 L）----
    def model(tt: np.ndarray, T: float, L: float) -> np.ndarray:
        tau = np.clip(tt - L, 0.0, None)
        return y0 + dy * (1.0 - np.exp(-tau / max(T, 1e-12)))

    try:
        popt, _ = curve_fit(model, t, y, p0=[T0, L0],
                            bounds=([1e-6, 0.0], [np.inf, t_end]),
                            maxfev=10000)
        T_fit, L_fit = float(popt[0]), max(float(popt[1]), 0.0)
        if not (np.isfinite(T_fit) and np.isfinite(L_fit)):
            raise RuntimeError("curve_fit 返回非有限值")
    except Exception:
        T_fit, L_fit = T0, L0  # 退回两点法

    return float(K), T_fit, L_fit


def _delay_pm_cap(T: float, L: float, budget: float = _PM_BUDGET) -> float:
    """时滞回路的带宽帽：解 wc*L + atan(wc*T) = budget（二分法，单调有界）。

    物理含义：回路在穿越频率处的相位损失（时滞 wc*L + 一阶滞后 atan(wc*T)）
    不超过 budget 弧度。实测表明离散 LADRC 在 budget ≈ 0.6 rad 以上开始振荡，
    而文献惯例 2/L 在本实现的所有时滞对象上均发散（过于乐观）。
    """
    lo, hi = 0.0, budget / L  # atan(wc*T) >= 0 => wc <= budget/L
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * L + np.arctan(mid * T) < budget:
            lo = mid
        else:
            hi = mid
    return lo


def auto_bandwidth(T: float, L: float, dt: float) -> tuple[float, float]:
    """根据辨识结果 (T, L) 与采样周期自动选择带宽。

    候选上限（取最严格者）：

    - 响应速度约束：omega_c < 1.0/T（闭环带宽不超过对象固有动态量级）；
    - 采样约束：omega_c < 0.1/dt（采样是硬约束；观测器 wo=3*wc 欧拉/半隐式
      离散在 wo*dt >~ 0.3 时开始振铃，实测 wo*dt = 1.0 必然发散）；
    - 时滞约束（L > 0 时两条，取更严）：文献惯例 omega_c < 2.0/L，以及
      相位裕量帽 wc*L + atan(wc*T) <= 0.6 rad —— 后者在本离散实现上更贴合
      实测稳定边界。

    并设下限 omega_c >= 0.1/T 以避免过慢；各上限在下限抬升后再次校验，
    不可违反。

    参数
    ----
    T  : 辨识得到的时间常数（秒）
    L  : 辨识得到的纯时滞（秒），L = 0 按无时滞处理
    dt : 控制采样周期（秒）

    返回
    ----
    (omega_c, omega_o) : 控制器带宽与观测器带宽，omega_o = 3*omega_c
                         （粗采样下半隐式 LESO 的实测稳定配比）
    """
    T = max(float(T), 1e-6)
    L = max(float(L), 0.0)
    if dt <= 0:
        raise ValueError("dt 必须为正")

    caps = [1.0 / T, 0.1 / dt]
    if L > 0.0:
        caps.append(2.0 / L)
        caps.append(_delay_pm_cap(T, L))
    omega_c = min(caps)
    omega_c = max(omega_c, 0.1 / T)  # 下限，避免过慢
    omega_c = min(omega_c, min(caps))  # 下限抬升不得违反任何上限

    omega_o = 3.0 * omega_c
    return float(omega_c), float(omega_o)
