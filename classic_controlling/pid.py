"""工业 PID 家族：微分先行 / 前馈 / 抗积分饱和 / 串级。

位置式离散 PID，并联结构（积分时间 Ti、微分时间 Td 约定）：

    u = Kp*(b*r - y) + I + Kp*Td*d_filt + ff,   I' = (Kp/Ti)*e

工业必备件全部内置且默认开启：

- 微分先行（derivative on measurement）：微分作用于测量值而非误差，
  消除设定值跳变瞬间的"微分冲击"；
- 微分滤波：一阶低通，系数 N（滤波时间常数 Td/N）；
- 抗积分饱和：输出限幅 + 反算（back-calculation）；
- 前馈：``step(r, y, ff=...)`` 直接叠加到控制量（限幅对总和生效）。

整定便捷构造：``PID.ziegler_nichols``（Z-N 反应曲线法）与
``PID.imc_pi``（IMC-PI，MacGregor 规则，适用于时滞≈0 时 Z-N 退化的情形）。
"""

from __future__ import annotations

import numpy as np


class PID:
    """工业全功能离散 PID（位置式）。

    参数
    ----
    kp    : 比例增益
    ti    : 积分时间 Ti（秒），越大积分越弱
    td    : 微分时间 Td（秒），0 表示纯 PI
    dt    : 采样周期（秒）
    u_min / u_max : 输出限幅（None 表示该方向不限）
    derivative_on_measurement : 微分先行（默认 True）。True 时微分作用于
            测量值，设定值跳变不产生微分冲击；False 时微分作用于误差
            （传统 textbook PID，用于对照）
    n_filter : 微分滤波系数 N，滤波时间常数 = Td/N（默认 10）
    setpoint_weight : 比例设定值权重 b（默认 1；0 时比例也完全先行，
            设定值跳变更柔和）
    anti_windup : 反算抗积分饱和（默认 True，仅限幅起作用时有效果）
    kaw   : 反算增益（1/秒），None 时取 1/Ti（跟踪时间常数 = Ti）
    """

    def __init__(self, kp: float, ti: float, td: float, dt: float,
                 u_min: float | None = None, u_max: float | None = None,
                 derivative_on_measurement: bool = True,
                 n_filter: float = 10.0, setpoint_weight: float = 1.0,
                 anti_windup: bool = True, kaw: float | None = None):
        if ti <= 0:
            raise ValueError("ti 必须为正")
        if td < 0:
            raise ValueError("td 不能为负")
        if dt <= 0:
            raise ValueError("dt 必须为正")
        if n_filter <= 0:
            raise ValueError("n_filter 必须为正")
        if not (0.0 <= setpoint_weight <= 1.0):
            raise ValueError("setpoint_weight 必须在 [0, 1] 内")
        if u_min is not None and u_max is not None and u_min >= u_max:
            raise ValueError("u_min 必须小于 u_max")
        if kaw is not None and kaw <= 0:
            raise ValueError("kaw 必须为正")
        self.kp, self.ti, self.td, self.dt = map(float, (kp, ti, td, dt))
        self.u_min = None if u_min is None else float(u_min)
        self.u_max = None if u_max is None else float(u_max)
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.n_filter = float(n_filter)
        self.setpoint_weight = float(setpoint_weight)
        self.anti_windup = bool(anti_windup)
        self.kaw = float(kaw) if kaw is not None else 1.0 / self.ti

        self.integral = 0.0
        self._y_prev = 0.0
        self._e_prev = 0.0
        self._d_filt = 0.0  # 滤波后的微分率（先行时为 -dy/dt，否则为 de/dt）

    # ------------------------------------------------------------ 整定规则
    @classmethod
    def ziegler_nichols(cls, K: float, T: float, L: float, dt: float,
                        **kw) -> "PID":
        """Z-N 反应曲线整定: Kp=1.2T/(K·L), Ti=2L, Td=0.5L。

        L 过小（辨识退化）时兜底为 2*dt 并限幅 Kp 到 [0.01, 20]。
        """
        L_eff = max(float(L), 2.0 * dt)
        kp = 1.2 * T / (K * L_eff)
        kp = float(np.clip(kp, 0.01, 20.0))
        return cls(kp=kp, ti=2.0 * L_eff, td=0.5 * L_eff, dt=dt, **kw)

    @classmethod
    def imc_pi(cls, K: float, T: float, L: float, dt: float, **kw) -> "PID":
        """IMC-PI 整定（MacGregor 规则），极点对消，lambda = max(0.2T, 1.7L)。

        适用于 L≈0 时 Z-N 反应曲线法 Kp 发散的退化情形。
        """
        lam = max(0.2 * T, 1.7 * L)
        return cls(kp=T / (K * (lam + L)), ti=T, td=0.0, dt=dt, **kw)

    # ------------------------------------------------------------ 运行
    def reset(self):
        """积分、微分滤波、历史量全部归零。"""
        self.integral = 0.0
        self._y_prev = 0.0
        self._e_prev = 0.0
        self._d_filt = 0.0

    def _clip(self, u: float) -> float:
        if self.u_min is not None:
            u = max(u, self.u_min)
        if self.u_max is not None:
            u = min(u, self.u_max)
        return u

    def step(self, r: float, y: float, ff: float = 0.0) -> float:
        """推进一拍，返回控制量 u。

        参数
        ----
        r  : 设定值
        y  : 测量值
        ff : 前馈控制量（默认 0），叠加在反馈项之后一起受限幅约束
        """
        r, y = float(r), float(y)
        e = r - y

        # 比例（设定值权重 b 柔化跳变）
        p = self.kp * (self.setpoint_weight * r - y)

        # 微分（默认先行：对测量，消除设定值跳变的微分冲击）+ 一阶滤波
        if self.derivative_on_measurement:
            d_raw = -(y - self._y_prev) / self.dt
        else:
            d_raw = (e - self._e_prev) / self.dt
        tf = max(self.td / self.n_filter, self.dt)
        alpha = self.dt / (tf + self.dt)
        self._d_filt += alpha * (d_raw - self._d_filt)
        self._y_prev = y
        self._e_prev = e
        d = self.kp * self.td * self._d_filt

        # 积分（先试投，饱和后反算回收）
        i_trial = self.integral + (self.kp / self.ti) * e * self.dt
        u_unsat = p + i_trial + d + float(ff)
        u = self._clip(u_unsat)
        if self.anti_windup:
            self.integral = i_trial + self.kaw * (u - u_unsat) * self.dt
        else:
            self.integral = i_trial
        return u


class CascadePID:
    """串级 PID：外环输出作为内环设定值。

    典型用法：位置环（外）→ 速度环（内）、主温（外）→ 加热功率（内）。
    外环的 ``u_min/u_max`` 天然成为内环设定值限幅。两个环各自整定后
    组合，本类只负责信号路由与复位。

    参数
    ----
    outer : 外环 PID（其输出量纲 = 内环设定值量纲）
    inner : 内环 PID（其输出直接驱动被控对象）
    """

    def __init__(self, outer: PID, inner: PID):
        if not isinstance(outer, PID) or not isinstance(inner, PID):
            raise TypeError("outer / inner 必须是 PID 实例")
        self.outer = outer
        self.inner = inner

    def reset(self):
        """内外两环同时复位。"""
        self.outer.reset()
        self.inner.reset()

    def step(self, r: float, y_outer: float, y_inner: float,
             ff: float = 0.0) -> float:
        """推进一拍，返回内环控制量 u。

        参数
        ----
        r       : 外环设定值（最终控制目标）
        y_outer : 外环测量值
        y_inner : 内环测量值
        ff      : 前馈控制量（加在内环输出上）
        """
        r_inner = self.outer.step(r, y_outer)
        return self.inner.step(r_inner, y_inner, ff=ff)
