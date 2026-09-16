"""线性自抗扰控制（LADRC，Gao Zhiqiang 带宽法）。

二阶线性扩张状态观测器（LESO）+ PD 组合控制律，是韩京清 ADRC
的线性化版本。全部参数由两个带宽决定：

- 控制器带宽 ``omega_c`` : kp = omega_c**2, kd = 2*omega_c
- 观测器带宽 ``omega_o`` : beta1 = 3*omega_o, beta2 = 3*omega_o**2, beta3 = omega_o**3
"""

from __future__ import annotations

import numpy as np


class ADRC:
    """二阶线性自抗扰控制器（离散实现，半隐式欧拉 LESO）。

    LESO 按 z3 -> z2 -> z1 顺序用同一拍残差 e 依次更新（Gauss-Seidel 式
    半隐式欧拉）：在粗采样下比同步显式欧拉稳定边界更高，且使 RLS 回归
    方程 (z2_k - z2_{k-1})/dt - z3_k = beta2*e + b0*u_{k-1} 严格成立。

    可选在线自适应（experimental，默认关闭）：adaptive=True 时用带遗忘
    因子的递推最小二乘（RLS）在线修正增益估计 b0。三层安全机制：

    - 限速：|db0/dt| <= b0_rate * |b0_initial|（双时间尺度设计，自适应
      必须远慢于闭环，平均化理论的标准条件）；
    - 钳位：[0.3, 3.0] * b0_initial 护栏；
    - 回退冻结：若 b0 持续贴护栏且跟踪误差停滞不降（回归 mis-specified
      的典型征兆），自动回退到 b0_initial 并冻结自适应（frozen=True）。
    """

    def __init__(self, b0: float, omega_c: float, omega_o: float, dt: float,
                 adaptive: bool = False, forgetting: float = 0.995,
                 u_threshold: float = 0.05, rail_patience: int = 100,
                 b0_rate: float = 0.01):
        """
        参数
        ----
        b0          : 被控对象增益估计（ÿ = ... + b*u 中 b 的估计值）
        omega_c     : 控制器带宽
        omega_o     : 观测器带宽
        dt          : 采样周期
        adaptive    : 是否开启 b0 的在线 RLS 修正
        forgetting  : RLS 遗忘因子（0 < forgetting <= 1，越小遗忘越快）
        u_threshold : 激励阈值系数，仅当 |u| > u_threshold * 历史峰值 |u|
                      时才做 RLS 更新（保证激励、避免无激励漂移）
        rail_patience : 回退判定的窗口半长（拍）：b0 连续贴护栏满
                      2*rail_patience 拍且误差停滞时才回退冻结
        b0_rate     : 自适应速率上限（每秒最多改变 b0_rate * |b0_initial|，
                      双时间尺度：自适应应远慢于闭环带宽）
        """
        if b0 == 0.0:
            raise ValueError("b0 不能为 0")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("forgetting 必须在 (0, 1] 内")
        if rail_patience < 1:
            raise ValueError("rail_patience 必须为正整数")
        if b0_rate <= 0:
            raise ValueError("b0_rate 必须为正")
        self.b0 = float(b0)
        self.dt = float(dt)

        # 观测器增益（LESO 特征多项式 (s + omega_o)^3 的系数）
        self.beta1 = 3.0 * omega_o
        self.beta2 = 3.0 * omega_o**2
        self.beta3 = omega_o**3

        # 控制器增益（闭环特征多项式 (s + omega_c)^2）
        self.kp = omega_c**2
        self.kd = 2.0 * omega_c

        # LESO 状态与上一拍控制量
        self.z1 = 0.0  # 输出 y 的估计
        self.z2 = 0.0  # 输出导数的估计
        self.z3 = 0.0  # 总扰动的估计
        self.u = 0.0   # 上一拍控制量

        # 在线自适应（RLS 修正 b0）
        self.adaptive = bool(adaptive)
        self.forgetting = float(forgetting)
        self.u_threshold = float(u_threshold)
        self.b0_rate = float(b0_rate)
        self.b0_initial = self.b0            # 钳位基准
        self.b0_history = [self.b0]          # 每拍 b0 轨迹，供测试/诊断
        self._rls_p = 1.0                    # RLS 标量协方差
        self._z2_prev = 0.0
        self._u_prev = 0.0
        self._u_peak = 0.0                   # 历史 |u| 峰值（激励阈值基准）
        # 贴护栏回退检测（问题 1 护栏：回归 mis-specified 时 b0 会漂到护栏）
        self.rail_patience = int(rail_patience)
        self.frozen = False                  # True 表示已回退并冻结自适应
        self._rail_steps = 0
        self._rail_err1 = 0.0
        self._rail_err2 = 0.0

    def step(self, r: float, y: float) -> float:
        """推进一拍控制。

        参数
        ----
        r : 设定值
        y : 测量输出

        返回
        ----
        u : 控制量
        """
        # 1) 用上一拍的 u 推进 LESO（半隐式欧拉：z3 -> z2 -> z1 依次更新）
        e = y - self.z1
        self.z3 += self.dt * (self.beta3 * e)
        self.z2 += self.dt * (self.z3 + self.beta2 * e + self.b0 * self.u)
        self.z1 += self.dt * (self.z2 + self.beta1 * e)

        # 1.5) 在线 RLS 修正 b0（利用本拍更新后的 z2、z3 与上一拍 u）
        if self.adaptive:
            self._update_b0(r, y)

        # 2) 扰动补偿 + PD 控制律
        self.u = (self.kp * (r - self.z1) - self.kd * self.z2 - self.z3) / self.b0
        return self.u

    def _update_b0(self, r: float, y: float):
        """一步带遗忘因子的 RLS，在线修正增益估计 b0（experimental）。

        由半隐式 LESO 的更新方程构造标量回归（严格成立）：

            量测 m = (z2_k - z2_{k-1})/dt - z3_k  =  beta2*e_{k-1} + b0*u_{k-1}

        仅当 |u_{k-1}| 超过激励阈值时更新（保证激励、避免无激励漂移），
        单步修正量限速在 b0_rate * |b0_initial| * dt 以内（双时间尺度），
        并把 b0 钳位在 [0.3, 3.0] * b0_initial 的安全护栏内。

        注意：该回归对二阶对象近似无偏，但对一阶 LTI 对象 z2/z3 是观测器
        伪影，回归 mis-specified，b0 可能持续漂到护栏并拖慢/振荡闭环。
        因此带回退护栏：若 b0 连续 2*rail_patience 拍贴护栏，且后一半窗口
        的平均跟踪误差不小于前一半（停滞或恶化）、且误差水平超过 5%|r|，
        则回退 b0 = b0_initial 并永久冻结自适应（frozen=True，reset 解除）。
        """
        if self.frozen:
            self._z2_prev = self.z2
            self._u_prev = self.u
            self.b0_history.append(self.b0)
            return

        phi = self._u_prev  # 回归量：驱动上一段状态的输入
        m = (self.z2 - self._z2_prev) / self.dt - self.z3
        self._u_peak = max(self._u_peak, abs(phi))
        if self._u_peak > 0.0 and abs(phi) > self.u_threshold * self._u_peak:
            lam = self.forgetting
            p = self._rls_p
            gain = p * phi / (lam + phi * phi * p)
            delta = gain * (m - phi * self.b0)
            # 双时间尺度限速：自适应增量单步有界
            cap = self.b0_rate * abs(self.b0_initial) * self.dt
            self.b0 += min(max(delta, -cap), cap)
            self._rls_p = min((p - gain * phi * p) / lam, 1e6)  # 防协方差发散
            lo, hi = 0.3 * self.b0_initial, 3.0 * self.b0_initial
            if lo > hi:  # b0_initial 为负时交换护栏两端
                lo, hi = hi, lo
            self.b0 = min(max(self.b0, lo), hi)

        self._check_rail_stagnation(abs(r - y), abs(r))

        self._z2_prev = self.z2
        self._u_prev = self.u
        self.b0_history.append(self.b0)

    def _check_rail_stagnation(self, err_track: float, r_abs: float):
        """贴护栏 + 误差停滞检测：回归 mis-specified 时回退并冻结自适应。"""
        lo, hi = 0.3 * self.b0_initial, 3.0 * self.b0_initial
        if lo > hi:
            lo, hi = hi, lo
        width = hi - lo
        at_rail = (self.b0 - lo < 0.02 * width) or (hi - self.b0 < 0.02 * width)
        if not at_rail:
            self._rail_steps = 0
            self._rail_err1 = self._rail_err2 = 0.0
            return
        n = self.rail_patience
        if self._rail_steps < n:
            self._rail_err1 += err_track
        else:
            self._rail_err2 += err_track
        self._rail_steps += 1
        if self._rail_steps >= 2 * n:
            m1 = self._rail_err1 / n
            m2 = self._rail_err2 / n
            # 误差水平显著且不再改善 -> 判定 mis-specified，回退冻结
            if m2 > 0.05 * max(r_abs, 1e-9) and m2 >= 0.9 * m1:
                self.b0 = self.b0_initial
                self.frozen = True
            self._rail_steps = 0
            self._rail_err1 = self._rail_err2 = 0.0

    def reset(self):
        """清空观测器状态和控制量，恢复到初始条件（含自适应状态与冻结标志）。"""
        self.z1 = self.z2 = self.z3 = 0.0
        self.u = 0.0
        # 自适应状态一并复位
        self.b0 = self.b0_initial
        self.b0_history = [self.b0]
        self._rls_p = 1.0
        self._z2_prev = 0.0
        self._u_prev = 0.0
        self._u_peak = 0.0
        self.frozen = False
        self._rail_steps = 0
        self._rail_err1 = 0.0
        self._rail_err2 = 0.0

    @classmethod
    def auto_tune(cls, plant, dt: float, u_step: float = 1.0,
                  n_settle: int = 200, adaptive: bool = False) -> "ADRC":
        """一键自整定：阶跃实验 -> FOPDT 辨识 -> 带宽自动选择。

        1) 阶跃实验：给 plant 施加幅值 u_step 的阶跃输入，采集 n_settle 步响应
        2) fit_fopdt 得 (K, T, L)
        3) b0 初值取 K/T（一阶对象的增益折算到二阶 ADRC 框架，
           LADRC 对阶次/增益失配本身鲁棒）
        4) auto_bandwidth 依物理约束（时滞相位裕量、采样）自动选带宽
        5) 返回配置好的 ADRC 实例

        adaptive=True 时开启在线 RLS 修正 b0（experimental，默认关闭：
        该回归对一阶 LTI 对象 mis-specified，已带贴护栏回退保护，
        但非自适应基线在良态对象上已经足够快且更可靠）。

        参数
        ----
        plant    : 被控对象，需实现 reset() 与 step(u) 协议（见 sim.py）
        dt       : 控制采样周期（秒），应与 plant 内部 dt 一致
        u_step   : 辨识用阶跃输入幅值（应使响应清晰可辨且不损坏对象）
        n_settle : 阶跃实验采样步数
        adaptive : 返回的控制器是否开启 b0 在线 RLS 修正
        """
        from .autotune import auto_bandwidth, fit_fopdt

        plant.reset()
        y = np.empty(n_settle)
        for k in range(n_settle):
            y[k] = plant.step(u_step)
        t = dt * (np.arange(n_settle) + 1.0)

        K, T, L = fit_fopdt(t, y, u_step)
        b0 = K / T  # T > 0 已由 fit_fopdt 保证
        if not np.isfinite(b0) or b0 == 0.0:
            b0 = 1.0  # 辨识退化时的兜底

        omega_c, omega_o = auto_bandwidth(T, L, dt)
        ctl = cls(b0=b0, omega_c=omega_c, omega_o=omega_o, dt=dt,
                  adaptive=adaptive)
        ctl.identified = (K, T, L)  # 保留辨识结果供诊断/测试
        return ctl
