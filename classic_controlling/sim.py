"""测试/演示用虚拟被控对象。

统一协议::

    plant.reset()     -> 归零内部状态
    y = plant.step(u) -> 施加控制量 u，内部按固定 dt 离散仿真推进一拍，返回输出 y

所有类都暴露真实物理参数（如 ``.true_gain``），供自动整定测试做盲测断言。
"""

from __future__ import annotations


class FOPDTPlant:
    """一阶加纯时滞对象:  T*ẏ + y = K*u(t - L)，欧拉法离散。

    参数
    ----
    gain          : 稳态增益 K（输出稳态值 / 输入稳态值）
    time_constant : 时间常数 T（秒），阶跃响应达到 63.2% 所需时间
    delay         : 纯时滞 L（秒），输入作用延迟到输出的时间
    dt            : 仿真步长 / 采样周期（秒）
    """

    def __init__(self, gain: float = 1.0, time_constant: float = 1.0,
                 delay: float = 0.0, dt: float = 0.01):
        if time_constant <= 0:
            raise ValueError("time_constant 必须为正")
        if dt <= 0:
            raise ValueError("dt 必须为正")
        if delay < 0:
            raise ValueError("delay 不能为负")
        self.true_gain = float(gain)
        self.true_time_constant = float(time_constant)
        self.true_delay = float(delay)
        self.dt = float(dt)
        # 时滞折算成采样拍数（队列长度，0 表示无时滞直通）
        self._n_delay = int(round(self.true_delay / self.dt))
        self.y = 0.0
        self._buf: list[float] = [0.0] * self._n_delay

    def reset(self):
        """输出与时滞队列归零。"""
        self.y = 0.0
        self._buf = [0.0] * self._n_delay

    def step(self, u: float) -> float:
        self._buf.append(float(u))
        u_d = self._buf.pop(0)  # n_delay 拍之前的输入（n_delay=0 时即当前 u）
        # 欧拉积分: ẏ = (K*u(t-L) - y) / T
        self.y += self.dt / self.true_time_constant * (self.true_gain * u_d - self.y)
        return self.y


class SecondOrderPlant:
    """二阶欠阻尼对象:  ÿ + 2ζωn ẏ + ωn² y = K ωn² u，欧拉法离散。

    参数
    ----
    gain    : 稳态增益 K（输出稳态值 / 输入稳态值）
    zeta    : 阻尼比 ζ，< 1 为欠阻尼（阶跃响应有超调）
    omega_n : 自然频率 ωn（rad/s）
    dt      : 仿真步长 / 采样周期（秒）
    """

    def __init__(self, gain: float = 1.0, zeta: float = 0.5,
                 omega_n: float = 3.0, dt: float = 0.01):
        if omega_n <= 0:
            raise ValueError("omega_n 必须为正")
        if zeta <= 0:
            raise ValueError("zeta 必须为正")
        if dt <= 0:
            raise ValueError("dt 必须为正")
        self.true_gain = float(gain)
        self.true_zeta = float(zeta)
        self.true_omega_n = float(omega_n)
        self.dt = float(dt)
        self.y = 0.0   # 位置输出
        self.v = 0.0   # 速度状态

    def reset(self):
        self.y = 0.0
        self.v = 0.0

    def step(self, u: float) -> float:
        wn = self.true_omega_n
        acc = (self.true_gain * wn**2 * float(u)
               - 2.0 * self.true_zeta * wn * self.v
               - wn**2 * self.y)
        self.v += self.dt * acc
        self.y += self.dt * self.v
        return self.y


class NonlinearPlant:
    """含未建模非线性的二阶对象:  ÿ = -ẏ + K*u + 0.5*y²。

    0.5*y² 项对控制器完全未知（ADRC 把它当作"总扰动"的一部分），
    且它使开环响应发散，用于展示 ADRC 的鲁棒性与在线自适应的价值。

    参数
    ----
    gain : 输入增益 K（ÿ 方程中 u 的系数，即 ADRC 框架下 b 的真值）
    dt   : 仿真步长 / 采样周期（秒）
    """

    def __init__(self, gain: float = 1.0, dt: float = 0.01):
        if dt <= 0:
            raise ValueError("dt 必须为正")
        self.true_gain = float(gain)
        self.dt = float(dt)
        self.y = 0.0
        self.v = 0.0

    def reset(self):
        self.y = 0.0
        self.v = 0.0

    def step(self, u: float) -> float:
        acc = -self.v + self.true_gain * float(u) + 0.5 * self.y**2
        self.v += self.dt * acc
        self.y += self.dt * self.v
        return self.y
