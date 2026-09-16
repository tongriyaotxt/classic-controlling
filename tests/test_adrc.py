"""LADRC 闭环仿真测试。

被控对象: ÿ = -1.0*ẏ + 1.0*u + d （欧拉法离散）
"""

import numpy as np

from classic_controlling import ADRC


class SecondOrderPlant:
    """ÿ = -a*ẏ + b*u + d，欧拉法离散仿真器。"""

    def __init__(self, a=1.0, b=1.0, dt=0.01):
        self.a, self.b, self.dt = a, b, dt
        self.y = 0.0
        self.v = 0.0
        self.d = 0.0  # 外加扰动

    def step(self, u):
        self.v += self.dt * (-self.a * self.v + self.b * u + self.d)
        self.y += self.dt * self.v
        return self.y


def simulate(controller, plant, r, steps):
    ys = np.empty(steps)
    y = plant.y
    for i in range(steps):
        u = controller.step(r, y)
        y = plant.step(u)
        ys[i] = y
    return ys


def make(dt=0.01):
    return ADRC(b0=1.0, omega_c=4.0, omega_o=20.0, dt=dt), SecondOrderPlant(dt=dt)


def test_step_response_converges():
    dt = 0.01
    ctl, plant = make(dt)
    ys = simulate(ctl, plant, r=1.0, steps=int(10 / dt))
    assert abs(ys[-1] - 1.0) < 0.02, f"最终误差 {abs(ys[-1]-1.0):.4f}"


def test_constant_disturbance_rejection():
    dt = 0.01
    ctl, plant = make(dt)
    simulate(ctl, plant, r=1.0, steps=int(5 / dt))
    plant.d = 0.5  # 突加常值扰动
    ys = simulate(ctl, plant, r=1.0, steps=int(10 / dt))
    assert abs(ys[-1] - 1.0) < 0.05, f"加扰后稳态误差 {abs(ys[-1]-1.0):.4f}"


def test_reset_and_retrack():
    dt = 0.01
    ctl, plant = make(dt)
    simulate(ctl, plant, r=1.0, steps=int(5 / dt))
    ctl.reset()
    assert ctl.z1 == 0.0 and ctl.z2 == 0.0 and ctl.z3 == 0.0 and ctl.u == 0.0
    plant2 = SecondOrderPlant(dt=dt)  # 复位后在新对象上重新跟踪
    ys = simulate(ctl, plant2, r=2.0, steps=int(10 / dt))
    assert abs(ys[-1] - 2.0) < 0.04, f"reset 后跟踪误差 {abs(ys[-1]-2.0):.4f}"
