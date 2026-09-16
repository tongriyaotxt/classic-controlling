"""classic_controlling: 三个经典控制/估计算法的高质量实现。

- ADRC      : 线性自抗扰控制（带宽法 LADRC）
- UKF       : 向量化无迹卡尔曼滤波
- LinearMPC : 轻量线性模型预测控制（稠密式 QP + 内置 ADMM 求解器 + 热启动）
"""

from .adrc import ADRC
from .ukf import UKF
from .mpc import LinearMPC
from .sim import FOPDTPlant, SecondOrderPlant, NonlinearPlant
from .autotune import fit_fopdt, auto_bandwidth

__all__ = ["ADRC", "UKF", "LinearMPC",
           "FOPDTPlant", "SecondOrderPlant", "NonlinearPlant",
           "fit_fopdt", "auto_bandwidth"]
__version__ = "0.1.0"
