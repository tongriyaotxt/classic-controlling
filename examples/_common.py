"""examples 共用工具：绘图风格、Z-N 整定 PID、阶跃性能指标。

仅供 examples/ 内演示脚本使用，不属于包 API。
"""
from __future__ import annotations

import numpy as np

_ZH_OK = False


def setup_plot():
    """配置 matplotlib：中文字体（找不到则退回英文标签）、网格、负号。"""
    global _ZH_OK
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境下也能出图
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    zh_fonts = [f for f in ("Microsoft YaHei", "SimHei") if f in names]
    _ZH_OK = bool(zh_fonts)
    if zh_fonts:
        plt.rcParams["font.sans-serif"] = zh_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["lines.linewidth"] = 1.6
    return plt


def L(zh: str, en: str) -> str:
    """按字体可用性选择中文/英文标签，避免豆腐块。"""
    return zh if _ZH_OK else en


# ------------------------------------------------------------------ PID（对照组）

from classic_controlling import PID  # noqa: E402  包内实现，此处仅为别名


def auto_tune_pid(K: float, T: float, L: float, dt: float):
    """基于辨识模型的 PID 整定（对照组），返回 (PID, 规则名)。

    - L >= 2*dt：经典 Ziegler-Nichols 反应曲线法（``PID.ziegler_nichols``）；
    - L ≈ 0（Z-N 反应曲线不适用，Kp 会发散）：回退到 IMC-PI
      （MacGregor 规则，``PID.imc_pi``）。
    """
    if L >= 2.0 * dt:
        return PID.ziegler_nichols(K, T, L, dt), "Z-N"
    return PID.imc_pi(K, T, L, dt), "IMC-PI"


def identify_fopdt(plant, dt: float, u_step: float = 1.0, n_settle: int = 200):
    """对 plant 做开环阶跃实验并用包内 fit_fopdt 辨识 (K, T, L)。"""
    from classic_controlling import fit_fopdt

    plant.reset()
    y = np.empty(n_settle)
    for k in range(n_settle):
        y[k] = plant.step(u_step)
    t = dt * (np.arange(n_settle) + 1.0)
    return fit_fopdt(t, y, u_step)


# ------------------------------------------------------------------ 性能指标

def step_metrics(t: np.ndarray, y: np.ndarray, r: float, idx_dist: int,
                 settle_band: float = 0.02, rec_band: float = 0.05) -> dict:
    """阶跃响应指标。

    超调量/调节时间在扰动施加前的窗口内计算；扰动恢复时间为扰动后
    输出最后一次越出 ±rec_band 带到回到带内的耗时。
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    rr = abs(r)

    pre = y[:idx_dist]
    overshoot = max(0.0, (pre.max() - r) / rr * 100.0)

    viol = np.nonzero(np.abs(pre - r) > settle_band * rr)[0]
    if viol.size == 0:
        t_settle = 0.0
    elif viol[-1] == idx_dist - 1:
        t_settle = float("nan")  # 扰动施加前未进入调节带
    else:
        t_settle = float(t[viol[-1] + 1])

    post_err = np.abs(y[idx_dist:] - r)
    viol2 = np.nonzero(post_err > rec_band * rr)[0]
    if viol2.size == 0:
        t_recover = 0.0
    elif viol2[-1] == len(post_err) - 1:
        t_recover = float("inf")  # 直到仿真结束仍未恢复
    else:
        t_recover = float(t[idx_dist + viol2[-1] + 1] - t[idx_dist])

    iae = float(np.sum(np.abs(y - r)) * (t[1] - t[0]))
    return dict(overshoot=overshoot, t_settle=t_settle,
                t_recover=t_recover, iae=iae)
