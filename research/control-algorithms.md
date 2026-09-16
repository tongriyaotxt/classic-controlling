# 经典/传统控制算法开源实现现状调研

> 调研时间：2026-09-15。数据来源：GitHub REST API（star 数、最近 push 时间、open issues 均为查询当日实时值）、各仓库 README 与源码、GitHub Issues。除特别注明外，"维护状态"以 `pushed_at`（最后一次 push 时间）为准。

## 总览速查表

| 实现 | 仓库 | Stars | 语言 | 最近 push | Open Issues | 覆盖算法族 |
|---|---|---|---|---|---|---|
| python-control | [python-control/python-control](https://github.com/python-control/python-control) | 2079 | Python | 2026-08 | 89 | PID/极点配置/LQR/LQG |
| Slycot | [python-control/Slycot](https://github.com/python-control/Slycot) | 156 | Python(Fortran 后端) | 2026-06 | 7 | 线性系统数值核心 |
| ControlSystems.jl | [JuliaControl/ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) | 581 | Julia | 2026-09 | 42 | 线性反馈全家桶 |
| do-mpc | [do-mpc/do-mpc](https://github.com/do-mpc/do-mpc) | 1452 | Python | 2026-09 | 95 | MPC/NMPC/MHE |
| acados | [acados/acados](https://github.com/acados/acados) | 1470 | C | 2026-09 | 79 | NMPC 实时求解器 |
| CasADi | [casadi/casadi](https://github.com/casadi/casadi) | 2292 | C++ | 2026-09 | 730 | MPC 底层符号/AD 框架 |
| OSQP | [osqp/osqp](https://github.com/osqp/osqp) | 2193 | C | 2026-01 | — | 线性 MPC 的 QP 求解 |
| filterpy | [rlabbe/filterpy](https://github.com/rlabbe/filterpy) | 3870 | Python | **2024-02** | 85 | KF/EKF/UKF/粒子滤波 |
| pykalman | [pykalman/pykalman](https://github.com/pykalman/pykalman) | 1331 | Python | 2026-04 | 85 | KF/EM 学习 |
| Stone Soup | [dstl/Stone-Soup](https://github.com/dstl/Stone-Soup) | 655 | Python | 2026-09 | 144 | 粒子滤波/多目标跟踪 |
| scikit-fuzzy | [scikit-fuzzy/scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) | 876 | Python | **2024-08** | 49 | 模糊控制 |
| fuzzylite | [fuzzylite/fuzzylite](https://github.com/fuzzylite/fuzzylite) | 326 | C++ | 2025-09 | — | 模糊控制 |
| Arduino-PID-Library | [br3ttb/Arduino-PID-Library](https://github.com/br3ttb/Arduino-PID-Library) | 2202 | C++ | **2024-05** | 87 | 嵌入式 PID |
| simple-pid | [m-lundberg/simple-pid](https://github.com/m-lundberg/simple-pid) | 909 | Python | 2024-07 | 21 | PID |
| qlibs | [kmilo17pet/qlibs](https://github.com/kmilo17pet/qlibs) | 103 | C | 2026-03 | — | 嵌入式 PID/模糊 |

---

## 1. 线性反馈类（PID、极点配置、LQR/LQG）

### 主流实现

- **python-control**（2079★，活跃，[README](https://github.com/python-control/python-control/blob/main/README.rst)）：Python 生态事实标准，提供 `place`（极点配置）、`lqr/lqe`（LQR/LQG）、`rootlocus_pid_designer` 等。重数值计算依赖可选的 **Slycot**（封装 Fortran SLICOT 库）。
- **ControlSystems.jl**（581★，非常活跃）：Julia 生态主力，配 [RobustAndOptimalControl.jl](https://github.com/JuliaControl/RobustAndOptimalControl.jl)（66★，2026-09 活跃）覆盖 H∞/LQG 等。
- **Slycot**（156★）：SLICOT 的 Fortran→Python 封装，是 python-control 高性能路径（Riccati 方程、状态空间变换）的关键。
- **Arduino-PID-Library**（2202★，C++）：嵌入式 PID 事实标准，但 2024-05 后基本停更，87 个 open issues。
- **simple-pid**（909★，纯 Python）：带抗饱和（输出限幅+积分分离）的教学/轻量级 PID。
- **qlibs**（103★，C，活跃）：嵌入式 PID + 模糊 + 插值库。

### 优化机会分析

1. **python-control 在无 Slycot 时退回纯 Python/NumPy 路径，性能与数值稳定性双输。** Slycot 只有 156★、7 个 open issues，安装门槛高（需 Fortran 工具链）。机会：把常用例程（`care/dare` Riccati 求解、 staircase 变换）做成可选的 Numba/Cython fallback，或推广 [SciPy 已有的 `solve_continuous_are`] 路径。证据：README 明确将 Slycot 作为可选加速后端（[README.rst](https://github.com/python-control/python-control/blob/main/README.rst)）。
2. **`hinfsyn` 存在已知性能缺陷**：issue [#152 "hinfsyn: very slow if stabilizing controller cannot be found"](https://github.com/python-control/python-control/issues/152) 至今仍 open——迭代不收敛时无早停/无 warm restart，γ-迭代的二分搜索每次从零求解 Riccati 方程。可操作优化：复用上一步 Riccati 解做热启动 + 早停判据。
3. **仿真层面**：历史 issue [#48 "lsim is slow"](https://github.com/python-control/python-control/issues/48)（已关闭但说明 `lsim` 长期是热点）；[#936](https://github.com/python-control/python-control/issues/936)（变步长积分器与离散控制器无法组合）仍是 open——连续/离散混合仿真的插值重采样是性能与精度双重痛点。
4. **Arduino-PID-Library**：issue [#148](https://github.com/br3ttb/Arduino-PID-Library/issues/148) 明确要求"for better efficiency"的改进（浮点运算可在定点 MCU 上换成定点/Q格式、去掉每次 Compute 的重复 `millis()` 调用与时间窗口检查的冗余分支）。库已事实停更（2024-05 后无实质更新，87 open issues），是"高影响力 × 低维护"的典型。
5. **simple-pid**：纯 Python 标量循环，无 NumPy/批量接口，无法一次对 N 个回路并行求值（DRL 训练/硬件在环批量仿真场景）。向量化改造成本极低（[仓库](https://github.com/m-lundberg/simple-pid)）。

---

## 2. 状态估计类（KF/EKF/UKF/粒子滤波）

### 主流实现

- **filterpy**（3870★）：星数最高，但 **2024-02 后停止维护**。作者在 README 中明确承认"pedalogical aim…even when that has a performance cost"（教学优先于性能，[README](https://github.com/rlabbe/filterpy)）。
- **pykalman**（1331★）：曾停更多年，2025 年起由新维护者复活，2026-01 发布 0.11.2（[commit 记录](https://github.com/pykalman/pykalman/commits)）。
- **Stone Soup**（655★，英国 Dstl 官方项目，活跃）：面向多目标跟踪，粒子滤波/EKF/UKF 齐全，144 个 open issues。

### 优化机会分析

1. **filterpy 的 UKF/EnKF 用纯 Python 循环逐 sigma 点/粒子传播**，这是最直接的可操作优化点：sigma 点传播 `fx` 调用逐个循环（[UKF.py L717/L726](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py)），EnKF 的 update 里 `for i in range(N): sigmas_h[i] = self.hx(self.sigmas[i])`（[ensemble_kalman_filter.py](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/ensemble_kalman_filter.html)）。若 `fx/hx` 支持批量接口即可整批向量化，或用 Numba 并行 `prange`。对 2n+1 个 sigma 点的小系统，NumPy 小矩阵开销反而拖累——README 自述"numpy's overhead on small matrices makes it run slower than writing each equation out by hand"，说明作者已知该问题但选择不优化。**机会：写一个 API 兼容的高性能 fork（filterpy 已停更，85 个 open issues 无人处理，如 [#319 pip 循环导入](https://github.com/rlabbe/filterpy/issues/319)、[#326 NumPy 拷贝弃用警告](https://github.com/rlabbe/filterpy/issues/326) 都是低成本修复点）。**
2. **粒子滤波的重采样/逐粒子循环未 SIMD 化**：filterpy 与 Stone Soup 的 PF 都是 Python 层逐粒子循环。系统性优化空间：向量化重采样（系统性重采样可纯 NumPy O(N) 实现）、权重计算的批量归一化、Numba 并行粒子传播。Stone Soup 活跃（2026-09 有合并 PR），提 PR 有被接受的可能。
3. **数值稳定性**：filterpy 默认 Joseph 形式协方差更新但平方根滤波（`square_root.py`）覆盖不全；多数实现缺 UD 分解/平方根 UKF 的完整路径，长时间运行的嵌入式场景会漂移。
4. **pykalman 刚复活**，代码仍是旧的纯 NumPy 标量风格，EM 算法有已知崩溃 bug 修复中（[commit "[BUG] KalmanFilter.em() crashes when n_timesteps=1"](https://github.com/pykalman/pykalman/commits)）——维护窗口期，提性能 PR 的时机好。

---

## 3. 预测与约束类（MPC/NMPC/DMC）

### 主流实现

- **do-mpc**（1452★，活跃）：CasADi 后端的全 Python NMPC/MHE 框架。
- **acados**（1470★，非常活跃）：C 语言实时 NMPC 求解器，SQP/RTI、HPIPM、代码生成，嵌入式可用（[官方特性页](https://docs.acados.org/features/index.html)）。
- **CasADi**（2292★，非常活跃，730 open issues）：符号微分/NLP 框架，do-mpc 与 acados 的共同底座。
- **OSQP**（2193★，C）：线性 MPC 常用 QP 求解器。
- **MPT3**：MATLAB 工具箱（[mpt3.org](https://www.mpt3.org/)），显式 MPC 事实标准，闭源 MATLAB 生态。
- **DMC**：无成熟库。GitHub 搜索 "dynamic matrix control" 最高仅 16★（[csianglim/DMC-Python](https://github.com/csianglim/DMC-Python)，Jupyter notebook，2022 年停更），其余为个位数 star 的课程作业。

### 优化机会分析

1. **do-mpc 求解未热启动/数据记录开销大**：do-mpc 走 CasADi+IPOPT 路径，每步完整 NLP 求解，无 RTI 式的 preparation/feedback 拆分（对比 acados 的 [RTI 文档](https://docs.acados.org/features/index.html)）；issue [#406 "Deactivate automatic storage of data in MPC, MHE, Simulator classes"](https://github.com/do-mpc/do-mpc/issues/406)（open）指出自动数据存储是内存与时间开销来源。可操作优化：求解器热启动选项、可选关闭数据记录、对线性 MPC 提供 OSQP 快速路径。
2. **线性 MPC 的 Python 生态断层**：学术原型要么用 do-mpc（重）要么手写 CVXPY（慢，每步重建问题）。机会：做一个"线性 MPC 编译一次、QP 参数化复用、OSQP 热启动"的轻量库——技术上完全可行（OSQP 支持参数化 QP 更新），市场上是空白。
3. **DMC 无维护中的开源实现**——工业界（过程控制）用量巨大但开源只有课程代码。空白即机会：DMC 本质是离线算好阶跃响应系数 + 在线 QP/最小二乘，实现一个向量化 NumPy 版工作量小。
4. **acados 本身是优化得最充分的**（C、RTI、HPIPM、代码生成），其空间更多在易用性而非性能；不作为优化目标，而作为"性能标杆"对照。

---

## 4. 非线性与鲁棒类（滑模、反步法、模糊、ADRC）

### 主流实现现状

这一族**没有成熟通用库**，全部是零散的论文复现/课程仓库：

- **滑模控制**：GitHub 搜索 "sliding mode control" 排名前列的是 MATLAB 单项目仓库，如 [harishsatishchandra/SMC-controller](https://github.com/harishsatishchandra/SMC-controller)（125★，2019 停更）、[sandeshthapa/Adaptive_Sliding_Mode_Control_of_Aerial_Manipulator](https://github.com/sandeshthapa/Adaptive_Sliding_Mode_Control_of_Aerial_Manipulator)（124★）。
- **反步法**：同样零散，最高 [iman-sharifi-ghb/Quadcopter-...](https://github.com/iman-sharifi-ghb/Quadcopter-Trajectory-Tracking-using-Adaptive-Nonlinear-Algorithms)（137★，MATLAB）。
- **模糊控制**：**scikit-fuzzy**（876★）**2024-08 后停更**（[commit 记录](https://github.com/scikit-fuzzy/scikit-fuzzy/commits)），纯 Python/NumPy，控制推理部分（`skfuzzy.control`）是逐规则 Python 循环求值；**fuzzylite**（326★，C++，活跃）及其 Python 绑定 pyfuzzylite（81★）。
- **ADRC**：无超过 40★ 的实现。最高 [frank1ma/DDRTC-of-UMSs](https://github.com/frank1ma/DDRTC-of-UMSs)（35★，MATLAB，2021 停更）；[psiorx/ADRC](https://github.com/psiorx/ADRC)（28★，C++，**2015 年停更**）。线性 ADRC（LADRC，带宽参数化后只需调 2-3 个参数）工程上极实用，但**没有任何语言有维护中的库**。

### 优化机会分析

1. **ADRC 是最大的"空白型机会"**：算法本身计算量极小（一个线性 ESO + PD 组合），缺的不是性能优化而是**合格的标准化实现**（离散化正确性、参数整定工具、抗噪微分器）。做一个带单元测试和整定指南的 Python/C 库即可成为该领域事实标准。
2. **scikit-fuzzy 的 `control` 模块性能差且停更**：推理过程逐规则、逐隶属函数 Python 循环，无法批量推理。向量化（把规则前件写成张量广播）或直接重写为 Numba 核，预期 1-2 个数量级加速。证据：仓库 2024-08 后无 commit（[commits](https://github.com/scikit-fuzzy/scikit-fuzzy/commits)），49 个 open issues。
3. **滑模/反步法零散 MATLAB 代码的通病**：符号微分（导数项）用数值差分近似、无向量化仿真、无单元测试。整合进 python-control 生态（其已支持非线性 I/O 系统）是清晰路径。

---

## 5. 自适应类（MRAC、自整定、ILC）

### 现状

- **MRAC**：GitHub 搜索 "model reference adaptive control" 最高 160★ 且是无关仓库；真正的实现如 [AleksandarHaber/Model-Reference-Adaptive-Control---MIT-Rule](https://github.com/AleksandarHaber/Model-Reference-Adaptive-Control---MIT-Rule)（22★，MATLAB）均为教学代码。
- **ILC**：搜索 "iterative learning control" 前列是 [mlab-upenn/LearningMPC](https://github.com/mlab-upenn/LearningMPC)（170★，C++，2020 停更，实为学习 MPC）和 MATLAB 论文代码（如 [Geng-Hao/Robust-Model-free-ILC...](https://github.com/Geng-Hao/Robust-Model-free-Iterative-Learning-Control-with-Convergence-Rate-Acceleration)，44★，2020 停更）。**无通用 ILC 库**。
- **自整定 PID**：无独立库；python-control 的 `rootlocus_pid_designer` 是交互式设计而非在线自整定。

### 优化机会分析

1. **整族空白**：MRAC（含 L1 自适应）、STR、ILC 没有任何维护中的跨语言库。这是"建库"级机会而非"优化"级机会——参考实现应包含：归一化/投影算子保证鲁棒性、死区修正、与 python-control 的 `NonlinearIOSystem` 集成。
2. ILC 计算上是离线批量问题（每次迭代解一个 least-squares / lifted-system QP），天然可向量化+GPU 化，现有 MATLAB 代码全是 for 循环逐次迭代。

---

## 优先级排序表（影响力 × 优化可行性）

| 排名 | 目标 | 类型 | 影响力 | 可行性 | 理由 |
|---|---|---|---|---|---|
| 1 | **filterpy 高性能 fork/补丁**（UKF/EnKF/PF 向量化+Numba） | 优化 | ★★★★★（3870★，教学生态入口） | ★★★★★（纯 Python 循环，改动局部） | 作者明示性能非目标且已停更，社区需求积压（85 open issues） |
| 2 | **ADRC 标准化开源库**（Python+C 双实现，含整定工具） | 建库 | ★★★★☆（工业需求大，无人机/电机控制热门） | ★★★★★（算法简单，LADRC 仅 ESO+PD） | 现状最高 35★ 且全停更，无竞争品 |
| 3 | **轻量线性 MPC 库**（QP 参数化 + OSQP 热启动 + DMC 特例） | 建库/优化 | ★★★★☆（介于 CVXPY 与 do-mpc 之间的空档） | ★★★★☆（复用 OSQP/CasADi 即可） | do-mpc #406 等 issue 证明重框架有性能抱怨 |
| 4 | **scikit-fuzzy.control 向量化重写** | 优化 | ★★★☆☆（876★ 但停更） | ★★★★☆（推理核是张量运算，易并行） | 停更项目，可能需 fork |
| 5 | **python-control `hinfsyn` 热启动/早停 + 无 Slycot fallback 加速** | 优化 | ★★★☆☆（2079★ 生态核心） | ★★★☆☆（需懂 H∞ 迭代数值细节） | issue #152 长期 open，维护活跃、PR 可落地 |
| 6 | **Arduino-PID-Library 效率改进** | 优化 | ★★★☆☆（2202★ 但场景窄） | ★★★★☆（issue #148 已给出清单） | 项目停更，大概率需 fork |
| 7 | **MRAC/ILC 通用库** | 建库 | ★★★☆☆（学术需求为主） | ★★★☆☆（鲁棒性修正需理论功底） | 完全空白，但受众窄于 ADRC |

## 附：事实来源索引

- star/issue/活跃度数据：GitHub REST API `api.github.com/repos/<owner>/<repo>`，查询于 2026-09-15。
- filterpy 性能立场：[README "I opt for clear code … even when that has a performance cost"](https://github.com/rlabbe/filterpy)
- filterpy UKF 逐点循环：[filterpy/kalman/UKF.py](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py)
- python-control hinfsyn 慢：[issue #152](https://github.com/python-control/python-control/issues/152)；lsim 慢：[issue #48](https://github.com/python-control/python-control/issues/48)；混合仿真：[issue #936](https://github.com/python-control/python-control/issues/936)
- do-mpc 数据记录开销：[issue #406](https://github.com/do-mpc/do-mpc/issues/406)
- Arduino-PID 效率：[issue #148](https://github.com/br3ttb/Arduino-PID-Library/issues/148)
- acados RTI 能力（作为性能标杆）：[docs.acados.org/features](https://docs.acados.org/features/index.html)
- DMC/ADRC/SMC/MRAC/ILC 仓库排名：GitHub Search API `search/repositories?q=...&sort=stars`，2026-09-15 查询结果。
