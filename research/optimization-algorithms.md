# 经典数学优化算法开源实现现状调研

> 调研日期：2026-09-15。方法：以一手来源为准——GitHub 官方仓库（README/issue/提交记录）、官方文档、Mittelmann benchmark 及其可视化镜像（mattmilten.github.io/mittelmann-plots）、官方技术博客与论文。GitHub star 数与最近推送时间来自 GitHub REST API（调研当日查询）。

---

## 0. 总览：一句话结论

- **LP/MILP**：HiGHS 是唯一世界级的开源求解器，但与商业求解器仍有 5–20 倍差距；GPU 一阶方法（PDLP 系）正在快速重塑格局。GLPK 与 lp_solve 事实上冻结。
- **QP**：OSQP 1.0 已有模块化线性代数后端（CUDA/MKL），但 GPU 后端覆盖面窄；qpOASES/qpSWIFT 维护近乎停滞。
- **NLP**：IPOPT 自身完全串行，并行全部押在外部线性求解器上；NLopt 内大量算法仍是 Fortran 77 时代代码的转写，bug 累累。
- **无导数/黑盒**：pycma、pymoo 等纯 Python 实现存在明显解释器开销；DEAP 实质停更；Powell 方法族正被 PRIMA 项目现代化重写，但尚未被下游（scipy/NLopt）普遍采纳。
- **组合优化启发式**：通用 SA/Tabu/LNS 开源库生态薄弱（simanneal 停更、Tabu 搜索无成熟开源实现），LocalSolver/Hexaly（闭源）在此领域碾压开源。
- **梯度类**：scipy 的 L-BFGS-B/SLSQP/COBYLA 是 Fortran 77 包装，存在已知缺陷；C++ 侧 liblbfgs 停更。

---

## 1. 线性规划（LP）：单纯形法与内点法

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | Benchmark 表现 |
|---|---|---|---|---|---|
| **HiGHS** | [ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS) | 1832 | C++（含 C），C/C#/FORTRAN/Python/Julia/Rust 接口 | 活跃（调研当日仍有提交；v1.12 引入新内点求解器 HiPO，HiPDLP 一阶求解器开发中） | Mittelmann LPopt：得分 13.07，解出 82%（COPT 1.0 为基准）；Network-LP：6.59；开源中最强 [来源](https://mattmilten.github.io/mittelmann-plots/) |
| **CLP** | [coin-or/Clp](https://github.com/coin-or/Clp) | 493 | C++ | 低活跃维护（2026-09 有提交） | LPopt 得分 27.38、解出 62%，落后 HiGHS 约 2 倍 [来源](https://mattmilten.github.io/mittelmann-plots/) |
| **GLPK** | [GNU GLPK](https://www.gnu.org/software/glpk/) | —（非 GitHub 主导） | ANSI C | **事实冻结**：5.0 发布于 2020 年，版权移交 FSF 后无重大更新 [来源](https://www.automicvault.com/zh-hans/pkg/brew/glpk/) | ALGLIB 对拍中落后 HiGHS/CLP 约 2 倍 [来源](https://www.alglib.net/linear-programming/benchmark.php) |
| **lp_solve** | [lp-solve/lp_solve](https://github.com/lp-solve/lp_solve)（SourceForge 为主） | — | C | **事实冻结**：主版本 5.5.2.5 停留在 2016–2017 年 [来源](https://www.freshports.org/math/lp_solve/) | 落后 HiGHS/CLP 约 3 倍 [来源](https://www.alglib.net/linear-programming/benchmark.php) |
| **NVIDIA cuOpt**（对照新势力） | [NVIDIA/cuopt](https://github.com/NVIDIA/cuopt) | 1040 | CUDA/C++，Apache 2.0 | 高度活跃（调研当日有提交） | Mittelmann LPfeas 开源第一（62/65 解出）；H100 上比 CPU 求解器 60% 实例更快、20% 实例快 10 倍以上、单例最高 5000 倍 [来源1](https://github.com/NVIDIA/cuopt/discussions/1767) [来源2](https://developer.nvidia.com/blog/accelerate-large-linear-programming-problems-with-nvidia-cuopt/) |
| **cuPDLP-C / cuPDLP.jl / HPR-LP**（一阶 GPU 方法） | [COPT 的 cuPDLP-C](https://arxiv.org/html/2506.02174v1) 等 | — | C / Julia | 活跃研究前沿 | HPR-LP.jl 在 Mittelmann 集上比 cuPDLP.jl 多解 3–5 题，SGM10 提速 2.7–3.7 倍 [来源](https://www.polyu.edu.hk/ama/profile/dfsun//files/HPR-LP_Published2025.pdf) |

### 关键事实

- HiGHS 是 SciPy 的默认 LP/MIP 求解器、MATLAB 默认求解器、JuMP 文档默认求解器 [来源](https://maths.ed.ac.uk/research/data-decisions/optimization-and-operational-research/software)。
- HiGHS 旧内点求解器 IPX **纯串行、与商业 IPM 差距大、无法推广到 QP**；v1.12 新增 HiPO（LDLᵀ 分解 + Metis 排序 + 多前线法并行消元树，依赖 BLAS），由 Google 资助开发 [来源](https://jump.dev/assets/jump-dev-workshops/2025/talk-julian-hall.pdf)。
- HiGHS 内点法在大型能源模型上可慢于 Gurobi **60–100 倍** [来源](https://highs.dev/assets/HiGHS_funding_proposal.pdf)。
- PDLP 类一阶方法有精度/鲁棒性问题：HiGHS 官方通讯报告 cuPDLP-C 在个别 Mittelmann 问题上相对准则满足但**绝对误差达 100 量级**，且在构造的小问题上发散 [来源](https://highs.dev/assets/HiGHS_Newsletter_25_0.pdf)。

### 优化机会

1. **HiPO 的 Python 发行与 BLAS 依赖工程化**：HiPO 目前"从源码构建/JuMP 可用，Python 构建 WIP" [来源](https://jump.dev/assets/jump-dev-workshops/2025/talk-julian-hall.pdf)。打通 highspy 发行链 = 直接惠及 SciPy/MATLAB 全部下游用户。
2. **HiGHS 并行化深化**：官方自述改进方向包括"dualisation、primal simplex switch、sifting、改进并行求解器、Idiot crash 与 crossover" [来源](https://www.fields.utoronto.ca/talk-media/1/41/79/slides.pdf)。单纯形并行（PAMI/PARIP 系）仍有明确空间。
3. **PDLP 的精度与鲁棒性修补**：crossover 后处理、自适应重启、绝对误差控制是已证实的痛点（见上），也是 HPR-LP 已展示可行性的方向 [来源](https://www.polyu.edu.hk/ama/profile/dfsun//files/HPR-LP_Published2025.pdf)。
4. **GLPK/lp_solve 不值得优化**：冻结 + GPL/老架构，正确策略是迁移到 HiGHS。

---

## 2. 二次规划（QP）

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | 特点/Benchmark |
|---|---|---|---|---|---|
| **OSQP** | [osqp/osqp](https://github.com/osqp/osqp) | 2193 | C（自包含，无外部依赖），Python/Julia/R/Matlab/Rust 接口 | 活跃（1.0 引入模块化线性代数后端） | v1.0 支持可切换代数后端：标准 CSC、CUDA、Intel MKL；FPGA 实验性 [来源](https://ism.engineer/content/slides/2023-05_SIAM-OPT_OSQP.pdf) |
| **qpOASES** | [coin-or/qpOASES](https://github.com/coin-or/qpOASES) | 553 | C++ | 低维护（COIN-OR 托管镜像，80 个未处理 issue） | 稠密活动集法，小问题快；规模增大后不敌 HPIPM [来源](https://arxiv.org/html/2510.21773v2) |
| **qpSWIFT** | [qpSWIFT/qpSWIFT](https://github.com/qpSWIFT/qpSWIFT) | 156 | ANSI C | **停滞**：最近推送 2023-03 | 稀疏原始-对偶内点法，面向嵌入式/机器人 |
| **HPIPM / PROXQP / DAQP / PIQP / Clarabel**（对照） | 见 [实时 QP 求解器综述](https://arxiv.org/html/2510.21773v2) | — | C/Rust | 活跃 | 腿足机器人基准：稀疏 MPC 上 HPIPM 最快；OSQP 稳健但偏慢 [来源](https://arxiv.org/html/2510.21773v2) |

### 关键事实

- OSQP GPU 版（cuOSQP，CUDA C）在大规模问题上比 CPU 版**快达两个数量级**，但全部数据驻留 GPU 显存、不支持多 GPU，且小问题 GPU 无优势 [来源](https://arxiv.org/pdf/1912.04263)。
- OSQP 1.0 的 GPU 路线图（HIP/OpenMP 可移植实现、cuDSS 直接法、批量模式、低/混合精度、GPU 上灵敏度计算）多数仍是"进行中" [来源](https://ism.engineer/content/slides/2025-03_SIAM-CSE_OSQP-GPU.pdf)。
- qpOASES 在嵌入式实时仿真中有已知的初始化阶段 CPU 过载问题（issue #72 悬置） [来源](https://github.com/coin-or/qpOASES/issues/72)。

### 优化机会

1. **OSQP GPU 后端的显存管理与多 GPU 扩展**：官方实现承认"问题规模受单卡显存限制；统一内存/多 GPU 是明确扩展方向" [来源](https://arxiv.org/pdf/1912.04263)。
2. **OSQP 批量 QP（batched）求解**：MPC/RL 场景一次求解成百上千个同构 QP，批处理 GPU 求解已被证明可行且有需求 [来源](https://ism.engineer/content/slides/2025-03_SIAM-CSE_OSQP-GPU.pdf)。
3. **qpSWIFT/q pOASES 接班人缺位**：嵌入式稀疏 IPM 的 qpSWIFT 停更，社区需要一个有热启动、代码生成、持续维护的轻量 C 实现（PIQP/Clarabel 部分填补但 Rust/C++ 门槛不同）。

---

## 3. 非线性规划（NLP）

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | 关键瓶颈 |
|---|---|---|---|---|---|
| **IPOPT** | [coin-or/Ipopt](https://github.com/coin-or/Ipopt) | 1782 | C++ | 活跃 | **求解器本体零并行**——"Ipopt itself does not have any parallelization enabled"，并行完全依赖外部线性求解器（MUMPS/HSL/Pardiso/SPRAL）[来源](https://github.com/coin-or/Ipopt/issues/533) |
| **SNOPT**（闭源对照） | — | — | Fortran 77 | 商业 | 稀疏 SQP，常作为 IPOPT 的对照基准 |
| **NLopt** | [stevengj/nlopt](https://github.com/stevengj/nlopt) | 2274 | C/C++ | 活跃（2.11 发布于 2026-06） | 大量算法是 Powell Fortran 77 的 C 转写，已知 bug 群（见下） |
| **PRIMA**（对照/替代） | [libprima/prima](https://github.com/libprima/prima) | 413 | 现代 Fortran(F2008)+C/Python/MATLAB/Julia | 高度活跃 | Powell 方法族（COBYLA/NEWUOA/BOBYQA/LINCOA/UOBYQA）的现代化参考实现 |

### 关键事实

- IPOPT + MUMPS 的坑：MUMPS 非线程安全（IPOPT 3.14 起用互斥锁保护）、并行运行多个 IPOPT 实例会崩溃（issue #733） [来源1](https://coin-or.github.io/Ipopt/FAQ.html) [来源2](https://github.com/coin-or/Ipopt/issues/733)。
- IPOPT 默认 MUMPS 只跑单线程；用户切换到 HSL MA86/97 后也常"看不到提速"，因配置门槛高（BLAS 线程数需设为 1 避免过度订阅）[来源](https://discourse.julialang.org/t/how-to-speed-up-ipopt-with-hsl-ma86-using-parallel-processing/117567)。
- NLopt 的 COBYLA/BOBYQA/NEWUOA 有一整个 bug 谱系：无限循环挂起（scipy #8998、#15527；nlopt #370、#118）、未初始化变量导致段错误、返回非最优点甚至违反约束却报成功（nlopt #57、#552、#254） [来源](https://github.com/stevengj/nlopt/issues/501)。NLopt 官方 issue 明确讨论"旧 COBYLA 实现 buggy 且难以维护，是否切换到 PRIMA"。
- GPU 批量 NLP 是新方向：arXiv 2026 论文用 JAX 批量 IPM 在单 GPU 上并行求解大量 NLP，指出 IPOPT 无法做异构迭代融合 [来源](https://arxiv.org/html/2606.26341v1)。

### 优化机会

1. **把 NLopt/scipy 的 Powell 族替换为 PRIMA 后端**：PRIMA 修复了 Fortran 77 原版的全部已知 bug（无限循环、段错误、返回劣点），且在函数评估次数上优于原实现；SciPy 纳入 PRIMA "正在讨论中" [来源](https://github.com/libprima/prima)。这是**低难度高影响力**的工程：接口形状完全一致。
2. **IPOPT 的并行化与 GPU 线性求解器接入**：本体串行 + 默认 MUMPS 单线程，引入 GPU 稀疏直接法（如 cuDSS/SPRAL-GPU）或多实例批量求解是明确的架构级机会 [来源](https://github.com/coin-or/Ipopt/issues/533)。
3. **IPOPT 线性求解器默认配置自动化**："MA86 不提速"类问题本质是线程配置陷阱，可做自动检测与推荐，影响所有下游用户。

---

## 4. 无导数 / 黑盒优化

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | 备注 |
|---|---|---|---|---|---|
| **scipy.optimize** | [scipy/scipy](https://github.com/scipy/scipy) | 15015 | Python + Fortran 77 包装（Nelder-Mead 为纯 Python） | 高度活跃（1837 个 open issues） | Nelder-Mead 有边界 bug（issue #19991）；COBYLA 挂起 bug 群 |
| **DEAP** | [DEAP/deap](https://github.com/DEAP/deap) | 6439 | 纯 Python | **实质停更**：最新 release 1.4.1（2023-07），281 个未处理 issue [来源](https://informatica.vu.lt/journal/INFORMATICA/article/1384/text) | 13 框架横评中 GA 实现垫底 [同上] |
| **pymoo** | [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) | 2957 | 纯 Python/NumPy | 活跃 | 核心模块残留大量可向量化的 Python 循环（issue #774） [来源](https://github.com/anyoptimization/pymoo/issues/774) |
| **pycma** | [CMA-ES/pycma](https://github.com/CMA-ES/pycma) | 1356 | 纯 Python | 活跃（4.x 持续修 bug） | 官方实现，功能最全但单进程 |
| **cmaes**（轻量对照） | [CMA-ES/cmaes](https://github.com/CMA-ES/cmaes) | — | Python ask-and-tell | 活跃 | 接口天然支持并行评估 [来源](https://arxiv.org/html/2402.01373v2) |
| **pagmo2** | [esa/pagmo2](https://github.com/esa/pagmo2) | 936 | C++（pygmo 绑定） | 活跃 | 横评中 GA/DE 实现质量最佳；island/archipelago 内建并行 [来源](https://informatica.vu.lt/journal/INFORMATICA/article/1384/text) |
| **nevergrad** | [facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) | 4207 | Python | 活跃 | ask/tell 接口、542 个算法变体 |

### 关键事实

- 13 个元启发式框架的受控横评（Informatica, 2025）：**同一算法在不同框架中的实现差异会显著改变求解质量**；DEAP 的 GA 显著垫底且缺边界控制，pymoo 的 DE 替换了标准选择规则、PSO 边界处理需迭代重试 [来源](https://informatica.vu.lt/journal/INFORMATICA/article/1384/text)。
- pymoo 的 CMA-ES 包装暴露 `parallelize`（批量评估）开关，说明种群级并行是公认需求 [来源](https://pymoo.org/algorithms/soo/cmaes.html)。
- pycma 用户普遍需要借助 mpi4py 等外部手段并行评估种群；官方 issue 231 等显示大种群下仍有 corner-case 失败在修 [来源1](https://inria.hal.science/hal-02173682/document) [来源2](https://github.com/CMA-ES/pycma)。
- Nelder-Mead 理论上可在二维严格凸函数上收敛到非驻点（McKinnon 反例），NLopt 作者 stevengj 直言其"已过时、应避免使用" [来源1](https://link.springer.com/article/10.1007/s11075-021-01221-7) [来源2](https://discourse.julialang.org/t/why-nelder-mead-minimization-without-minimal-property-check/102236)。

### 优化机会

1. **pymoo 的向量化/编译加速**：官方 issue 已列出具体文件（`rnd.py`、`misc.py`、`pcx.py`）的可向量化循环；进一步可用 Numba/Cython 处理热点算子。影响面 = 所有 pymoo 用户的每一代迭代 [来源](https://github.com/anyoptimization/pymoo/issues/774)。
2. **pycma 的一等并行/批 GPU 评估**：种群评估天然并行，但官方实现把并行留给用户手工拼装（mpi4py）；内置 `multiprocessing`/批量接口是低难度高实用改进（pymoo 与 cmaes 库的接口设计已证明可行性）。
3. **DEAP 的正确性修复或正式退役引导**：GA 垫底 + 无边界控制 + 停更，大量存量用户仍在使用——修复边界控制或文档层面引导迁移到 pymoo/pagmo 都有公共价值 [来源](https://informatica.vu.lt/journal/INFORMATICA/article/1384/text)。
4. **scipy Nelder-Mead 现代化**：纯 Python 实现 + 已知边界 bug；替换为带重启/充分下降条件的变体（Kelley 重启、Fortran 或 C 实现）可同时解决性能与收敛缺陷。

---

## 5. 组合优化启发式（SA / Tabu / LNS）

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | 备注 |
|---|---|---|---|---|---|
| **OR-Tools（CP-SAT + 路由库）** | [google/or-tools](https://github.com/google/or-tools) | 14047 | C++（多语言绑定） | 高度活跃（调研当日有提交） | CP-SAT = CDCL 核心 + LP 松弛 + 并行组合 portfolio + LNS/违规局部搜索子求解器 [来源](https://d-krupke.github.io/cpsat-primer/search_core.html) |
| **simanneal** | [perrygeo/simanneal](https://github.com/perrygeo/simanneal) | 692 | 纯 Python | **停更**（最近推送 2024-05，实质多年无功能更新） | 单线程 SA 教学级实现 |
| **jMetal** | [jMetal/jMetal](https://github.com/jMetal/jMetal) | 559 | Java | 活跃（6.7, 2025-05） | 以多目标 EA 为主，非通用局部搜索框架 |
| **LocalSolver/Hexaly**（闭源对照） | — | — | C++ | 商业活跃 | JSSP 大横评：在 n≫m 实例上 Hexaly gap 0.34% 对 OR-Tools >30%；装配线平衡上碾压 CP Optimizer [来源1](https://www.mdpi.com/2227-7390/14/12/2179) [来源2](https://homepages.laas.fr/artigues/manuscrit_these_lea_blaise.pdf) |

### 关键事实

- CP-SAT 的并行是**配置组合（portfolio）+ LNS 交织子求解器**，额外核数可带来超线性加速；但实证显示 7 个 worker 之后收益消失甚至变慢（通信开销） [来源1](https://d-krupke.github.io/cpsat-primer/search_core.html) [来源2](https://www.solvermax.com/blog/well-that-escalated-quickly-or-tools)。
- 大规模 JSSP 受控评测（2026, MDPI Mathematics）：OptalCP > CP Optimizer > OR-Tools > Hexaly（按平均秩）；但结构性场景（n≫m）中元启发式引擎（Hexaly）碾压区间传播类求解器 [来源](https://www.mdpi.com/2227-7390/14/12/2179)。
- **通用 SA/Tabu 开源生态近乎空白**：Python 侧只有教学级 simanneal；Java 侧 jMetal 偏 EA；没有与 LocalSolver 对标的开源通用大规模邻域搜索求解器。

### 优化机会

1. **开源通用 LNS/局部搜索求解器的空白**：LocalSolver 的成功证明"模型无关的邻域搜索 + 增量评估"架构的价值；开源世界（除 OR-Tools 路由特化外）没有等价物。这是**影响力最大但难度也最大**的方向。
2. **simanneal 级别的 SA 实现升级**：纯 Python、单线程、无自适应退火调度——一个现代化的 SA 库（NumPy 向量化/批量候选评估/自适应温度/并行重启）可轻易提速数量级。
3. **CP-SAT worker 组合的自适应调度**：实证显示默认组合在部分模型上过犹不及；按实例特征在线裁剪 portfolio（现有 `subsolvers`/`ignore_subsolvers` 参数只是手动杠杆）是具体可做方向 [来源](https://www.solvermax.com/blog/well-that-escalated-quickly-or-tools)。

---

## 6. 梯度类方法（L-BFGS / CG / 信赖域）

### 主流实现

| 实现 | 链接 | Star | 语言 | 维护状态 | 备注 |
|---|---|---|---|---|---|
| **scipy L-BFGS-B / SLSQP / trust-constr** | [scipy/scipy](https://github.com/scipy/scipy) | 15015 | Fortran 77 包装（trust-constr 为纯 Python） | 高度活跃 | L-BFGS-B 不尊重 maxiter、dtype 硬编码 double、回调输出与迭代值不一致（issue #8372/#8373）；SLSQP 多处边界/缩放 bug（#10081、#14861） [来源1](https://github.com/scipy/scipy/issues/8373) [来源2](https://github.com/scipy/scipy/issues/10081) |
| **liblbfgs** | [chokkan/liblbfgs](https://github.com/chokkan/liblbfgs) | 599 | C | **停更**（最近推送 2023-06，17 个未处理 issue） | 曾是 C 侧事实标准 |
| **LBFGSpp** | [yixuan/LBFGSpp](https://github.com/yixuan/LBFGSpp) | 661 | 头文件 C++（Eigen） | 活跃 | 现代 header-only 替代品 |

### 优化机会

1. **scipy 的 Fortran 77 遗产替换**：L-BFGS-B/SLSQP/COBYLA 三个最常用的 `minimize` 后端全部是 1988–1995 年的 Fortran 代码包装，bug 修复困难（issue #8373 标题即"scipy.optimize has broken my trust"）。PRIMA（COBYLA）已有现代化替代，L-BFGS-B 可用现代 C++ 重写并对拍原实现。影响力极大（scipy 全量用户）[来源](https://github.com/scipy/scipy/issues/8373)。
2. **trust-constr 的性能工程**：纯 Python 实现的信赖域内点法，热点（投影 CG、SQP 子问题）有编译化空间；已知 NaN/inf 崩溃路径（issue #10081）。
3. **liblbfgs 退役引导**：C 用户应迁往 LBFGSpp 或自维护 fork；文档/包管理层面值得标注。

---

## 7. 优先级排序表（影响力 × 可行性）

评分：影响力（用户基数 × 性能/正确性收益）与可行性（工程量、接口兼容性、有无已验证路径），均为 1–5。

| 排名 | 优化对象 | 方向 | 影响力 | 可行性 | 依据 |
|---|---|---|---|---|---|
| 1 | **scipy/NLopt 的 Powell 族 → PRIMA 后端** | 修正确性 bug（挂起/段错误/返回劣点）+ 提速 | 5 | 5 | bug 清单完整、PRIMA 已成熟且接口等价、SciPy 已在讨论纳入 [来源](https://github.com/libprima/prima) |
| 2 | **pymoo 热点向量化/编译化** | 消除核心算子的 Python 循环 | 4 | 5 | 官方 issue 已列出具体文件与改法 [来源](https://github.com/anyoptimization/pymoo/issues/774) |
| 3 | **HiGHS：HiPO 发行链 + 单纯形并行深化** | 内点法并行化落地到 Python/SciPy 下游 | 5 | 3 | HiPO 已存在但 Python 构建 WIP；官方自述改进清单 [来源](https://jump.dev/assets/jump-dev-workshops/2025/talk-julian-hall.pdf) |
| 4 | **pycma 一等并行评估** | 内置种群级并行/批量接口 | 3 | 5 | 用户目前手工 mpi4py；ask/tell 与批量模式已有先例 [来源](https://arxiv.org/html/2402.01373v2) |
| 5 | **OSQP GPU：显存扩展 + batched 模式** | 统一内存/多 GPU/批量 QP | 4 | 3 | 官方论文与路线图明确点名 [来源](https://arxiv.org/pdf/1912.04263) |
| 6 | **scipy L-BFGS-B / SLSQP / trust-constr 现代化** | Fortran 77 遗产重写或对拍修复 | 5 | 2 | issue #8373/#10081；工作量大但用户基数最大 [来源](https://github.com/scipy/scipy/issues/8373) |
| 7 | **IPOPT 并行/GPU 线性求解器接入** | 本体串行的架构级突破 | 4 | 2 | issue #533；GPU 批量 IPM 已有学术验证 [来源](https://arxiv.org/html/2606.26341v1) |
| 8 | **通用 SA/Tabu/LNS 开源求解器** | 对标 LocalSolver 的空白领域 | 4 | 2 | Hexaly 在大规模实例上的碾压证明价值 [来源](https://www.mdpi.com/2227-7390/14/12/2179) |
| 9 | **DEAP 修复或退役引导** | 边界控制修复/迁移文档 | 3 | 4 | 横评 GA 垫底、281 issue 未处理 [来源](https://informatica.vu.lt/journal/INFORMATICA/article/1384/text) |
| 10 | **GLPK / lp_solve / qpSWIFT / liblbfgs** | 不建议投入：迁移引导即可 | 2 | 5 | 全部事实冻结 [来源](https://www.freshports.org/math/lp_solve/) |

---

## 附：数据可信度说明

- Star 数与 `pushed_at` 来自 GitHub REST API，查询时间 2026-09-15。
- Benchmark 数字以 Mittelmann 可视化镜像（mattmilten.github.io/mittelmann-plots，数据日期 2026 年 4–6 月）与各论文/官方博客为准；商业求解器对比数字（如"60–100 倍"）来自 HiGHS 官方筹资文档，可能随版本变化。
- 本报告未实际运行任何 benchmark，所有性能断言均为引用。
