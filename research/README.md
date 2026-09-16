# 传统算法开源实现优化机会 · 总调研报告

> 调研方法：以[第一性原理框架](./first-principles-framework.md)（五种根本资源 × 六条优化杠杆）为纲，
> 三个子调研分别深入控制算法、CS 基础算法、数学优化算法，所有事实性断言均以 GitHub 仓库/issue/官方文档为准。
> 子报告：[控制算法](./control-algorithms.md) · [CS 算法](./cs-algorithms.md) · [优化算法](./optimization-algorithms.md)

## 一、跨领域的三个元结论

### 1. 最快的实现已经存在，落差集中在「冻结层」
三个领域呈现出完全相同的模式：某处已有验证过的最优技术（Rust driftsort、SwissTable、x86-simd-sort、PRIMA、acados 热启动），但**主流用户实际在用的实现停留在教科书年代**——因为主流入口是 C ABI 冻结的标准库、教学优先的 Python 库、Fortran 77 转写代码。**最大机会是「移植已验证技术」，而非发明新算法。**

### 2. 线性/成熟族 vs 非线性/前沿族的断裂
- **已成熟**：LQR/MPC（acados/python-control）、LP/QP（HiGHS/OSQP）、排序/哈希（Rust/C++ 生态）→ 机会在细节优化。
- **无成熟库**：ADRC、滑模、MRAC、ILC、通用 SA/Tabu/LNS → 机会是**建库**，竞争者是 35★ 以下的停更项目，成为事实标准的门槛极低。

### 3. 反向结论同样值钱：别优化死项目
GLPK、lp_solve、qpSWIFT、liblbfgs、simanneal、scikit-fuzzy、filterpy（作者已明示教学优先）均已事实冻结。给它们写 PR 不如做**迁移引导**（文档、shim 层、benchmark 对比）。

## 二、Top 优化机会总榜（影响力 × 可行性）

| 排名 | 机会 | 领域 | 杠杆 | 评分 | 一句话理由 |
|------|------|------|------|------|-----------|
| 1 | **scipy/NLopt 无导数族换 PRIMA 后端** | 优化 | 迁移已验证技术 | 5×5 | 修掉无限循环/段错误等成体系 bug，接口等价，SciPy 官方已在讨论 |
| 2 | **libc++ `std::regex` 重写** | CS | 移植已验证技术 | 5×4 | 比可用实现慢 10–30×，llvm issue #60991 维护者公开欢迎补丁 |
| 3 | **pocketfft 1D FFT 向量化** | CS | SIMD | 5×4 | NumPy/SciPy 默认引擎，1D 路径零 SIMD，AVX 化可追平 FFTW |
| 4 | **CPython `fastsearch` SIMD 化** | CS | SIMD | 5×3 | glibc 2018 年已验证路线（5–6.6×），几乎可直接移植 |
| 5 | **filterpy UKF/EnKF 向量化** | 控制 | 向量化 | 4×4 | 3870★ 教学库，逐 sigma 点 Python 循环，改造局部收益明确 |
| 6 | **pymoo 热点算子向量化** | 优化 | 向量化 | 4×5 | 官方 issue #774 已逐文件列好改造清单，不改 API |
| 7 | **轻量线性 MPC 库（新建）** | 控制 | 生态位空缺 | 4×4 | do-mpc 太重 / CVXPY 太慢之间的空档，技术路线（编译一次+OSQP 热启动）完全现成 |
| 8 | **C++ `std::sort` pdqsort/SIMD 特化** | CS | SIMD | 4×3 | 不受 ABI 冻结约束，x86-simd-sort 已证 10–17× 加速 |
| 9 | **ADRC 标准化库（新建）** | 控制 | 生态位空缺 | 4×4 | 工程需求极大，现有最高实现 35★ 且停在 2015 年 |
| 10 | **HiGHS HiPO 内点 Python 发行链打通** | 优化 | 工程打通 | 5×3 | 缓解官方承认的"比 Gurobi 慢 60–100×"痛点，卡在构建链 |
| 11 | **pycma 内置种群级并行评估** | 优化 | 并行 | 3×5 | CMA-ES 天然并行，pymoo/cmaes 已证明低风险 |
| 12 | **Edlib 短字符串路径** | CS | 单点修复 | 3×5 | issue #144 开放 5 年，生物信息百万次级调用，可行性最高 |
| 13 | **python-control `hinfsyn` 热启动/早停** | 控制 | 热启动 | 3×4 | issue #152 长期 open，项目活跃 PR 落地概率高 |
| 14 | **OSQP GPU batched 求解** | 优化 | GPU | 4×3 | cuOSQP 已证两个数量级加速，官方路线图点名方向 |
| 15 | **通用 SA/Tabu/LNS 开源求解器（新建）** | 优化 | 生态位空缺 | 5×2 | 结构性空白，闭源领先数量级，但工程量最大——长期方向 |

## 三、给不同投入者的建议路径

**想要快速产出 PR（1–2 周）**：#6 pymoo 向量化、#12 Edlib、#13 hinfsyn —— issue 已开、维护者欢迎、改动局部。

**想做有影响力的中型项目（1–3 个月）**：#3 pocketfft 向量化、#5 filterpy 改造、#1 PRIMA 接入 scipy。

**想建一个能成为事实标准的库（3 个月以上）**：#7 轻量 MPC、#9 ADRC —— 都是「空白生态位」，先发者几乎没有竞争对手。

**想啃硬骨头换巨大影响力**：#2 libc++ regex 重写、#15 通用启发式求解器。

## 四、文件索引

| 文件 | 内容 |
|------|------|
| [first-principles-framework.md](./first-principles-framework.md) | 五资源 × 六杠杆的统一分析框架 |
| [control-algorithms.md](./control-algorithms.md) | 15+ 仓库的控制算法调研（含实时 GitHub 数据） |
| [cs-algorithms.md](./cs-algorithms.md) | 5 大算法族 + 教育/生产落差专题 + 15 项排序表 |
| [optimization-algorithms.md](./optimization-algorithms.md) | 6 大求解器族 + Mittelmann benchmark 数据 |
