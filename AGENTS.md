# classic_controlling — 项目进展与待办

> 本文件给下次会话的 agent 看：**开会话后请先读这里，主动向用户弹出下方「唯一待办」的提问**，再继续工作。

## ⏰ 下次开会话时：只剩这一件事

发布前最后一步（发布帖可直接代写，无需凭证）：

1. ~~GitHub 建库并 push~~ —— ✅ 已完成（2026-09-16）：https://github.com/tongriyaotxt/classic-controlling （public，main 分支，topics 已设）
2. ~~PyPI 发包~~ —— ✅ 已完成（0.1.0 + 0.2.0，2026-09-16）：https://pypi.org/project/classic-controlling/ （pip install 实测通过）
3. **发布帖** —— 知乎中文版：✅ 已发布（2026-09-16，GUI 自动化填入+用户确认发布，草稿存档 docs/posts/zhihu_post.md）；Reddit r/ControlTheory 英文版：✍️ 草稿已写（docs/posts/reddit_post.md，署名 tongriyao (田晓潼)，主打盲测表），**用户没有 Reddit 账号且本机网络不通 reddit.com（curl 超时），无法代发，搁置中**

## 项目是什么

经典控制算法的高质量现代实现，纯 NumPy/SciPy，目标发 GitHub 获取关注。
核心卖点：**`ADRC.auto_tune()` 一行代码零人工调参**（阶跃辨识 → 自动带宽 → 可选 RLS 在线修正）。

- `classic_controlling/` —— 包：adrc.py（LADRC+auto_tune+RLS）、pid.py（工业 PID：微分先行/前馈/反算抗饱和/设定值权重 + CascadePID 串级 + Z-N/IMC-PI 整定）、autotune.py、sim.py、ukf.py（向量化，12×加速）、mpc.py（自带 ADMM QP + 热启动）
- `tests/` —— 32 个测试全绿；改代码后必须 `python -m pytest tests/ -v` 保持全绿
- `examples/` + `docs/figures/` —— 5 个 demo 脚本和 5 张对比图（README 已嵌入；demo_pid_features.py 为 PID 四大件消融对比）
- `research/` —— 前期调研报告（第一性原理框架 + 三个领域的优化机会调研）
- README.md 双语、MIT LICENSE、pyproject.toml（name=classic-controlling）已就绪，wheel 构建验证过
- 当前版本 **0.2.0**（✅ 2026-09-16 已发 PyPI + 已 push；新增工业 PID 家族，pip install 实测通过）

## 质量门槛（改代码必须遵守）

- 测试全绿是底线；对拍（vs 参考实现/SLSQP）是既有惯例，新增算法照做
- 所有 benchmark/性能数字必须来自真实运行，禁止编造；README 数字改动要重跑 benchmark
- 出图后必须亲眼 ReadMediaFile 检查再交付

## 关键技术决策（别走回头路）

- ADRC 带宽规则：`ωc = min(1/T, 2/L, 0.1/dt, 相位裕量帽)`，ωo=3ωc。采样帽 0.1/dt 和相位裕量帽是实测发散边界换来的，**不要放宽**
- `auto_tune` 默认 `adaptive=False`（RLS 标为 experimental）：RLS 回归在一阶对象上固定点系统性偏低 ~0.65×，靠限速（1%·b0i/s）+ 贴护栏回退冻结控制危害，偏差本身未消除
- LESO 用半隐式欧拉（z3→z2→z1），改回显式会破坏 RLS 回归一致性
- PID 积分用「先试投后反算」（i_trial → clip → 反算回收 kaw=1/Ti）；examples/_common.py 的 PID 只是包内 PID 的别名，改 pid.py 后必须重跑 demo_adrc_vs_pid.py 核对对照组指标无漂移
- 新增功能测试惯例：同参数只切一个开关的对照仿真（消融）断言该功能有效；整定规则与经典公式逐数值对拍
- MPC 热启动需缓存并时移完整 (z,s,y) 三元组，只缓存 z 会空转（已踩过）

## 发布数据存档

所有发布相关账号/链接/凭证位置/发版命令已汇总到 **`docs/PUBLISHING.md`**（不含密钥本体）。发新版本、写 Reddit 帖先读它。

## 环境

Windows + bash(Git Bash)、Python 3.13.5、numpy 2.3.3、scipy 1.16.3、matplotlib 3.10.7、pytest 9.1.1，均已装好；包已 `pip install -e .`。

**网络注意**：本机 github.com:443（HTTPS/git over https）不通，但 api.github.com 和 SSH 正常。git 推送必须走 SSH（remote 已配 `git@github.com:...`，~/.ssh/config 走 ssh.github.com:443）；gh CLI 已登录 tongriyaotxt（repo scope）可正常用。git 身份已在本仓库 local 配置（田晓潼 <tongriyao@qq.com>）。

**PyPI 凭证**：细节（token/TOTP 路径、账号、发版命令）见本地 `docs/PUBLISHING.md`（已 gitignore，不进公开仓库）。发版：`python -m build && twine upload` 用 `__token__` + 家目录 token 文件。
- 注意：screen-use 的 click_scaled 用的是**最近一次**截图的 scale；do_actions 返回的截图是 scale=2.0，手动 screenshot(max_size=800) 是 3.2，混用会点偏。表单操作每步后截图确认再点下一个
