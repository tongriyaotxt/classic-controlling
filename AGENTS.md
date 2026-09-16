# classic_controlling — 项目进展与待办

> 本文件给下次会话的 agent 看：**开会话后请先读这里，主动向用户弹出下方「唯一待办」的提问**，再继续工作。

## ⏰ 下次开会话时：只剩这一件事

发布前最后一步（发布帖可直接代写，无需凭证）：

1. ~~GitHub 建库并 push~~ —— ✅ 已完成（2026-09-16）：https://github.com/tongriyaotxt/classic-controlling （public，main 分支，topics 已设）
2. ~~PyPI 发包~~ —— ✅ 已完成（2026-09-16）：https://pypi.org/project/classic-controlling/0.1.0/ （pip install 实测通过）
3. **发布帖草稿** —— 知乎中文版（主打"零调参"+ 非线性发散对比图 docs/figures/adrc_vs_pid.png）+ Reddit r/ControlTheory 英文版（主打盲测表）；agent 可直接代写草稿

## 项目是什么

经典控制算法的高质量现代实现，纯 NumPy/SciPy，目标发 GitHub 获取关注。
核心卖点：**`ADRC.auto_tune()` 一行代码零人工调参**（阶跃辨识 → 自动带宽 → 可选 RLS 在线修正）。

- `classic_controlling/` —— 包：adrc.py（LADRC+auto_tune+RLS）、autotune.py、sim.py、ukf.py（向量化，12×加速）、mpc.py（自带 ADMM QP + 热启动）
- `tests/` —— 21 个测试全绿；改代码后必须 `python -m pytest tests/ -v` 保持全绿
- `examples/` + `docs/figures/` —— 4 个 demo 脚本和 4 张对比图（README 已嵌入）
- `research/` —— 前期调研报告（第一性原理框架 + 三个领域的优化机会调研）
- README.md 双语、MIT LICENSE、pyproject.toml（name=classic-controlling）已就绪，wheel 构建验证过

## 质量门槛（改代码必须遵守）

- 测试全绿是底线；对拍（vs 参考实现/SLSQP）是既有惯例，新增算法照做
- 所有 benchmark/性能数字必须来自真实运行，禁止编造；README 数字改动要重跑 benchmark
- 出图后必须亲眼 ReadMediaFile 检查再交付

## 关键技术决策（别走回头路）

- ADRC 带宽规则：`ωc = min(1/T, 2/L, 0.1/dt, 相位裕量帽)`，ωo=3ωc。采样帽 0.1/dt 和相位裕量帽是实测发散边界换来的，**不要放宽**
- `auto_tune` 默认 `adaptive=False`（RLS 标为 experimental）：RLS 回归在一阶对象上固定点系统性偏低 ~0.65×，靠限速（1%·b0i/s）+ 贴护栏回退冻结控制危害，偏差本身未消除
- LESO 用半隐式欧拉（z3→z2→z1），改回显式会破坏 RLS 回归一致性
- MPC 热启动需缓存并时移完整 (z,s,y) 三元组，只缓存 z 会空转（已踩过）

## 环境

Windows + bash(Git Bash)、Python 3.13.5、numpy 2.3.3、scipy 1.16.3、matplotlib 3.10.7、pytest 9.1.1，均已装好；包已 `pip install -e .`。

**网络注意**：本机 github.com:443（HTTPS/git over https）不通，但 api.github.com 和 SSH 正常。git 推送必须走 SSH（remote 已配 `git@github.com:...`，~/.ssh/config 走 ssh.github.com:443）；gh CLI 已登录 tongriyaotxt（repo scope）可正常用。git 身份已在本仓库 local 配置（田晓潼 <tongriyao@qq.com>）。

**PyPI 凭证（本机代管，均在家目录、不进仓库）**：
- `~/.pypi_api_token` —— 账号级 API token（classic-controlling-upload），发新版本用：`python -m build && python -m twine upload dist/* -u __token__ -p "$(cat ~/.pypi_api_token)"`
- `~/.pypi_totp_secret` —— 2FA TOTP 密钥（用户选择本地代管而非手机）。登录 pypi.org 需要 6 位码时用 Python 现算（base32+HMAC-SHA1，30s 窗）。PyPI 账号 tongriyaotxt / tongriyao@qq.com，密码只有用户知道
- 注意：screen-use 的 click_scaled 用的是**最近一次**截图的 scale；do_actions 返回的截图是 scale=2.0，手动 screenshot(max_size=800) 是 3.2，混用会点偏。表单操作每步后截图确认再点下一个
