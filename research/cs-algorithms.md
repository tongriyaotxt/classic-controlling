# 传统计算机科学基础算法：开源实现现状与优化机会调研

> 调研日期：2026-09-15
> 方法：以一手来源为准（GitHub 仓库、语言标准库源码、官方文档、经典/近期论文），辅以 benchmark 与 issue 追踪。
> 所有事实性断言均附来源链接。

---

## 1. 比较类排序 / 选择

### 1.1 主流与最优实现

| 实现 | 语言/归属 | 技术特点 | 来源 |
|---|---|---|---|
| `pdqsort` (pattern-defeating quicksort) | C++，Orson Peters | 块分区（BlockQuicksort）消除分支误预测；对有序/重复数据自适应达到 O(n)；坏枢轴累计后回退 heapsort | [github.com/orlp/pdqsort](https://github.com/orlp/pdqsort)；论文 [arXiv:2106.05123](https://arxiv.org/pdf/2106.05123.pdf) |
| Go 标准库 `sort`/`slices.Sort` | Go 1.19+ | 已切换到 pdqsort（但**禁用了 BlockQuicksort 优化**，因为"在 Go 中表现差"） | [golang/go#50154](https://github.com/golang/go/issues/50154) |
| Rust 标准库 | Rust | `sort_unstable` = **ipnsort**（pdqsort 的演化版）；`sort` = **driftsort**（glidesort 的变种，powersort 合并策略 + 稳定 quicksort，全分支消除，对低基数数据 O(n log k)） | [orlp.net](https://orlp.net/)；[power-sort.github.io](https://power-sort.github.io/) |
| C++ `std::sort` | libstdc++/libc++ | 仍是 1990 年代 SGI 风格的 **introsort**，无分支消除、无 SIMD | [hftuniversity 分析](https://hftuniversity.com/post/the-c-standard-library-has-been-walking-itself-back-for-fifteen-years-and-the-receipts-are-public) |
| Java `Arrays.sort` | JDK | 基本类型用 Yaroslavskiy **双枢轴 quicksort**，对象数组用 TimSort | [JDK TimSort 源码注释](https://www.cnblogs.com/helloz/p/11610660.html) |
| CPython `list.sort` | C | TimSort（2002 年 Tim Peters 原版血脉），最小化比较次数、利用天然 run + galloping | [listsort.txt 解读](https://liwt31.github.io/2018/10/27/timsort/) |
| Intel `x86-simd-sort` | C++ header-only | AVX-512/AVX2 的 quicksort/qselect/argsort/partial_sort；被 **NumPy（10–17× 加速）与 PyTorch（torch.sort 大数组最高 10×）**采纳 | [Phoronix v1.0](https://www.phoronix.com/news/x86-simd-sort-1.0)、[v6.0/PyTorch](http://www.linuxeden.com/a/142092) |
| `ips4o` / `ips2ra` | C++ | 就地并行 superscalar samplesort / radix sort；在 21 种实现 × 6 种类型 × 10 种分布的大规模对比中总体最优，比最优对手快 1.2–1.8× | [arXiv:2009.13569](https://arxiv.org/pdf/2009.13569)、[github.com/ips4o/ips2ra](https://github.com/ips4o/ips2ra) |
| `ska_sort` | C++ header-only | 通用化 radix sort | [probablydance 对比](https://probablydance.com/2017/01/17/faster-sorting-algorithm-part-2/) |
| AVX-512 向量化 quicksort 研究 | C/asm | 用 VCOMPRESS 系列指令做 SIMD 分区 + Bitonic 排序网络处理小块；比 GNU STL 快 4×、比 Intel IPP 快 1.4× | [Bramas, arXiv:1704.08579](https://arxiv.org/html/1704.08579v2) |
| `blqsort` | C/C++ header | 全分支消除 quicksort，比 std::sort 快约 30%、比 pdqsort 快约 27%（单线程，2025 新发布） | [80aj 报道](https://www.80aj.com/2026/06/05/blqsort-sort-performance/) |

### 1.2 优化机会

1. **C++ 标准库排序未用 SIMD / 分支消除**（高影响）。`std::sort` 仍是 introsort；而 AVX-512 研究已证明可快 4×，Intel x86-simd-sort 已被 NumPy/PyTorch 采纳。C++ 标准不冻结排序算法实现（不像 unordered_map 受 ABI 约束），这是**直接可操作**的方向：向 libstdc++/libc++ 贡献对基本类型特化的 pdqsort/branchless 路径。证据：[arXiv:1704.08579](https://arxiv.org/html/1704.08579v2)、[hftuniversity](https://hftuniversity.com/post/the-c-standard-library-has-been-walking-itself-back-for-fifteen-years-and-the-receipts-are-public)。
2. **Go 的 pdqsort 禁用了块分区**——ByteDance 团队称"BlockQuicksort 在 Go 中表现差"而未复现 C++/Rust 的 15–20% 提升，存在重新评估或用 Go 汇编/编译器改进找回该收益的空间。证据：[golang/go#50154](https://github.com/golang/go/issues/50154)。
3. **稳定排序的现代化未普及**：driftsort/glidesort（powersort 合并 + 稳定 quicksort + 全分支消除 + 双向交错归并）已在 Rust 落地并报告随机数据 3× 于旧 Timsort 版 Rust sort，但 **CPython/Java 的 TimSort 仍是 2002/2010 年代设计**，且未用 powersort 的近最优合并策略。证据：[orlp/glidesort](https://github.com/orlp/glidesort)、[power-sort.github.io](https://power-sort.github.io/)。
4. **AVX-512 排序研究的并行化空白**：Bramas 论文明确将 task-based 并行列为 future work。证据：同上论文结论段。

---

## 2. 查找结构

### 2.1 二分查找

| 实现 | 特点 | 来源 |
|---|---|---|
| `std::lower_bound` / `std::binary_search` | 教科书有分支二分 | — |
| Khuong & Morin 数组布局研究 | **无分支 + 显式预取**的 Eytzinger（堆序 BFS 布局）搜索在大数组上"大幅超过所有二分查找变体"；小数组上分支消除二分最优 | [arXiv:1509.05053](https://arxiv.org/pdf/1509.05053.pdf) |
| SOSD benchmark（学习索引） | 分支消除二分 + Eytzinger 是 learned index 也难以在常数空间击败的基线 | [arXiv:2107.09480](https://arxiv.org/pdf/2107.09480v3) |
| 批量二分（batching） | 批处理 + 预取可把查找吞吐量再提数倍 | [curiouscoding.nl 实验](https://curiouscoding.nl/posts/binsearch/) |

**优化机会**：各语言标准库的 `binary_search`/`lower_bound` 仍是有分支版本，未提供无分支/Eytzinger/批量 API；Rust 生态有 `eytzinger` crate 但主流语言标准库缺位。证据：[Khuong-Morin 论文](https://arxiv.org/pdf/1509.05053.pdf)、[eytzinger-interpolation crate](https://github.com/sanity/eytzinger-interpolation)。

### 2.2 哈希表

| 实现 | 特点 | 来源 |
|---|---|---|
| **Abseil Swiss Table**（`absl::flat_hash_map`） | 开放寻址 + 16 槽一组 + 每槽 1 字节控制位，用 **SSE2 并行比对**；比 `std::unordered_map` 快约 3× | [CppCon 2017 设计记录（hftuniversity 引用）](https://hftuniversity.com/post/all-further-discussion-will-assume-chaining-how-a-2003-sentence-locked-c-out-of-fast-hash-maps) |
| **Folly F14** | 14 路探测 + SIMD 过滤 + prehash token；`F14ValueMap` 内存效率约为 dense_hash_map 的 2× | [folly/F14.md](https://github.com/facebook/folly/blob/main/folly/container/F14.md) |
| **Rust `HashMap`**（hashbrown） | Swiss Table 的 Rust 移植 + `foldhash`，是语言默认实现 | [orlp.net](https://orlp.net/) |
| **Go 内置 map**（1.24+） | 2025 年起替换为 Swiss Table：8 槽组 + 控制字 + 大表分裂为 ≤1024 槽的表避免全量 rehash；微基准 map 操作快至 30–35%，全程序 geomean 约 1.5% | [Go 1.24 说明](https://www.ayokoding.com/en/learn/software-engineering/programming-languages/golang/release-highlights/go-1-24/)、[VictoriaMetrics 解析](https://victoriametrics.com/blog/go-swiss-table-map/) |
| `boost::unordered_flat_map`、`ankerl::unordered_dense`、`phmap` | 各版本开放寻址，均数倍快于 `std::unordered_map` | [martin.ankerl.com 基准](https://martin.ankerl.com/2019/04/01/hashmap-benchmarks-01-overview/) |
| `std::unordered_map` | **被 C++ 规范锁死为链式**：桶/迭代器稳定性要求实际禁止开放寻址，委员会无法修复 | [hftuniversity 两篇文章](https://hftuniversity.com/post/the-c-standard-library-has-been-walking-itself-back-for-fifteen-years-and-the-receipts-are-public) |

**优化机会**：① Go Swiss Table 仍有已知回归——冷缓存/稀疏大 map 场景（目录间接层带来额外 cache miss，Prometheus 实测回归，追踪于 golang/go#70835），是可操作的具体改进点（[来源](https://blog.gaborkoos.com/posts/2026-07-24-Golang-Maps-How-Swiss-Tables-Replaced-the-Old-Bucket-Design/)）。② 高负载因子下的行为差异极大，benchmark 方法论混乱，仍缺公认统一基准（[ankerl 评论](https://www.libhunt.com/compare/ktprime-emhash-vs-robin-hood-hashing)）。

### 2.3 B 树 / B+ 树

| 实现 | 特点 | 来源 |
|---|---|---|
| `absl::btree_map/set` | 节点默认 256 字节贴合 cache line；叶节点不存子指针；`std::set<int32>` 每值约 40 字节 vs btree 约 5.1 字节 | [abseil.io 设计文档](https://abseil.io/about/design/btree) |
| Rust `BTreeMap` | 标准库即 B 树 | 同上 hftuniversity 引用 |
| `std::map`/`std::set` | 红黑树，每节点一次堆分配 + 指针追逐；STL 之父 Stepanov 承认"今天重来会选内存内 B\* 树" | [Stepanov 访谈引用](https://gist.github.com/justinmeiners/57f38bddae9029db3c6401fae113bd7c) |

**优化机会**：C++ 委员会 C++23 只加了 `std::flat_map`（有序 vector 适配器）而未动 `std::map`；纯 Swift 实现的 B 树比最优红黑树快 300–400% 且省内存——其他语言（Python `SortedDict`、Java `TreeMap`）仍是红黑树，替换空间明确。证据：[Swift 论坛 lorentey 回复](https://forums.swift.org/t/introducing-swift-collections/47169)、[hftuniversity](https://hftuniversity.com/post/the-c-standard-library-has-been-walking-itself-back-for-fifteen-years-and-the-receipts-are-public)。

### 2.4 布隆过滤器

| 实现 | 特点 | 来源 |
|---|---|---|
| Blocked Bloom（Apache Impala 版） | AVX2 内在函数手写，单次 cache line 访问；查询最快但内存更大 | [Graf & Lemire, Xor Filters 论文](https://arxiv.org/pdf/1912.08258) |
| **Xor filter / Xor+** | 恰好 3 次可并行的内存访问、无分支；比 Bloom 快且省约 15% 内存；存储开销距理论下界 23%（Bloom 为 44%）；构建慢约 2× 且不可删除 | 同上 |
| **Binary fuse filter** | 存储距下界仅 13%（可压到 8%），构建比 xor filter 快 2× 以上，查询不降速 | [arXiv:2201.01174](https://arxiv.org/pdf/2201.01174v1.pdf) |
| Cuckoo filter | 支持删除；桶数非 2 的幂时存在可证明的假阴性缺陷（95% 负载下最高 10% 假阴性） | [maltsev.space](https://maltsev.space/blog/010-cuckoo-filters) |

**优化机会**：大量生产系统（数据库、CDN、缓存层）仍默认教科书 Bloom filter；xor/binary fuse 的论文指出"构建并行化、批量查询"是明确的 future work。证据：[xor filter 论文结论](https://arxiv.org/pdf/1912.08258)。

### 2.5 跳表

跳表在生产中的代表是 Redis zset 与 Java `ConcurrentSkipListMap`。现状：在内存随机访问场景下跳表的 cache 行为全面劣于 B 树（见 2.3 节 B 树 vs 红黑树的同源论证），其价值集中在**无锁并发实现简单**。优化空间相对明确但影响面窄（Redis 生态、并发有序结构），本调研未找到新的大宗证据，列为低优先级观察项。

---

## 3. 字符串算法

### 3.1 单模式匹配（KMP / Boyer-Moore 家族）

| 实现 | 特点 | 来源 |
|---|---|---|
| glibc `strstr`/`memmem` | Two-Way（Crochemore-Perrin，线性最坏 + O(1) 空间）为基座；2018–2019 年 Arm 工程师 Wilco Dijkstra 加入**改进版 Horspool（字符对哈希跳表 + 自适应过滤）**与 strnlen/memchr SIMD 预扫，AArch64 上 memmem 快 6.6×、strstr 快至 5.8× | [glibc memmem.c 源码注释](https://codebrowser.dev/glibc/glibc/string/memmem.c.html)、[Phoronix 报道](https://www.phoronix.com/news/Glibc-Arm-Optimizations-Mem-Str)、[glibc-cvs 提交](https://sourceware.org/pipermail/glibc-cvs/2019q3/067629.html) |
| CPython 子串查找（`in` 运算符） | `fastsearch.h`：Boyer-Moore/Horspool 混合 + Two-Way 兜底；**纯标量，无 SIMD** | [CPython 源码注释（DMOJ 引用）](https://dmoj.ca/problem/bf4) |
| LLVM libc | 亦用 Two-Way + BM 坏字符表 | [reviews.llvm.org/D86162](https://reviews.llvm.org/D86162) |

**优化机会**：CPython `fastsearch` 是标量实现——glibc 的 SIMD 化路径（2018 年就已在 glibc 落地）从未被移植到 CPython；Python 3.10 改进的只是最坏复杂度。证据见上表。

### 3.2 多模式匹配（Aho-Corasick）

| 实现 | 特点 | 来源 |
|---|---|---|
| Rust `aho-corasick` crate | **Teddy** SIMD 预过滤（pshufb nibble 查表，Slim/Fat Teddy）+ NFA/DFA 自动选择；2023 年补齐 aarch64 SIMD，M1/M2 上提速 2–10× | [github.com/BurntSushi/aho-corasick](https://github.com/BurntSushi/aho-corasick)、[Rust 日报报道](https://rustcc.cn/article?id=4d6419c6-b526-4b09-9dfe-bbd3bd5a5821) |
| Hyperscan | Teddy 的发源地；SIMD 多模式正则/字符串匹配的事实标准 | [branchfree.org 论文评述](https://branchfree.org/2019/02/28/paper-hyperscan-a-fast-multi-pattern-regex-matcher-for-modern-cpus/) |
| Go 生态 | 出现声称 6 GB/s 单核、零分配的 AC 库，但标准库没有 AC | [dev.to 文章](https://dev.to/kolkov/aho-corasick-in-go-multi-pattern-string-matching-at-6-gbs-with-zero-allocations-2jog) |

**优化机会**：Teddy 依赖 pshufb 类指令，**AVX-512 版 Teddy / 大模式集场景（超过预过滤能力上限时回落纯自动机）仍是空白**；aho-corasick 文档明确说模式数变大后预过滤即被弃用。证据：[aho-corasick 文档](https://teaclave.apache.org/api-docs/client-sdk-rust/aho_corasick/index.html)。

### 3.3 编辑距离

| 实现 | 特点 | 来源 |
|---|---|---|
| **Edlib** | Myers 位向量 + Ukkonen 带状算法；公认最快精确实现，比 SeqAn 快 2.5–100×、比 Parasail 快 12–1000× | [Edlib 论文（PMC5408825）](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5408825/) |
| WFA（wavefront）/ eWFA-GPU | GPU 版比 Edlib 快至 265× | [Aguado-Puig 论文](https://upcommons.upc.edu/bitstreams/fccb6a47-6343-4850-a961-8e16f99e5bac/download) |
| 短字符串场景 | Edlib 在 30 字节级短串上反而**慢于**朴素的 `Levenshtein` 库（0.43µs vs 1.88µs）——issue 建议引入 Navarro 的 JEA06 技术，未解决 | [edlib issue #144](https://github.com/Martinsos/edlib/issues/144) |

**优化机会**：① Edlib 短串路径的开销问题（issue #144 自 2020 年开放）；② 位向量算法的 AVX2/AVX-512 宽化（GPU 已证明数量级收益，CPU 向量化仍有空间）。证据见上表。

### 3.4 后缀数组 / 自动机

| 实现 | 特点 | 来源 |
|---|---|---|
| **libsais** | SA-IS 的工程极致：每桶双指针分离诱导方向、MSB 标记完全分支消除的排名、按字母表大小分 4 档实现；支持 OpenMP 多核 | [github.com/IlyaGrebnov/libsais](https://github.com/IlyaGrebnov/libsais) |
| libdivsufsort | 经典实现，实践中比纯 SAIS 快（非递归排序初始后缀） | [arXiv:1710.01896](https://arxiv.org/pdf/1710.01896) |
| libcubwt | GPU 构建后缀数组/BWT | [libsais 文档推荐](https://gitcode.com/gh_mirrors/li/libsais/overview) |
| 分布式 fDCX | 2 个节点起即超过单机 48 核 libsais | [LIPIcs.ESA.2025.47](https://drops.dagstuhl.de/storage/00lipics/lipics-vol351-esa2025/LIPIcs.ESA.2025.47/LIPIcs.ESA.2025.47.pdf) |

**优化机会**：libsais 已对软件预取/内存速度高度敏感（作者自述）；单机 SIMD 化（诱导排序的散射/聚集天然难向量化）与 Rust 生态缺成熟封装（[curiouscoding 盘点](https://curiouscoding.nl/posts/suffix-array-crates/)）是可见方向。

### 3.5 正则引擎

| 实现 | 特点 | 来源 |
|---|---|---|
| **Hyperscan**（Intel） | 混合自动机分解 + SIMD；2017 年多引擎横评中总耗时约为第 2 名的 1/3 | [rust-leipzig 横评](https://rust-leipzig.github.io/regex/2017/03/28/comparison-of-regex-engines/)、[branchfree.org](https://branchfree.org/2019/02/28/paper-hyperscan-a-fast-multi-pattern-regex-matcher-for-modern-cpus/) |
| RE2（Google） | 自动机线性时间保证，适合安全场景；RE2::Set 多模式 | [Velox issue #9823](https://github.com/facebookincubator/velox/issues/9823) |
| Rust `regex` crate | SIMD 预过滤 + lazy DFA，横评中与 PCRE2-JIT 并列第二档 | [rust-leipzig 横评](https://rust-leipzig.github.io/regex/2017/03/28/comparison-of-regex-engines/) |
| **`std::regex`** | **公认灾难**：libstdc++ 版比 Boost 慢 10× 以上，`optimize` 标志无效；libc++ 版比 libstdc++ 再慢约 10×；CTRE 编译期实现快约 30×。根因：模板 ABI 冻结导致实现无法更换 | [lmi 邮件列表实测](https://lists.nongnu.org/archive/html/lmi/2016-07/msg00010.html)、[llvm-project issue #60991](https://github.com/llvm/llvm-project/issues/60991)、[StackOverflow 分析](https://stackoverflow.com/questions/70583395/why-is-stdregex-notoriously-much-slower-than-other-regular-expression-librarie) |
| Go `regexp` | RE2 风格线性时间 NFA，**无 SIMD 预过滤**；第三方 pure-Go 引擎声称快 3–3000× | [golangnews](https://golangnews.com/stories/4922-coregex-pure-go-production-grade-regex-engine.-up-to-3-3000x-faster-than-stdlib.) |
| CPython `re` | 回溯引擎（非线性时间保证），无 SIMD；500+ 模式场景比 re2.Set 慢约 400× | [geekmonkey 基准](https://geekmonkey.org/regular-expression-matching-at-scale-with-hyperscan/) |

**优化机会**（本族内最大）：`std::regex` 的两个主流实现都是"教科书级慢"且 ABI 锁定——短期机会在 libc++（issue 自 2023 年开放、维护者表态"欢迎补丁"）；Go regexp 缺 SIMD literal 预过滤（Rust regex 的 memchr/Teddy 路径可直接借鉴）；Hyperscan 默认不支持 Unicode 且 x86 专用（ARM 需 vectorscan fork）。证据见上表。

---

## 4. 图算法

### 4.1 最短路（Dijkstra / A* / Bellman-Ford）

| 实现 | 特点 | 来源 |
|---|---|---|
| Boost Graph Library（BGL） | 最高性能的**单线程** CPU 图库之一，Dijkstra + 堆 | [Gunrock 论文对比](https://arxiv.org/pdf/1501.05387) |
| Ligra（CMU） | 多核共享内存框架，SSSP 用 Bellman-Ford 式标签传播 + 方向优化（push/pull 切换） | [GraphBLAST 论文 Table 3/9](https://par.nsf.gov/servlets/purl/10294352) |
| Gunrock / GraphBLAST（GPU） | 对 BGL/PowerGraph 平均 6–337× 加速；GraphBLAST 比 SuiteSparse GraphBLAS 顺序 CPU 快 geomean 36× | 同上两篇 |
| SuiteSparse:GraphBLAS | 稀疏线性代数表达图算法（Dijkstra→min-plus 半环 SpMV），首个多线程 CPU GraphBLAS | [Davis TOMS 论文](https://people.engr.tamu.edu/davis/GraphBLAS_files/toms_graphblas.pdf) |
| NetworkX | 纯 Python，仅适合教学/小图（此为本族"教育实现"的典型代表） | 常识性定位，见其官方定位文档 |

**优化机会**：① **主流语言生态（Python NetworkX、igraph 默认路径、BGL）的 Dijkstra 均为单线程**，而 delta-stepping、方向优化在学术界与 GPU 库中已成熟，移植到通用 CPU 库的 gap 明显（Gunrock 对 BGL 的数量级差距即证据）。② GraphBLAS 表达下 Dijkstra 复杂度退化为 O(n²) 级 SpMV 迭代（见 FOSDEM slides 的复杂度表），语义层与算法层之间有优化空间。证据：[GraphBLAS slides](https://archive.fosdem.org/2020/schedule/event/graphblas/attachments/slides/4132/export/events/attachments/graphblas/slides/4132/graphblas_introduction.pdf)。

### 4.2 SCC / 网络流 / 并查集

| 主题 | 现状 | 来源 |
|---|---|---|
| Tarjan SCC | 教科书递归实现仍为主流；并行 SCC 主要存在于 Ligra/Galois 研究库 | [GraphBLAST 论文](https://par.nsf.gov/servlets/purl/10294352) |
| 最大流 | 实践最快是 **Push-Relabel + gap 启发式**（BGL 的 max-flow 即此变体）；Dinic 在小而稀疏/网格图上更快；计算机视觉用 Boykov-Kolmogorov；**多数开源实现（含各 OJ 模板库）仍是裸 Dinic/Edmonds-Karp**；2020 年代近线性时间最大流理论结果尚无生产实现 | [lbenicio 对比](https://blog.lbenicio.dev/post/2021/02/14/the-algorithmics-of-network-flow-dinics-algorithm-with-scaling-vs.-push-relabel-with-gap-heuristics/)、[mysimulator](https://www.mysimulator.uk/articles/dinics-algorithm/) |
| Hopcroft-Karp | "实践中最快的二分匹配"，但大量项目仍用 BFS 增广路（Edmonds-Karp 退化版） | [Codeforces 讨论](https://codeforces.com/blog/entry/118098) |
| 并查集 | 按秩合并 + 路径分裂已接近理论极限；优化前沿在无锁/批量并行连通性（Ligra 的 CC 即并查集变体），无大宗单线程优化空间证据 | [GraphBLAST 论文 CC 对比](https://par.nsf.gov/servlets/purl/10294352) |

---

## 5. 数值基础

### 5.1 FFT

| 实现 | 特点 | 来源 |
|---|---|---|
| **FFTW 3.3.x** | codelet 自动生成的规划器架构；支持 SSE/AVX/AVX2/NEON；官方基准页自 2005 年未更新 | [fftw.org](https://www.fftw.org/) |
| Intel MKL（闭源基准点） | 在第三方 3D FFT 基准中明显快于 FFTW（个别尺寸 4× 以上） | [gemmi benchmarking-fft](https://github.com/project-gemmi/benchmarking-fft) |
| **pocketfft** | 单头文件 C++，NumPy 1.17 起与 SciPy 1.4 起的默认 FFT；**1D 变换不做向量化**（仅多维/批量路径向量化）；AVX 下与 FFTW 持平 | 同上 benchmark 仓库 |
| RustFFT | 2020 年实验版已超过 FFTW；仅支持 AVX，NEON/AVX-512 受 Rust nightly 限制未落地 | [Rust 论坛公告](https://users.rust-lang.org/t/rustfft-5-0-0-experimental-1-now-faster-than-fftw/53049) |
| VkFFT | Vulkan/CUDA/HIP/OpenCL 跨平台 GPU FFT | [ETH 论文](https://www.research-collection.ethz.ch/bitstreams/3559caac-8938-465e-8d78-76f8b44076ba/download) |
| juFFTe | 把高性能 FFT 带到 RISC-V RVV 1.0，多核下显著超 FFTW | [arXiv:2608.28076](https://arxiv.org/html/2608.28076v1) |

**优化机会**：① pocketfft（NumPy/SciPy 的默认引擎，覆盖 Python 数值生态几乎所有用户）**1D FFT 无 SIMD**——这是覆盖面极广的具体优化点；② FFTW 对新指令集（AVX-512 深度利用、ARM SVE、RVV）跟进缓慢，第三方已逐个证明收益（juFFTe、RustFFT）。证据见上表。

### 5.2 矩阵乘法（Strassen / 分块）

| 实现 | 特点 | 来源 |
|---|---|---|
| OpenBLAS | 手写汇编微内核 + Goto 式分块；比最朴素实现快数十倍 | [attractivechaos 实测](https://attractivechaos.wordpress.com/2016/08/28/optimizing-matrix-multiplication/) |
| BLIS | 微内核 + 多维并行化选择；小矩阵/瘦高矩阵上优于 OpenBLAS（OpenBLAS 对小 M 无法选对并行循环层级）；AMD AOCL 即 BLIS 血统 | [IPDPS 论文](https://jianbinfang.github.io/files/2020-12-11-ipdps.pdf)、[FEniCS 讨论](https://fenicsproject.discourse.group/t/multicore-and-ubuntu/3080) |
| Eigen / Julia MaBLAS / RecursiveFactorization | 小矩阵（≤256）领域纯高级语言实现可超 OpenBLAS | [Julia discourse](https://discourse.julialang.org/t/ann-paddedmatrices-jl-julia-blas-and-partially-sized-arrays/38215/18) |
| 不规则/瘦高 GEMM | IrGEMM 对 MKL/BLIS/OpenBLAS 最多 2–3.4× 提速，说明传统分块未考虑输入形状 | [IrGEMM 论文](https://wang-luhan.github.io/files/IrGEMM.pdf) |

**优化机会**：① OpenBLAS 的**新 CPU 识别滞后**（Zen5 落到 generic/Cooperlake 路径、线程数默认为 1 的实测案例）；② 小矩阵与不规则形状 GEMM 是 OpenBLAS/MKL 共同的已知弱区；③ Strassen 在主流 BLAS 中仍未默认启用。证据：[Julia discourse 案例](https://discourse.julialang.org/t/poor-openblas-performance-for-large-matrix-multiply/119354)、IrGEMM 论文。

### 5.3 大数运算

| 实现 | 特点 | 来源 |
|---|---|---|
| **GMP** | 分层算法（schoolbook→Karatsuba→Toom-3/4→Schönhage-Strassen FFT），阈值逐平台 tune；阈值调优本身可带来 2–5× 差异 | [GMP 文档（gmp.info）](https://forge.greyc.fr/svn/toulbar2-caen/lib/win32/gmp/info/gmp.info-5)、[综述论文 Table 2](https://simonrs.com/eulercircle/irpw2025/shamik-multalg-paper.pdf) |
| CPython `int` | 仅 schoolbook + Karatsuba（Objects/longobject.c），无 Toom/FFT 路径——超大整数乘法与 GMP 差距是渐近级的 | CPython 源码（Objects/longobject.c，Karatsuba 节） |
| Java `BigInteger` | schoolbook + Karatsuba + Toom-Cook 3，无 FFT | JDK 源码（java.math.BigInteger） |
| Harvey–van der Hoeven O(n log n) | "银河算法"，不实用 | 同上综述 |

**优化机会**：解释型语言的内建大数（CPython 最明显）停留在 Karatsuba，而 GMP 的完整分层 + 逐平台阈值调优模型未被复制；新指令（VPCLMULQDQ 等）在二进制多项式乘法已被证明有效（[HQC 优化论文](https://arxiv.org/html/2608.18145)），移植到整数大数库的公开工作很少。

### 5.4 随机数生成

| 实现 | 特点 | 来源 |
|---|---|---|
| xoshiro256\*\* / xoroshiro128++ | 亚纳秒级、通过 BigCrush 与 PractRand 16TB；xoroshiro128+ 快约 15% 但低位是 LFSR | [prng.di.unimi.it shootout](https://prng.di.unimi.it/) |
| PCG64 / PCG64-DXSM | NumPy 默认；统计质量极佳，速度约为 xoroshiro 一半 | [nullprogram shootout](https://nullprogram.com/blog/2017/09/21/) |
| Philox/Threefry | 计数器型，GPU/大规模并行的事实选择 | [Numerics.NET 选型指南](https://numerics.net/documentation/latest/mathematics/random-numbers/choosing-a-random-number-generator) |
| Mersenne Twister | 仍是 C++ `std::mt19937` 与大量遗留系统默认；速度慢、状态 2.5KB、统计质量逊于当代方案 | 同上两来源 |

**优化机会**：本族算法层已收敛（xoshiro/PCG 够用且快），剩余空间在**批量生成的 SIMD 化**（Philox/SFC 的向量化填充）与遗留默认值的替换（C++ std 默认 MT19937），属低优先级。

---

## 6. 专题：教育与生产的落差

调研中反复出现的模式——教科书/标准库实现与已知最优实现之间存在多年甚至数十年的落差：

1. **`std::regex`：教科书级实现 + ABI 冻结 = 永久锁定**。libstdc++ 比 Boost.Regex 慢 10× 以上（[lmi 实测](https://lists.nongnu.org/archive/html/lmi/2016-07/msg00010.html)），libc++ 再慢 10×（[issue #60991](https://github.com/llvm/llvm-project/issues/60991)）。这是落差最严重、证据最充分的案例。
2. **`std::unordered_map`：2003 年标准化文档中一句"后续讨论均假设链式"把 C++ 排除在开放寻址之外 15 年**（[hftuniversity](https://hftuniversity.com/post/all-further-discussion-will-assume-chaining-how-a-2003-sentence-locked-c-out-of-fast-hash-maps)）。Rust/Go/Abseil 已分别完成 Swiss Table 化，C++ 标准无法跟进。
3. **`std::map`/`std::set` 红黑树**：STL 之父本人承认今天会选 B\* 树（[Stepanov](https://gist.github.com/justinmeiners/57f38bddae9029db3c6401fae113bd7c)）；`std::set<int32>` 每元素 40B vs absl B 树 5.1B（[abseil 文档](https://abseil.io/about/design/btree)）。
4. **CPython 的字符串查找与正则**：`fastsearch` 无 SIMD（[源码注释](https://dmoj.ca/problem/bf4)），`re` 多模式慢 400×（[geekmonkey](https://geekmonkey.org/regular-expression-matching-at-scale-with-hyperscan/)）——而 glibc 早在 2018 年就完成了同类函数的 SIMD/Horspool 化。
5. **CPython/Java 的 TimSort 停留在 2002 年设计**：Rust 的 driftsort（glidesort 血统）报告随机数据 3× 于 Rust 旧 TimSort，且不依赖 SIMD（[glidesort](https://github.com/orlp/glidesort)）；powersort 的近最优合并策略已有 Java 参考实现但未进 JDK（[power-sort.github.io](https://power-sort.github.io/)）。
6. **Go 排序的"半移植"**：pdqsort 进标准库时主动砍掉 BlockQuicksort 优化（[issue #50154](https://github.com/golang/go/issues/50154)）。
7. **图算法的主流库（NetworkX、BGL）单线程化**，与 Ligra/Gunrock 的并行实现差 1–2 个数量级（[Gunrock 论文](https://arxiv.org/pdf/1501.05387)）。
8. **NumPy/SciPy 默认 FFT（pocketfft）的 1D 路径无向量化**（[gemmi benchmark](https://github.com/project-gemmi/benchmarking-fft)）——影响面极大但少有人注意。

正面案例：Go 1.24 的 Swiss Table map 证明"老代码里仍有显著收益可挖"（[Go 1.24 说明](https://www.ayokoding.com/en/learn/software-engineering/programming-languages/golang/release-highlights/go-1-24/)）；x86-simd-sort 进 NumPy/PyTorch 证明排序 10× 级收益可以直达最终用户。

---

## 7. 优先级排序表（影响力 × 优化可行性）

评分 1–5；影响力 = 受影响用户规模 × 潜在加速幅度；可行性 = 技术路径明确度 × 工程阻力（ABI/治理）的倒数。

| # | 目标 | 影响力 | 可行性 | 综合 | 依据 |
|---|---|---|---|---|---|
| 1 | **libc++ `std::regex` 重写/优化**（llvm-project issue #60991 开放中，维护者欢迎补丁） | 5 | 4 | **20** | [issue](https://github.com/llvm/llvm-project/issues/60991)、[lmi 实测](https://lists.nongnu.org/archive/html/lmi/2016-07/msg00010.html) |
| 2 | **pocketfft 1D FFT 向量化**（NumPy/SciPy 默认引擎，覆盖几乎全部 Python 数值用户） | 5 | 4 | **20** | [gemmi benchmark](https://github.com/project-gemmi/benchmarking-fft) |
| 3 | **CPython `fastsearch` SIMD 化**（照抄 glibc 2018 年已验证的 Horspool+SIMD 路径） | 4 | 5 | **20** | [CPython 源码](https://dmoj.ca/problem/bf4)、[glibc 提交](https://sourceware.org/pipermail/glibc-cvs/2019q3/067629.html) |
| 4 | **libstdc++/libc++ `std::sort` 对基本类型特化 pdqsort/branchless 或 AVX-512 路径**（排序实现不受 ABI 冻结） | 5 | 3 | **15** | [arXiv:1704.08579](https://arxiv.org/html/1704.08579v2)、[Phoronix](https://www.phoronix.com/news/x86-simd-sort-1.0) |
| 5 | **Go `regexp` 增加 SIMD literal 预过滤**（借鉴 Rust regex 的 memchr/Teddy 路径） | 4 | 4 | **16** | [golangnews](https://golangnews.com/stories/4922-coregex-pure-go-production-grade-regex-engine.-up-to-3-3000x-faster-than-stdlib.)、[aho-corasick](https://github.com/BurntSushi/aho-corasick) |
| 6 | **Edlib 短字符串路径优化**（issue #144 自 2020 年开放；生物信息学高频子程序） | 3 | 5 | **15** | [issue #144](https://github.com/Martinsos/edlib/issues/144) |
| 7 | **Java/CPython 稳定排序引入 powersort 合并策略**（已有 Java 参考实现，合并顺序改动低风险） | 3 | 4 | **12** | [power-sort.github.io](https://power-sort.github.io/) |
| 8 | **Go Swiss Table map 冷缓存回归修复**（golang/go#70835，Prometheus 实测回归） | 3 | 4 | **12** | [分析](https://blog.gaborkoos.com/posts/2026-07-24-Golang-Maps-How-Swiss-Tables-Replaced-the-Old-Bucket-Design/) |
| 9 | **Go `sort` 恢复/重评 BlockQuicksort 块分区** | 3 | 3 | **9** | [issue #50154](https://github.com/golang/go/issues/50154) |
| 10 | **通用 CPU 图库引入 delta-stepping 并行 SSSP / 方向优化 BFS** | 3 | 3 | **9** | [Gunrock](https://arxiv.org/pdf/1501.05387)、[GraphBLAST](https://par.nsf.gov/servlets/purl/10294352) |
| 11 | **OpenBLAS 小矩阵/不规则形状 GEMM 与新 CPU 识别** | 3 | 3 | **9** | [IrGEMM](https://wang-luhan.github.io/files/IrGEMM.pdf)、[Julia discourse](https://discourse.julialang.org/t/poor-openblas-performance-for-large-matrix-multiply/119354) |
| 12 | **xor/binary fuse filter 取代教科书 Bloom**（数据库/缓存层普及） | 3 | 3 | **9** | [arXiv:2201.01174](https://arxiv.org/pdf/2201.01174v1.pdf) |
| 13 | 无分支/Eytzinger/批量二分进入语言标准库 API | 2 | 4 | **8** | [Khuong-Morin](https://arxiv.org/pdf/1509.05053.pdf) |
| 14 | Aho-Corasick 大模式集场景的 AVX-512 Teddy | 2 | 3 | **6** | [aho-corasick 文档](https://teaclave.apache.org/api-docs/client-sdk-rust/aho_corasick/index.html) |
| 15 | 大数运算：CPython `int` 引入 Toom/FFT 分层 | 2 | 2 | **4** | [GMP 文档](https://forge.greyc.fr/svn/toulbar2-caen/lib/win32/gmp/info/gmp.info-5) |

---

## 8. 方法论说明与局限性

- 以一手来源（GitHub issue/源码、官方文档、论文）为主；部分中文博客/CSDN 转载仅用于佐证已有一手来源的结论。
- benchmark 数据高度依赖硬件与负载形态（哈希表负载因子、排序数据分布等），引用时已注明出处语境，横向比较需谨慎（[ankerl 的告诫](https://www.libhunt.com/compare/ktprime-emhash-vs-robin-hood-hashing)）。
- 未覆盖：外部存储算法（B+ 树磁盘页管理）、分布式算法；GPU 方向仅在与 CPU 库直接可比处引用。
