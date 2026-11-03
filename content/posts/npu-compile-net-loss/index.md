---
title: "E2E 加速 1.57 倍，设备侧却净亏 426μs：torch.compile 在 NPU 上的错位账本"
date: 2026-11-03
slug: npu-compile-net-loss
draft: false
description: "compile 后 E2E 加速 1.565x/1.379x，但 NPU 设备侧 kernel 时间不降反升（+426/+1568 μs/step）——加速全部来自 kernel 之外的调度削减。两套独立分析收敛同一靶点：归一化族融合 kernel。"
tags: ["NPU", "Ascend", "torch.compile", "性能调优", "Benchmark", "Inductor"]
categories: ["性能调优"]
summary: "torch.compile 给 NPU 带来 1.565x 的 E2E 加速，但设备侧 kernel 时间不降反升——钱赔在归一化族的 Triton 融合 kernel 上，加速全部来自调度削减。本文拆一本完整的 kernel 损益表，附两个 benchmark 口径陷阱，以及事后复盘：这批数据测在缺陷组合上，哪些结论翻案了、哪些站住了。"
---

> **TL;DR**：nanogpt 双栈对照（A100 fp32 vs 910B2 fp16）发现一组反直觉的账：
> 1. `torch.compile` 给 NPU 带来 **1.565x（eval）/ 1.379x（train）的 E2E 加速**，但设备侧 kernel 时间**不降反升**（eval **+426**、train **+1568** μs/step）；GPU 侧 compile 是真赚（−411 / −1278）。**NPU 的加速全部来自 kernel 之外的调度削减**。
> 2. 赔的钱几乎全在**归一化族**：CANN 原生 `LayerNormV3` 11.65 μs/次，compile 换成 Triton 融合后 71.81 μs/次，同一个算子慢 **6.2 倍**。旁证排除 launch 开销——是 kernel 内部执行慢。
> 3. NPU 只有约 42.5%（eval）/ 47.1%（train）的时间在设备上：**kernel 时间砍半，E2E 也只降四分之一**。归一化回退 + 数据搬运消除两件事，不动 GEMM 就能拿回 **21~23% E2E**。
> 4. 事后复盘：这批数据测在一个后来确认的**缺陷版本组合**上（[前文](https://mol9026.github.io/tech-blog/posts/cann92-triton-ascend-triage/)）。哪些结论翻案了、哪些站住了，文末交代。

## 数据与方法：先把口径交代清楚

负载：nanogpt（124M，12 层 × 768）单机单卡，eager/compile × inference/train 四矩阵，GPU 侧 A100-SXM4-80GB（torch 2.14.1+cu126，inductor），NPU 侧 Ascend 910B2 64GB（torch_npu 2.14.0，`triton_experimental` 后端），两侧均关闭 graph 捕获、batch=1。

动手看数据前先交代三个口径坑——它们决定了后文所有倍数的态度：

1. **dtype 不一致**：GPU 全 fp32，NPU 全 fp16。fp16 理论上利好 NPU，所以后文所有"NPU 慢 X 倍"都是**偏保守**的。
2. **迭代配置不一致**：GPU warmup=3/active=10/repeat=3，NPU warmup=10/active=50/单次。折算 per-step 时用两侧 kernel 调用次数的**最大公约数**反推 step 数再归一。
3. **同构性校验**：这个折算不是白算的——GELU 融合 kernel 折算后两侧都是 **12 次/step**，与 nanogpt `n_layer=12` 的结构完全吻合，证明归一正确、两侧确为同一模型配置。口径先对齐，数字才有资格说话。

## 反常的账本

compile 在两边的"损益表"长得完全不一样：

| 设备侧 kernel 时间变化（compile vs eager） | eval | train |
|---|---|---|
| GPU A100 | **−411 μs/step**（真赚） | **−1278 μs/step**（真赚） |
| NPU 910B2 | **+426 μs/step**（净亏） | **+1568 μs/step**（净亏） |

但 E2E 上 NPU 的 compile 依然给出 1.565x / 1.379x 的加速。两头对不上的唯一解释：**加速全部来自 kernel 之外的调度削减**——kernel launch 次数、Python 分发、内存分配、同步开销。这不是坏消息里挑好消息：compile 确实消灭了一大块调度成本（compiled 态 NPU 有 4,594 μs/eval-step、13,691 μs/train-step 不在 kernel 上，eager 态这块更大），**但设备计算本身是被拖慢的**。

这是两侧最本质的行为差异：GPU 的 compile 是"kernel 更快 + 调度更省"双赚；NPU 的 compile 是"kernel 更亏 + 调度大省"单赚。**如果只看 E2E 加速比，这笔账完全看不出来。**

## 钱赔在哪：归一化族

把净亏的 kernel 时间按算子家族拆开，几乎全在归一化：

| 同一算子两种命运 | 耗时 |
|---|---|
| CANN 原生 `LayerNormV3`（eager） | 11.65 μs/次 |
| Triton 融合版（compile） | 71.81 μs/次，**6.2 倍** |
| 按 step 累计（eval，25 次/step） | 291.3 → 1795.2 μs/step |

「融合不够」的假设可以先排除——MatMulV2 7.5 μs、AsStrided 4.5 μs 说明 NPU 上 kernel 启动本身不慢，**是融合 kernel 内部执行慢**。这不是融合范围问题，是融合方向问题。

按方案口径的硬对照也指向同一处，规律非常干净：

| 融合组（shape/次数严格对齐） | NPU vs GPU |
|---|---|
| 纯逐元素 GELU | 1.65x（基本持平） |
| ADD+LayerNorm（归约，shape [64,768]，26 vs 25 次对齐） | **17.5x** |
| GELU 反向（归约，shape [3072,64] 对齐） | **29.4x** |

**带归约的融合组慢一个数量级，纯 pointwise 基本持平。** 把四组严格可比的相同融合拉通算：GPU 合计 152.9 μs/step，NPU 合计 2016.0 μs/step——相同融合下 13.2 倍，典型的"相同融合性能 GAP"。

两个分析视角独立走下来（一个按 GAP 分类口径，一个按编译损益表），收敛到同一个靶点。**结论互检过，是稳的。**

## 两个栈的瓶颈画像根本不同

| kernel 时间主体 | GPU A100 | NPU 910B2 |
|---|---|---|
| 厂商 GEMM 库（cuBLAS/cutlass ↔ CANN MatMulV2） | 48~94%（多数 70~90%） | 2~18% |
| Triton/Inductor 融合 kernel | 1~27% | 44~67% |

GPU 的性能押在厂商 GEMM 库上（已经很难再快），NPU 的性能押在编译器生成的融合 kernel 上——**优化主战场根本不在同一个地方**。这直接影响排优先级：在 NPU 上盯融合 kernel 质量，比在 GPU 上有高得多的杠杆。

顺带一个惊人的数据点：`llama_v2_7b_16h` train 中，matmul 被下沉进了 Triton reduction kernel——15 个名为 `matmul` 的融合 kernel 占掉该模型 20.6% 的 kernel 时间，其中单个 kernel 一次调用 **3595 μs**（GPU 上同角色的 reduction kernel 是 6.78 μs）。GEMM 是 NPU 厂商库最强的部分，编译器把它下沉进 Triton 反而交了五百倍的学费——**下沉不是收益，得逐案验证**。

## 数据整形税

GPU 上 view/reshape 是零拷贝的元数据操作，不产生 kernel；NPU 上同样的操作落成**真实 kernel**——train 每步 102 次数据搬运 kernel、725 μs/step（eval 侧仅 AsStrided 一类就是 36 次、161.6 μs/step）。另外 compile 后 NPU 的 kernel 发射次数 190 次/step vs GPU 147 次/step（1.29x），这部分整形税**理论上可以完全消除**。

## 天花板：kernel 优化只够到 E2E 的一半

NPU 只有 42.5%（eval）/ 47.1%（train）的时间花在设备上。这意味着一个容易忽略的算术：**就算把 kernel 时间砍半，E2E 也只降四分之一左右**。"融合分析"视角最容易漏掉这一点——要再上一个台阶，必须动 kernel 之外的那一半（launch、同步、内存分配、fallback）。

在 kernel 之内，两件不动 GEMM 的事的收益上限：

| 优化项 | eval 可省 | train 可省 | E2E 收益 |
|---|---|---|---|
| 归一化回退 CANN 原生算子 | 1504 μs/step | 5244 μs/step | −19% / −20% |
| 消除数据搬运 kernel | 175 μs/step | 725 μs/step | −2.2% / −2.8% |
| **合计** | **1679 μs/step** | **5969 μs/step** | **7.989→6.310 ms（−21.0%）/ 25.902→19.933 ms（−23.0%）** |

两件事加起来拿回两成 E2E，**全程没碰 GEMM 和注意力**——这个负载的瓶颈根本不在算力。

## 两个口径陷阱（方案没覆盖、差点污染结论的）

1. **train 口径夹带了 optimizer step**：GPU train 含 `multi_tensor_apply` 2202 μs/step，NPU 侧明确 no optimizer step。不扣除会低估 GPU——扣掉后 train 差距从 **2.87x 扩大到 3.81x**。跨栈比 train，先核对两侧是否都含 optimizer。
2. **NPU 侧算不出 P50/P90**：`kernel_details.csv` 只有 count/total/avg 三个聚合值，没有逐次耗时。长尾问题（抖动、偶发慢调用）在这份导出里是**不可见**的——需要逐次耗时要提前改采集脚本，事后补不了。

## 事后复盘：这批数据测在缺陷上

诚实交代：这批数据采集于 2026-10-30，NPU 栈是 CANN 9.2.0 + torch_npu git54e365d——三天后（见[前文](https://mol9026.github.io/tech-blog/posts/cann92-triton-ascend-triage/)）我们确认这个组合带着 **LN 融合 kernel 约 10 倍回退的缺陷**，当年 1 层 gate 全绿没能拦住它。

用健康栈（CANN 9.1.0）的同期数据对照，可以清楚地切开两类结论：

**翻案的**：归一化族"6.2 倍、融合做错了方向"的**量级**是缺陷栈特有的——9.1.0 上同一条 ADD+LN 融合只要 6.72 μs/次，**比 eager 的 11.65 μs 还快**。归约融合这个方向本身是对的， defect 栈上"暂关闭归约类融合、回退 CANN 原生算子"是正确的止血动作，但长期结论不是"别做归约融合"，是"先把版本组合管起来"（这正是后来 g13 环境门禁存在的理由）。

**站住的**：这些结论不依赖具体版本，今天依然有效——

- compile 的 E2E 加速必须拆开设备侧损益表看，调度削减会掩盖 kernel 退化（本次 +426/+1568 μs 就是这么藏住的）；
- 两个栈的瓶颈画像不同（GEMM-bound vs fusion-bound），优化杠杆位置不同；
- 数据整形税真实存在且可完全消除；
- 设备时间占比 ~45% 决定了 kernel 优化的天花板；
- 口径纪律（dtype、迭代配置、optimizer step、同构性校验）是跨栈 benchmark 的生死线。

一次基准测试里埋着一个后来才引爆的版本缺陷，这件事本身是最好的教材：**反直觉的数据（kernel 净亏但 E2E 大赚）当时就该当成线索追下去，而不是当成"调度优化很有效"收下。**

## 方法论沉淀

1. **看 E2E 之前先看设备侧损益表**：kernel 时间 compile 前后各记一次，Δ 值是"compile 质量"的直接体检指标。E2E 加速比会被调度收益污染，Δ 不会。
2. **融合健康度看家族分布**：pointwise 持平 + 归约族崩坏，说明问题在归约路径而非融合器整体；家族粒度的拆账比总账有用得多。
3. **跨栈 benchmark 先过口径关**：dtype、迭代配置、optimizer step、P50/P90 可得性，四个坑每一个都能把结论带沟里；同构性要用结构证据（如融合 kernel 调用次数 = n_layer）交叉验证，不能靠模型名相同。
4. **反直觉数据是最好的线索**：账对不上的地方，往往埋着尚未命名的缺陷。

---

*数据口径与完整明细：4 模型 ×（eager/compile）×（inference/train）全矩阵采集，报告 A（GAP 分类口径）与报告 B（编译损益口径）双路独立分析。文中倍数均为偏保守值（GPU fp32 vs NPU fp16）。*
