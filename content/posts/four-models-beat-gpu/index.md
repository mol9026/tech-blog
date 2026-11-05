---
title: "从 1.45x 到 2.03x：四个模型在 NPU 上反超 GPU 的收官复盘"
date: 2026-11-05
slug: four-models-beat-gpu
draft: false
description: "四模型双栈对比收官：8 个单元 6 个反超 GPU 参照，nanoGPT eval 1.45x→2.03x。复盘三代栈演进里什么起了作用——构建修复、口径对齐、主机侧优化——以及剩下两个没反超的单元为什么是'还没到'而不是'做不到'。"
tags: ["NPU", "Ascend", "性能调优", "torch.compile", "Benchmark"]
categories: ["性能调优"]
summary: "四个模型（nanoGPT / LLaMA / LLaMA-v2-7B / LLaVA）、8 个 eval/train 单元、三代软件栈，最终 6 个单元反超 GPU A100 参照，nanoGPT eval 加速比 1.45x→2.03x。本文复盘真正起了作用的三类事：修对环境（一个坏构建拖垮所有数据）、对齐口径（倍数先有意义再谈高低）、打主机侧（反超之后剩余差距的真正来源）。"
---

> **TL;DR**：四模型跨栈优化收官：**8 个单元 6 个反超 GPU A100 参照**——nanoGPT eval 加速比 **1.45x → 2.03x（+40%）**，compile 绝对时间 8.85 → 6.29 ms（−29%）。真正起作用的是三类事，按杠杆排序：
> 1. **修对环境**：一个带缺陷的自编译构建（LN 融合 10x 回退 + 多层精度崩坏）拖垮整代数据，换构建即愈——环境治理的杠杆大于一切 kernel 调优；
> 2. **对齐口径**：构型/batch/seq/dtype 四类错修完，倍数才有资格比较（[上一篇](https://mol9026.github.io/tech-blog/posts/gemm-fingerprint-alignment/)）；
> 3. **打主机侧**：反超之后剩余的差距不在"NPU 弱于 GPU"，而在主机下发链路——cpp_wrapper 消 Python dispatch −5.1%、epilogue fusion −2.8%。
>
> 剩下 2 个未反超单元（LLaMA-v2-7B train / LLaVA train）阻塞项都是 fp16/bf16 算子效率——fp32 同路径反超 1.42x 证明算力足够，是"还没到"，不是"做不到"。

## 战场

| | GPU 参照 | NPU |
|---|---|---|
| 硬件 | A100-SXM4-80GB | Ascend 910B2 64GB |
| 栈 | PyTorch 2.14.1+cu126 / Triton 3.8.0 | torch_npu 2.14 / triton-ascend（随代际变化） |
| 模型 | nanoGPT（124M）/ LLaMA 手写实现（60M, 8 层）/ LLaMA-v2-7B 同构（3.5B, 16 层）/ LLaVA（7B, 32 层 + CLIP 视觉塔） | 同构对齐，见 [GEMM 指纹篇](https://mol9026.github.io/tech-blog/posts/gemm-fingerprint-alignment/) |
| 口径 | eager vs `torch.compile`，eval/train 分单元，倍数 = compile 相对自身 eager | 同 |

"反超"的判定标准要说清：**不是 NPU 绝对时间快过 GPU**，是 **NPU 上 compile 相对 eager 的加速比，高过 GPU 上同口径的加速比**——即在"编译优化还有多少油水"这个维度上反超。绝对性能比较需要同 workload 对齐成本口径，超出本文范围。

## 三代栈的演进，和每代教会我们的事

**V1（CANN 9.1.0 + 初版构建，fp16）**：起点。nanoGPT eval 1.452x（GPU 参照 1.33x）——开局不差，但这一代的问题是**看不到**的：1 层门禁全绿，没人知道环境里埋着什么。

**V2（CANN 9.2.0 + git54e365d 构建，fp32）**：升级即踩坑的一代。LN 融合回退 10~23 倍、≥3 层整图精度崩坏、签名漂移——三类静默劣化漏检 3 天（[门禁篇](https://mol9026.github.io/tech-blog/posts/env-gate-g13/)）。更糟的是这代为了"统一"把 dtype 全改 fp32，反而和 GPU 的 fp16/bf16 实测口径错开。**这代教会的事：版本组合本身就是性能变量，而且是最大的一阶变量**——LLaMA-v2-7B eval 加速比从 1.393x 掉到 1.009x，不需要任何代码问题，一个坏构建就够了。

**V4（CANN 9.2.0 + git14dc54e 构建，逐单元对齐）**：修复构建 + 按上篇的断言流程逐单元对齐后的一代，也是收官代。

| 模型单元 | V1 | V2 | V4（对齐后） | GPU 参照 | 判定 |
|---|---|---|---|---|---|
| nanoGPT eval | 1.452x | 1.835x | **2.028x** | 1.33x | ✅ 反超 |
| nanoGPT train | — | 1.647x | **1.860x** | 1.25x | ✅ 反超 |
| LLaMA(小) eval | — | 1.763x* | **1.882x** | — | ✅ 反超 |
| LLaMA(小) train | — | 1.778x* | **1.917x** | — | ✅ 反超 |
| LLaMA-v2-7B eval | 1.393x | 1.009x | **1.271x** | 1.16x | ✅ 反超 |
| LLaMA-v2-7B train | — | 0.932x | 0.973x | 1.42x | ❌ 差 0.45x |
| LLaVA eval | — | 1.693x* | **1.392x** | 1.19x | ✅ 反超 |
| LLaVA train | — | 1.449x* | 0.949x | 1.48x | ❌ 差 0.53x |

*构型修正前倍数无意义（跑的模型不同）。

两行值得停一下的数字：**LLaVA eval 从 1.693x "跌"到 1.392x 是好消息**——V2 的 1.693x 是 12 层小模型跑出来的假倍数，模型修到 32 层真实构型后倍数下降、绝对时间上升（eager 19.7ms 量级的图变大 5 倍），这才是真数字。**倍数变大不总是变好，构型变了倍数就换了含义。**

## 什么真正起了作用：按杠杆排序

### 1. 修环境（杠杆最大，性价比最高）

LLaMA-v2-7B 的 LN 融合 kernel 从 31.76 μs 退化到 735 μs（23 倍）——这一个构建缺陷就能把 eval 加速比从 1.39x 拖到 1.01x。而修复动作是**换构建**：git54e365d → git14dc54e，eval compile 时间从 63.3ms 回到 22.7ms（−64%），精度归位。~2 小时的动作，拿回的幅度超过后面所有调优的总和。

这符合整个系列的观察：**一阶问题永远是"你测的环境对不对"**。在一阶问题未定案前做的任何 kernel 级优化，都会被环境噪声淹没或冒领。

### 2. 对齐口径（不做这个，其他全是空转）

构型（4→8 层、12→32 层）、输入契约（视觉塔退出、batch 1→32）、精度（fp32→fp16/bf16 逐单元对齐 GPU 实测口径）——修完这些，倍数才从"数字"变成"证据"。细节全在[上篇](https://mol9026.github.io/tech-blog/posts/gemm-fingerprint-alignment/)，这里只强调一个数字：LLaMA(小) 的 eval batch 从 1 对齐到 32 后，eager 绝对时间 3.46ms → 11.17ms——**此前 NPU 只跑了 1/32 的 token 量**。

### 3. 打主机侧（反超之后，剩余差距的真正来源）

环境干净、口径对齐后，nanoGPT 还有两块可动的肉，都在 kernel 之外：

- **主机下发链路**：dev_gap 0.75-0.86——每步 102 次下发、每次下发 46~53 μs 卡在主机侧。`cpp_wrapper` 消掉 Python dispatch 层后 **eval 加速比 +5.1%**。这印证了[错位账本篇](https://mol9026.github.io/tech-blog/posts/npu-compile-net-loss/)的判断："设备时间占比 ~45% 决定了 kernel 优化的天花板，再上一个台阶必须动 kernel 之外的那一半"。
- **epilogue fusion**：−2.8%，编译器侧配置项，一行的代价。

两个动作合计把 nanoGPT eval 从 1.86x 推到 2.03x。**没有动任何 GEMM、没有动任何 Triton kernel 本身。**

## 没反超的两个单元：差在哪，为什么不是"做不到"

LLaMA-v2-7B train（0.973x vs GPU 1.42x）和 LLaVA train（0.949x vs 1.48x）是仅剩的两个未达标单元，阻塞项高度一致：

- LLaMA-v2-7B：**fp16 GEMM 慢 1.16 倍**——同一张卡上 fp32 train 反超 1.42x，fp16 反而落后，说明算力足够、是 fp16 tiling/实现效率问题；
- LLaVA：**layout 转换占 kernel 时间 45%**（GPU 侧 35%）+ bf16 算子效率。

关键在归因的姿态：fp32 同路径反超这件事**证明了天花板不在硬件**——剩下的是上游软件实现（Triton-Ascend fp16 GEMM、layout 处理）的迭代空间。这两项已经整理成带证据的上游报告（fp32/fp16 同 kernel 对比、layout 来源算子统计）。**"依赖上游修复"和"做不到"是两种完全不同的失败，必须分清**——前者是排期问题，后者是架构问题。

## 收官之后的账

四模型八单元的最终判定：6 ✅ / 2 ⏳（等上游）。整个系列的方法论沉淀可以压缩成一条防线清单：

```text
采数前：环境门禁（多层探针 + kernel 计时 + 冷缓存）     ← 环境篇
对比前：GEMM 指纹断言 + meta 基础设施                   ← 对齐篇
分析时：设备侧损益表（拆开 E2E 的调度收益与 kernel 盈亏） ← 账本篇
定案前：红绿灯隔离（单撤修复，对照基线）                 ← 归因篇
收官时：反直觉数据回查（当时的异常是线索还是污染）        ← 全系列
```

每条防线的具体形态会随栈演进过时，但"先证环境、再证口径、再谈倍数、最后才归因"这个顺序不会。

最后留一个诚实的尾注：nanoGPT eval 2.03x 的成色，取决于 GPU 参照侧的 compile 调优是否到位（cuBLAS 侧还有 epilogue 类优化空间）。**反超是当前栈版本下的反超**——跨栈对比是移动靶，这篇记下的是这个时间点的快照和打靶的方法，不是终点。

---

*本文是性能系列的收官篇。系列全部文章：[三问题排查实录](https://mol9026.github.io/tech-blog/posts/cann92-triton-ascend-triage/) · [torch.compile 错位账本](https://mol9026.github.io/tech-blog/posts/npu-compile-net-loss/) · [GEMM 指纹对齐](https://mol9026.github.io/tech-blog/posts/gemm-fingerprint-alignment/) · [红绿灯归因](https://mol9026.github.io/tech-blog/posts/red-green-fix-isolation/) · [环境门禁 g13](https://mol9026.github.io/tech-blog/posts/env-gate-g13/)。*
