---
title: "50 个模型精度普查：13 个全绿之后我学到的"
date: 2026-11-07
slug: puncture-survey-50-models
draft: false
description: "用四件套对 50 个模型逐一做图模式精度验证。13 个全绿不是重点，重点是普查教的事：HEALTHY 的三种含义、graph break 不可怕但要写清楚、L1 就够吗、工具边界怎么变成下个工具的需求。"
tags: ["NPU", "精度排查", "torch.compile", "模型验证"]
categories: ["精度排查"]
summary: "修复态编译器到底覆盖多少模型？带着这个问题做 50 模型穿刺普查。本文记录前 13 个模型的全绿背后：HEALTHY 需要定义（被捕获子图正确 ≠ 整图覆盖）、graph break 的两类模式与处置、L1 短路让普查成本降到每模型 10 分钟、以及最意外的收获——工具边界暴露的参数构造难题，变成了下一个工具的需求来源。"
---

> **TL;DR**：用诊断四件套（[前文](https://mol9026.github.io/tech-blog/posts/precision-toolkit-design/)）对 50 个模型逐一验证图模式精度（"穿刺"）。前 13 个全绿——纯 decoder、手写 enc-dec、相对位置编码、双模态 ×2、双向 attention、跨层参数共享、Meta 原版手写实现。但全绿不是这篇的重点，重点是普查教的事：
> 1. **HEALTHY 要定义**：graph break 后"被捕获子图全部正确"，不等于整图编译覆盖——两者必须分开说；
> 2. **graph break 不可怕，但要写清楚断在哪**：两类模式（data-dependent indexing、断言类检查）都是模型侧结构，不是编译器缺陷；
> 3. **L1 短路是普查的生死效率**：23 秒健康判定，普查成本从"每模型半天"压到"每模型 10 分钟"；
> 4. **工具边界会变成下个工具的需求**：L2 深检被 149 个碎参数卡住，倒逼出 capture-replay 的参数自动构造设计。

## 任务定义：修复态编译器到底行不行

背景：编译器侧四个根因修复在位（[红绿灯篇](https://mol9026.github.io/tech-blog/posts/red-green-fix-isolation/)），单模型（llama2_7b）精度已经恢复。但"一个模型修好了"和"这版编译器能用了"是两回事——**覆盖面**没有证据。

于是有了穿刺普查：对负责的约 50 个模型（TorchBench 口径为主）逐一跑图模式精度验证，判定标准两条：

```text
L1 cos > 0.999（graph vs eager）
且 bitwise 确定（两次执行结果逐位一致）
→ HEALTHY，记台账，下一个
```

选"L1 就够"是个明确的工程决策，后面展开。先说普查跑出来的东西。

## 13 个全绿，绿得各有含义

| # | 模型 | 架构要点 | 结论注记 |
|---|---|---|---|
| 1 | nanogpt | 无 RoPE 纯 decoder | bitwise 确定 |
| 2 | speech_transformer | enc-dec + mask | graph break（data-dependent indexing） |
| 3 | T5 | enc-dec + 相对 PE | tile-contract 性能警告（精度无恙） |
| 4 | llava | CLIP+Llama 双塔 | graph break；image token 构造要点（见下） |
| 5-9 | BERT 家族 ×5 | 双向 attention、参数共享、绑定 LM 头 | 首个双向架构通过 |
| 10 | moondream 同构 | SigLIP+Phi 部分 RoPE | 第二支 vision 家族 |
| 11 | llama_v2_7b_16h | 原始故障模型 | 修复态三口径互证（1.004152） |
| 12 | llama（Meta 原版） | 手写 RMSNorm/RoPE/attn | 独立实现通过 |
| 13 | Llama-3.2-1B 同构 | **GQA 32q/8kv** | GQA 结构通过 |

全绿的价值在架构覆盖面，不在数量：**decoder-only / enc-dec / 双向 / 多模态 / 参数共享 / GQA / 手写实现**——七类结构都过了，修复态编译器的精度覆盖才算有了初步证据。反过来说，如果第一个碰到的就是 BERT（双向）或者 Llama-3.2（GQA），覆盖面结论就要保守得多。

**cos 数值的解读要小心**：好几个模型 cos 是 1.003~1.03（"大于 1"），这不是精度问题的信号，而是随机权重下 logits 尺度大、方向高度对齐的正常表现。判定靠的是阈值 + bitwise 确定性两把尺，不是把 cos 当误差率读。

## HEALTHY 的三种含义，必须分开说

普查最大的口径风险：同样是"HEALTHY"，含义可能完全不同。实测里出现了三种：

1. **整图编译，精度无恙**——最理想的 HEALTHY（nanogpt、BERT 家族）；
2. **graph break，被捕获子图全部正确**——dynamo 在某些结构上切图（speech_transformer 切了 7+ 个子图），被编译的部分全对，break 段保持 eager（正确性由 eager 兜底）；
3. **graph break + 性能隐患**——T5 的相对位置偏置路径产生 strided load，tile 契约无法救援，精度无恙但性能有嫌疑。

台账里每条 HEALTHY 都带注记写明是哪种。原因很实际：**含义 2 的模型，如果团队口径要求"整图覆盖"，就不算达标**——mask 构造的重写是另一项工作。HEALTHY 三个字不加定义，三个月后没人记得它当时承诺了什么。

## graph break：两类模式，都不是编译器的锅

13 个模型里 graph break 出现两类，值得记录断点模式：

- **data-dependent indexing**：`mask[i, lengths[i]:] = 0` 这类按张量值索引的写法，dynamo 无法静态化——break；
- **断言类检查**：`torch._check_tensor_all` 在 placeholder mask 路径触发——break（llava 的 4 子图产物即此）。

两类都是**模型侧代码结构**，不是编译器缺陷。处置姿势：break 段 eager 兜底，精度不受影响；要消掉 break 是模型代码改造的活，收益是性能不是正确性。排查纪律是**别把 graph break 当缺陷报**——它在台账里是注记，不是结论。

## 一个构造陷阱：llava 的 image token 数

多模态模型的穿刺有个不显然的坑：`LlavaForConditionalGeneration` 要求**序列里的 image token 数必须等于 patch 数**（336/14=24 → 576 个连续 image token），否则 placeholder mask 校验直接报错。用 `from_config` 造同构小模型绕开 gated 权重时，输入构造要按这个契约来——图像尺寸定 patch 数，patch 数定 image token 数。这类构造约束台账里必须记全，否则下次换个人穿刺同一个模型会再踩一遍。

## L1 就够吗：分层判定的经济账

穿刺标准只跑 L1（cos + 确定性），L2-L4 深检没跑——不是偷懒，是算过的：

- **L1 的判定力**：cos > 0.999 且 bitwise 确定，对"模型是否健康"已是充分条件。逐位一致的两次执行 + 与 eager 的方向对齐，能排除全部五类已知缺陷的显形特征（[指纹库](https://mol9026.github.io/tech-blog/posts/npu-precision-fingerprints/)的五条特征 L1 都能给出初判信号）；
- **L2-L4 的定位属性**：它们是"出了问题后的定位工具"，不是"没出问题的证明工具"。健康模型跑 L2-L4 是纯开销；
- **经济账**：L1 短路 23 秒/模型（有缓存时），13 个模型一个工作日跑完；若每个都跑全层（含编译 5-10 分钟），成本翻一个数量级。

配套策略：**L1 全绿推进，命中问题的模型才下钻 L2-L4**。普查是宽采样，定位是深挖掘，两件事用不同的工具深度——这个分层后来固化成了 doctor 的默认行为（L1 健康即短路）。

## 意外收获：工具边界变成工具需求

普查里最贵的教训反而来自一次"没跑成"：nanogpt 想跑 L2 深检，发现 capture 需要向产物 Runner.call 传**全量参数**——dynamo 把模型权重按访问序展开成 graph 参数，nanogpt 展开出 149 个（llama 1 层是手工列的 14 个）。手工构造不可扩展，三个候选方案（hook compile 截获真实 inputs / fx_passes 注入探针 / dynamo export 拿参数——第三个实测 1738 个碎参数，不可行）。

这个边界后来直接催生了隔离器的 **capture-replay 架构**：探针注入真实执行、自动抓取全量 buffer——"参数自动构造"从普查的阻塞项变成了工具的核心能力（[四件套篇](https://mol9026.github.io/tech-blog/posts/precision-toolkit-design/)）。**大规模重复性任务遇到工具边界时，边界本身往往就是下个工具的需求说明书**——手工绕过一次两次可以，第 N 次就该造工具了。

## 台账即资产

普查的产出不是"13 个 PASS"一句话，是逐模型台账：日期、L1 数值、确定性、深层检查、结论注记、证据文件路径。它的三种用法：

1. **覆盖面证据**：修复态编译器的架构覆盖矩阵，新模型来了先查台账——同构架已有先例，置信度直接继承；
2. **适用性预判**：四个已知根因对各类架构的适用性表（无 RoPE 架构对相位漂移免疫、LayerNorm+残差是并发竞争的温床……），新模型穿刺前先做风险预判；
3. **性能线索池**：graph break 模式、tile 警告这类注记，是性能方向的候选调查点——精度普查顺路产出的性能 backlog。

## 方法论沉淀

1. **普查用浅探针 + 短路，定位才用深工具**——采样宽度和挖掘深度分开定价。
2. **HEALTHY 必须带定义**——被捕获子图正确 ≠ 整图覆盖，口径不清的绿灯三个月后就是糊涂账。
3. **graph break 是注记不是缺陷**——eager 兜底保证正确性，消除 break 是性能优化，两条线分开。
4. **构造约束记台账**——image token=patch 数这类"文档里没有、报错才知道"的契约，是模型验证的隐性知识，不复盘就流失。
5. **工具边界是下个工具的需求说明书**——手工第 N 次绕过同一个坑，就是造工具的信号。

---

*本文是精度排查系列第五篇。普查用的工具设计见[四件套](https://mol9026.github.io/tech-blog/posts/precision-toolkit-design/)，判定依据见[指纹库](https://mol9026.github.io/tech-blog/posts/npu-precision-fingerprints/)。*
