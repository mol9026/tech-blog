---
title: "NPU 图模式缺陷指纹库：五类特征、两条警告"
date: 2026-11-05
slug: npu-precision-fingerprints
draft: false
description: "五类图模式精度缺陷的指纹：特征→假设→验证实验。五条里两条带着'别直接信我'的归因修正警告——负样本是指纹库里最值钱的条目。"
tags: ["NPU", "精度排查", "torch.compile", "调试方法论"]
categories: ["精度排查"]
summary: "图模式输出不对时，最贵的动作是从零开始猜。指纹库把已知的五类缺陷压缩成'特征→假设→验证实验'的查表条目：相位漂移、索引错位、并发竞争、layout 契约、buffer 复用。五条里两条后来被证明归因错了——修正后的条目反而成了库里最有价值的部分，因为它们教会你'特征像'和'真的是'之间的距离。"
---

> **TL;DR**：NPU 图模式（`torch.compile` + Triton 后端）精度排错，我们积累了一张五条目的指纹库，每条都是"匹配特征 → 机制假设 → 验证实验"三段式：
> 1. **相位漂移**：误差随 token 位置单调增长 → fp32 常量被量化 → 查 dtype 做对照 ⚠ 已证伪（新依赖已自动修复）
> 2. **索引/轴合并错位**：单 kernel 重放与图内不一致 → 索引按错误 layout 生成 → 喂 contiguous 对照（已修复，见[两行代码实锤](https://mol9026.github.io/tech-blog/posts/two-line-discriminating-experiment/)）
> 3. **并发竞争**：同输入多次执行 bitwise 不一致 → in-place 读改写竞态 → 10 次重放采样（已修复）
> 4. **layout 契约缺失**：错误从喂 strided 数据的算子下游开始 → 算子要求 contiguous 未声明 ⚠ 对 FA v3 不适用（实证支持 strided）
> 5. **buffer 复用/时序**：kernel 全对整链错 → 复用写进还在读的 buffer → data_ptr 同址比对
>
> 五条里两条带 ⚠ 归因修正。**负样本是指纹库里最值钱的条目**——它们标出了"特征像"和"真的是"之间的沟。

## 指纹库怎么用

指纹库不是知识库文章，是排错流水线的一环。完整链路：

```text
模型图模式输出不对
  → 诊断工具产出误差特征（哪层坏、哪个 kernel、误差怎么分布、是否确定）
  → 特征查指纹表，得到 1~2 个假设 + 对应的验证实验
  → 跑验证实验，实锤后修复、定案、（若新发现）写回指纹库
```

关键设计是**指纹只窄化假设，不代替验证**。每条指纹的"假设"字段写的是机制猜想，"验证"字段写的是能实锤它的实验——命中指纹只是让你少绕路，结论永远来自实验。

用表之前还有一道前置：**先核对环境指纹**。版本组合本身会改变缺陷行为（同一个栈在不同 CANN/torch_npu 组合上病灶不同），版本号不对，后面全白查。

## 五条指纹

### 1. 相位漂移（量化类）⚠ 已修正

- **特征**：`err_grows_with_position`——误差按 token 位置分桶严格单调递增
- **假设**：构造侧把 fp32 常量（rotary 的 inv_freq 等）量化成模型 dtype，相位误差 × 位置线性放大
- **验证**：查 inv_freq.dtype；对照 from_pretrained（保 fp32）vs from_config+.to(fp16)（会量化）

**⚠ 归因修正**：这条指纹的机制在新版依赖里已经不存在了——transformers 的 rotary embedding 前向会把 inv_freq cast 回 fp32 计算，实测两种构造 cos 1.004152 vs 1.004134，噪声级差异。**在修复态编译器上此指纹为噪声级**：命中先跑对照实验，别直接按它修（它曾让我们差点提一个不必要的 PR，详见[红绿灯篇](https://mol9026.github.io/tech-blog/posts/red-green-fix-isolation/)）。

### 2. 索引/轴合并错位类

- **特征**：`kernel_replay_mismatch`——单个 kernel 重放结果与图内执行不一致
- **假设**：融合 kernel 的索引按错误的内存 layout 生成（跨轴合并/重排）
- **验证**：单 kernel 隔离测试，喂 contiguous 输入对照
- **已修复实例**：融合器允许不相邻轴合并，重索引的位置语义无法表达，RoPE 系数交叉（PR [#47232](https://gitcode.com/ascend/pytorch/pulls/47232)）。定位过程见[两行代码实锤](https://mol9026.github.io/tech-blog/posts/two-line-discriminating-experiment/)——"数学错还是索引错"一刀分开的对照实验。

### 3. 并发竞争（in-place 读改写类）

- **特征**：`nondeterministic`——同一输入多次执行，bitwise 不一致
- **假设**：reduction kernel 读改写同一 buffer，多核 group dispatch 下循环交错
- **验证**：单 kernel 重放 10 次看 bitwise；查编译产物的 `mutated_arg_names: ['in_out_ptr0']` + `'reuse'` 注释
- **已修复实例**：RMSNorm 融合 kernel 的统计循环与写回循环共用 in-out buffer，分母用到脏数据，输出放大约 300 倍、偶发 NaN（PR [#47261](https://gitcode.com/ascend/pytorch/pulls/47261)）。

### 4. layout 契约缺失类 ⚠ 部分修正

- **特征**：`downstream_break_after_strided_input`——第一个出错的 buffer，上游是喂 strided 数据的 extern 算子
- **假设**：算子库要求物理 contiguous 但算子未声明约束，编译器直喂 strided 数据
- **验证**：喂 contiguous 对照——strided 错、contig 对 = 实锤

**⚠ 归因修正（范围性）**：对 attention 算子（FA v3）**不适用**——隔离实验三组全绿证明它实际支持 strided BNSD（D 维 unit stride 即可）；历史上 cos 0.158 的惨案是另一个缺陷未修复期的测量污染。但**其他算子库算子命中此特征时仍按流程验证**——修正划的是边界，不是把整条指纹作废。

### 5. buffer 复用/调度时序类

- **特征**：`single_ok_chain_bad`——所有 kernel 单点测试都正确，整链结果错
- **假设**：in-place 复用把后续输出写进了仍被上游读取的 buffer
- **验证**：逐 buffer 溯源，data_ptr 同址比对找污染链；查产物 `bufN = bufM` + `del bufM  # reuse` 模式
- **静态预检**：产物审计工具秒级扫描危险复用模式（reduction+in-place 组合、reuse-after-write），在跑任何实验之前先过一遍。

## 负样本为什么最值钱

直觉上，归因修正是"错误记录"，该悄悄改掉。我们反着做：修正必须**显式标注**（⚠ + 修正日期 + 对照数据 + 证据文件路径），和原条目放在一起。三个理由：

1. **未来你会再遇到同样的特征**。误差随位置增长这个现象是真实存在的——下次看到，你需要知道的不是"这是相位漂移"，而是"这个特征在当前栈上有两种解释，先跑对照"。没有修正记录，你会把老路再走一遍。
2. **特征与根因是多对多的**。同一个"坏"现象可以由不同机制产生；同一机制在不同架构上表现不同。指纹库真正记录的是"特征 → 值得怀疑的方向"，修正条目恰好把多对多的边界画清楚了。
3. **修正过程本身就是可复用资产**。rc2/rc3 的修正靠的是隔离红绿灯（单撤修复 → 对照基线），这个流程写在修正条目里，等于给每个后来者附了一份操作示范。

## 维护纪律

- **只增不删**：修正不改写原条目，加 ⚠ 段落保留原文——证据链可审计，回看时能看到"当时为什么这么判"。
- **新条目三段式缺一不可**：没有验证实验的指纹不收录——"特征 + 猜想"只是线索，收录进库会被当成结论用。
- **条目要有适用边界**：像指纹 4 那样写清"对哪些算子适用、对哪些已证伪"，比笼统的"可能适用"有用得多。
- **环境指纹先行**：库的第一页永远是版本组合表——缺陷行为随组合漂移，脱离版本谈指纹是空中楼阁。

---

*本文是精度排查系列的第二篇。前篇：[红绿灯归因](https://mol9026.github.io/tech-blog/posts/red-green-fix-isolation/)。指纹 2 的定位故事单开了一篇：[两行代码实锤](https://mol9026.github.io/tech-blog/posts/two-line-discriminating-experiment/)。产指纹特征的四件套工具见[下一篇](https://mol9026.github.io/tech-blog/posts/precision-toolkit-design/)。*
