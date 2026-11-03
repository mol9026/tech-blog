---
title: "CANN 9.2.0 + triton-ascend 升级排查实录：7~23x 性能回退与「3 层即坏」的精度缺陷"
date: 2026-11-03
slug: cann92-triton-ascend-triage
draft: false
description: "一次 CANN 升级引出三个问题：融合 kernel 性能回退 7~23 倍、≥3 层 Transformer 单图编译数值错误、select_algorithm 签名漂移。附完整隔离矩阵与交叉验证方法。"
tags: ["NPU", "Ascend", "torch.compile", "Triton", "精度排查", "性能调优"]
categories: ["排查实录"]
summary: "升级 CANN 9.1.0 → 9.2.0 后，LayerNorm 融合 kernel 回退 7~23 倍，3 层 Transformer 整图编译直接数值崩坏。本文记录完整排查过程：隔离矩阵如何把缺陷逼到「跨 kernel 运行时交互」，双机交叉验证如何排除环境因素，以及可用的规避手段。"
---

> **TL;DR**：把 CANN 9.1.0 升到 9.2.0（torch_npu 走 `triton_experimental` 后端做 `torch.compile`）之后，遇到三件事：
> 1. **LayerNorm/RMSNorm 融合 kernel 性能回退 7~23 倍**，E2E 从 1.38x 加速变成 0.69x 负加速；
> 2. **≥3 层 Transformer 整图编译数值错误**——1 层、2 层都 PASS，3 层 cos 直接掉到 0.823，且单测全 PASS，缺陷藏在跨 kernel 的运行时交互里；
> 3. `select_algorithm` 没跟上 torch 2.14 的签名变更，multi-template 路径必现 `LoweringException`。
>
> 双机交叉验证最终把问题收敛到 **CANN 9.2.0 正式版 + 特定 torch_npu/triton-ascend 组合**，而非"9.2.0 全线引入"或"我方环境装坏"。规避手段：graph_break / 逐层 compile（有性能代价）。

## 环境栈

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B2 (A2) 64GB；交叉验证机 910B4 |
| CANN | 9.1.0（对照）/ **9.2.0（问题栈）** / 9.2.0-beta.2（交叉） |
| torch / torch_npu | 2.14.0+cpu / 2.14.0+git54e365d |
| triton-ascend | 3.6.0 |
| 编译入口 | `torch.compile(options={"npu_backend": "triton_experimental"})` |

测试负载：nanogpt train/eval、16 层 Llama 构造（bs1 seq512 fp16），Kineto trace + `kernel_details.csv` 采集。

## 问题一：融合 kernel 性能回退 7~23 倍

同一硬件、同一模型构造、同一采集脚本，只升 CANN 版本，同名 Triton 融合 kernel 单次耗时：

| kernel（同一条融合） | CANN 9.1.0 | CANN 9.2.0 | 回退 | A100 参照 |
|---|---|---|---|---|
| `add+native_layer_norm+view`（nanogpt, shape[64,768]） | 6.72 μs | 72.7 μs | **10.8x** | 3.8 μs |
| `add+LN+LN_backward+view`（nanogpt） | 10.36 μs | 72.7 μs | **7.0x** | 3.5 μs |
| `_to_copy+add+matmul+mean+pow+rsqrt`（RMSNorm 融合） | 31.76 μs | 735.1 μs | **23.1x** | 6.78 μs |

E2E 影响实打实：16 层 Llama eval 的 compile 加速比从 **1.378x 掉到 0.691x**——升级前是加速，升级后是负优化。nanogpt 里 LN 融合组占了 kernel 总时间 40%+。

**一个很有价值的旁证**：融合器行为也退化了。9.1.0 上 eval 图里 extern（不可融合、落回算子库）的算子只有 4 类——attention/cat/embedding/logical_and，都是天然不可融合项；9.2.0 上 extern 暴涨到 33 类，Mul/Add/Cast/ReduceMean/Rsqrt 全部被拆散裸跑。**性能回退不只是 kernel 本身变慢，融合器整体在退化。**

复现很简单：任意含 LayerNorm/RMSNorm 的模型 + `torch.compile(npu_backend="triton_experimental")`，对比两个 CANN 版本的 `kernel_details.csv`，盯 `triton_unk_fused_add_native_layer_norm*` 和 `triton_unk_fused__to_copy_add_matmul_mean_mul_pow_rsqrt*`。

## 问题二：≥3 层 Transformer 整图编译，数值直接崩

这个比性能回退严重得多。Llama 构造（随机权重 fp16）单图编译，logits 对 eager 的余弦相似度：

| 单图层数 | logits cos(vs eager) |
|---|---|
| 1 层 | 0.999999 PASS |
| 2 层 | 0.999998 PASS |
| **3 层** | **0.823 FAIL（稳定复现）** |
| 16 层 | 0.634 FAIL |

误差特征排除了几类常见嫌疑：**与 token 位置、输入内容无关的均匀偏差；非缩放、非低秩**；层数越多 NaN 风险越大。

### 隔离矩阵：把缺陷逼进「跨 kernel 交互」这个角落

这部分是整个排查里最值得沉淀的方法。我们逐一做了这些隔离，**全部 PASS**：

- 单层 forward 任意层单独 compile —— PASS
- 两两组合层 —— PASS
- 三层逐层**分开** compile —— PASS
- 任意位置插一个 `graph_break`，整图立即恢复 —— PASS
- rotary 融合 kernel、norm 融合 kernel、mask 构造 kernel 独立单测 —— 全 PASS
- bare-triton in-place rotary 双 buffer 连续调用 —— PASS
- 整图 13 个 Triton kernel 逐一静态解剖（q/k in-place rotary 写回、neg/cat rotate_half、FA 实参、norm 自读回链）—— 语义全部正确

单 kernel 语义都对、单层都对、拆开跑都对，唯独**整图 ≥3 层**崩——结论只剩一个：**缺陷在跨 kernel 的运行时交互**。最可疑的方向：同 stream 上 Triton in-place kernel（q/k rotary 直接写回 q/k buffer）与 extern `npu_fusion_attention_v3` 之间的**内存可见性/执行顺序**。层数越多 FA 越多，越容易踩中。

这解释了为什么是"3 层"这个阈值：2 层图里 FA 与 rotary 写回的调度顺序恰好没出问题，3 层开始概率性地暴露——而它稳定复现，说明不是纯随机竞争，是调度模式在 3 层处发生了切换。

> 顺带排除一个干扰项：环境里的 PR #47261（reduction in-place 消除）已生效（`mutated_arg_names=[]`），与本缺陷无关。

### 最小复现

```python
import torch, torch_npu
from transformers import LlamaConfig, AutoModelForCausalLM
torch.manual_seed(42)
cfg = LlamaConfig(num_hidden_layers=3)          # 3 层即触发
m = AutoModelForCausalLM.from_config(cfg).to("npu", torch.float16).eval()
m.config.use_cache = False
torch.manual_seed(123)
ids = torch.randint(0, cfg.vocab_size, (1, 128), device="npu")
attn = torch.ones_like(ids)
with torch.no_grad():
    ref = m(input_ids=ids, attention_mask=attn).logits
c = torch.compile(m, options={"npu_backend": "triton_experimental"})
with torch.no_grad():
    out = c(input_ids=ids, attention_mask=attn).logits
# cos(ref, out) ≈ 0.82
```

### 规避

- 任意位置 `graph_break`（比如插入 `torch._dynamo.graph_break()`）
- 逐层分开 compile

两者都能恢复精度，但都有性能代价——等于放弃了整图融合的收益。

## 问题三：select_algorithm 签名漂移

`torch_npu/_inductor/select_algorithm.py` 的 `patch_algorithm_selector.__call__` 没有接收 torch 2.14 新增的 `best_config_future / is_collective / min_speedup_threshold / benchmark_with_cudagraphs` 四个参数，`tuned_mm` 走 multi-template 路径时直接 `LoweringException: TypeError`，清空 inductor 缓存后必现。本地补 4 形参 + `**kwargs` 即恢复。这是典型的上游 API 漂移没跟住，建议随下一版收敛。

## 交叉验证：怎么排除「我环境装坏了」

排查到这一步，最刺耳的质疑永远是："你们环境是不是装错了？" 处理方式是**在另一台物理机（910B4）、另一套独立安装、另一个 torch_npu 构建**（CANN 9.2.0-beta.2）上跑同样的复现：

| 测试 | 910B2 @ 9.2.0 正式版 | 910B4 @ 9.2.0-beta.2（交叉） |
|---|---|---|
| 2 层整图 cos | 0.999998 PASS | 0.999998 PASS |
| **3 层整图 cos** | **0.823 FAIL** | **0.999997 PASS** |
| LN 融合 kernel | ~70 μs | **3.6 μs** |

三个信息一次读全：

1. 交叉环境（同样在 9.2.0 线、同样走 Triton-Ascend）**精度和性能都正常** → 排除"9.2.0 全线引入"；
2. 问题栈可稳定复现 → 排除"我方装坏/偶发"；
3. 交叉环境 LN kernel 3.6 μs 甚至优于 9.1.0 的 6.7 μs → **Triton-Ascend 的 LN 融合本身能做到很快**，进一步支持"特定组合退化"而非"能力不足"。

最终收敛：问题特定于 **CANN 9.2.0 正式版 + torch_npu git54e365d + triton-ascend 3.6.0 这一组合**（或该构建的打包缺陷），对版本组合敏感——别的组合可能已修复，也可能还没触发。

## 方法论沉淀

这次排查可以压缩成四条纪律，下次直接复用：

1. **单变量对照先行**：同硬件、同模型、同脚本，只切 CANN 版本，数据表自己会说话。A100 参照列值得常备——它告诉你"融合天花板"在哪，防止把"本来就该这么慢"误判成回退。
2. **隔离矩阵 + 静态解剖双管齐下**：动态隔离（单测/分层/graph_break）划出缺陷的边界，静态解剖排除 kernel 自身语义错误；两者交集就是"跨 kernel 交互"这类最难缠的缺陷。
3. **交叉验证杀掉环境假设**：另一台机器、另一套安装、另一个构建，三重独立性。这一步不做，所有结论都悬在"可能只是你环境坏了"上面。
4. **extern 算子分布是融合器健康度的免费体检指标**：对比正常/异常栈的 extern 类目数，一眼看出融合器在退化。

## 现状

- 问题一、二已定位到特定版本组合，规避手段可用（graph_break / 逐层 compile），待版本收敛；
- 问题三已本地修复；
- 另有一个 `can_merge_dims` 相邻轴守卫的已知修复（对应 triton-ascend 仓库 issue #5209 / PR #47232）在重装后遗漏恢复，已补回。

后续有版本收敛结果我会更新在这篇文章里。
