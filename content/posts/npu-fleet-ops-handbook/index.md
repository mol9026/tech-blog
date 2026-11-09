---
title: "四台 NPU 服务器的运维手账：接入、审计、病灶、备份、迁移"
date: 2026-11-07
slug: npu-fleet-ops-handbook
draft: false
description: "共享 NPU 机群的实战运维：SSH config 别名制、资源审计五件套、系统盘病灶的命中率排序、备份双端校验、重装前迁移清单。六条踩坑，每条有事故背景。"
tags: ["NPU", "运维", "服务器管理", "工程效率"]
categories: ["工程效率"]
summary: "搞模型的人迟早要管机器。四台共享 910B 的实战沉淀：接入走别名制（禁止裸连）、审计五件套一条命令、系统盘满的病灶从来不在 /home（LVM 独立分区）、备份必须双端 md5（不一致=没备份）、重装迁移按数据性质分流（权重与代码不同命）。核心原则只有一条：共享机上，确认归属再动手。"
---

> **TL;DR**：四台共享 NPU 服务器（多团队共用）的运维实战，沉淀成五段流程：
> 1. **接入**：SSH config 别名制，2 分钟/台——禁止裸连是纪律不是洁癖；
> 2. **审计**：五件套（CPU/内存/NPU 卡/磁盘/谁在用）一条命令扫一台；
> 3. **病灶**：系统盘满的排查按命中率排序——`/root` 开发缓存 > docker > /var/log，**从来不是 /home**（LVM 独立分区）；
> 4. **备份**：tar + 服务器/本地双端 md5，不一致 = 没备份；
> 5. **迁移**：按数据性质分流（权重上 NFS 恢复用软链、代码带 env_manifest、/tmp 产物不备份），重装后跑精度门禁验收等价性。
>
> 六条踩坑垫在底下：/tmp 丢过仓库、818G 的别人目录只能找物主、NFS 防重装不防误删。

## 前提：共享机上的三条铁律

搞模型调优的人，机器是绕不开的副产品。我们的小机群是四台 910B（B2/B4 混合），多团队共享，/home 下几十个工号目录。这个前提下所有流程都从三条铁律长出来：

1. **接入走别名**——裸连（命令行甩 `-i` + IP + 一串 flags）迟早连错机器；别名制让"连哪台"成为一个显式、可审计的名字。
2. **先确认归属再动手**——共享机上最危险的动作是"看起来该清理的目录"；删错工号目录就是事故。
3. **备份必做双端校验**——tar 打包成功 ≠ 备份成功，传输中断产生半截文件是常态；不校验 md5 的备份是薛定谔的备份。

## 1. 接入：别名制（2 分钟/台）

`~/.ssh/config` 一处管理：

```sshconfig
Host npu203              # 别名 = npu + IP 末段，一眼知道是哪台
    HostName 192.168.9.203
    User root
    IdentityFile ~/.ssh/id_ed25519_npu
    StrictHostKeyChecking no
    ServerAliveInterval 60
```

公钥分发遇到"只有密码"的机器，用 paramiko 一次性脚本（本机无 sshpass 时的替代），装完 `ssh npu203 hostname` 免密即验收通过。**别名规划有个小讲究**：用 IP 末段而不是用途命名——机器会被重装、易主、改用途，IP 末段是唯一不变量。

## 2. 审计：五件套一条命令

接手新机器、或者隔一阵盘点，一条命令扫完：

```bash
ssh npu203 '
hostname
lscpu | grep -E "^CPU\(s\)|Model name" | head -2
free -g | grep Mem
ls /dev/davinci* | wc -l | xargs echo "davinci:"
npu-smi info | grep -E "910B|910 " | head -8
df -h / /home | grep -v tmpfs
uptime
ps aux --sort=-%cpu | head -4 | tail -3
ls /usr/local/Ascend
'
```

读数三个要点：

- **NPU 型号决定容量预算**：910B2 = 64G HBM，910B4 = 32G——B4 跑 7B 全模型推理很紧张，排任务前先看型号；
- **`/` 与 `/home` 是 LVM 独立分区**：`df -h / /home` 必须分开看——这个知识点是后面病灶定位的前提；
- **`ps aux --sort=-%cpu` 看的是"谁在用"**：共享机上 119% CPU 跑了三天的 pytest 是别人的长任务——**动任何东西前先看归属**。

## 3. 病灶：系统盘满，按命中率排查

四台机器的实测经验，系统盘 93~100% 的病灶**从来不在 /home**（独立分区，物理上不可能）。按命中率排序的排查顺序：

```bash
du -xh -d1 / 2>/dev/null | sort -rh | head -6     # -x 锁定本文件系统，别误入挂载
du -h -d1 /root | sort -rh | head -6              # 命中率第一：开发缓存
du -h -d1 /var/lib/docker | sort -rh | head -3    # 第二：容器层
```

实测病灶分布表（openEuler）：

| 病灶 | 典型大小 | 处置 | 风险 |
|---|---|---|---|
| `/root/.{vscode,trae,cursor}-server` | 5-9G/个 | 删，下次用时重装 | 低 |
| `/root/.cache` + `.triton` + `.ccache` | 5-15G | 删（triton 首次编译变慢） | 低 |
| `/var/lib/docker/overlay2` | 40G+ | `docker system prune`，**先核对容器归属** | 中 |
| `/home/<工号>` 巨型目录 | 数百 G | **找物主**，不属于你的不要动 | 高 |

两个细节：docker 的数据目录**不一定是** `/var/lib/docker`——`docker info | grep "Docker Root Dir"` 先看真实位置（我们有一台实际在 `/home/docker`）；`du` 不带 `-x` 会误入 NFS 挂载点，扫出一堆不属于本机的数据。

## 4. 备份：双端校验，否则等于没备份

```bash
# 服务器侧打包（排除清单很重要：权重、conda、clone 的大仓、旧备份）
cd /home && tar --exclude='<user>/llama-2-7b' --exclude='<user>/envs' \
  -czf /tmp/<user>_backup_$(date +%Y%m%d).tar.gz <user>/

# 拉到本地 + 双端 md5
scp npu203:/tmp/<user>_backup_*.tar.gz "<本地备份目录>/"
md5sum  # 服务器侧 vs certutil -hashfile 本地侧 —— 不一致 = 没备份成功

# /tmp 清理：它会被系统清，别放长期数据
rm -f /tmp/<user>_backup_*.tar.gz
```

保留策略：最近 1~2 个日期版本，更早的确认后删。**排除清单和备份清单同样重要**——模型权重（可从 NFS 恢复）、conda 环境（可重建）、clone 的大仓（可重新 clone）进备份只会让 tar 包涨到无法传输，而真正要保的是代码、文档、证据、配置。

## 5. 迁移：按数据性质分流

重装/换机前，数据按"重装后还值不值钱"分流到 NFS 共享盘：

| 数据 | 去向 | 理由 |
|---|---|---|
| 模型权重 | `/mnt/weight/<user>/` | 大文件只读资产；恢复时**软链不复制**，省空间省时间 |
| 代码/文档/证据 | `/mnt/share/<user>/{docs,scripts,evidence}/` | `pip freeze` 必存——环境清单是重装后的施工图 |
| conda env 整包（可选） | `/mnt/share/<user>/build_env.tar.gz` | `conda pack`；**前提是重装后 CANN 版本一致**，跨版本 pack 是废纸 |
| 不备份 | /tmp 产物、compile_debug、可重新 clone 的仓 | 重装即弃 |

三条时间纪律：

1. **重装前 3 天完成备份 + 恢复演练**（解包抽查关键文件）——不要拖到最后一天，演练才能暴露"备份里缺了什么"；
2. **NFS 是共享的**——它防重装，不防误删；关键证据恢复后在 /home 落一份工作副本；
3. **恢复验收用门禁说话**：环境重建后跑精度门禁（g12/g13，见[前文](https://mol9026.github.io/tech-blog/posts/env-gate-g13/)）全 PASS，才可以宣告"与重装前等价"——"能 import torch" 和 "编译行为等价" 之间隔着整个版本组合问题。

## 六条踩坑（每条有事故背景）

1. **/tmp 会清数据**：PR 工作仓库放 /tmp 丢过一次（好在分支推上了远端）——仓库类一律 /home 或 /mnt；
2. **818G 的别人目录**：盘点时遇到的巨型工号目录，唯一正确动作是找物主——共享机上"顺手清理"是事故之源；
3. **docker 目录漂移**：`/var/lib/docker` 与 `/home/docker` 都见过，先查 Docker Root Dir；
4. **系统盘满 ≠ /home 满**：LVM 独立分区，病灶三巨头是 IDE 缓存、编译缓存、docker；
5. **NFS 防重装不防误删**：双写原则——关键证据 NFS 一份 + 机器本地一份；
6. **别人的长任务**：先 `ps aux` 看归属，再决定动不动——机器是共享的，节奏是别人的。

## 沉淀成流程的意义

这五段流程没有一段是高技术含量的——SSH config、du、tar、md5——但把它们固化成"每台 2 分钟接入 / 一条命令审计 / 命中率排序的病灶表 / 双端校验的备份 / 分流的迁移"之后，运维动作从"凭记忆和手感"变成"照清单执行"。两次全量备份、一次病灶定位、一次重装迁移，全程零流程修改——**清单被实战验证过不改版，才算立住了**。

对做模型的人来说，这套东西的真正价值是：**机器管得越省心，花在精度和性能上的注意力越多**。运维不是我们的主业，正因为不是主业，才更要用最少的脑力把它做对。

---

*系列关联：重装迁移后的环境等价性验收见[环境门禁 g13](https://mol9026.github.io/tech-blog/posts/env-gate-g13/)——门禁全 PASS 才算迁移完成。*
