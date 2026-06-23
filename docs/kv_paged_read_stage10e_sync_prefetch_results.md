# Stage 10-E：WikiText-2 corpus-backed 多会话场景下的同步预取结果

## 1. 本阶段目标

Stage 10-E 的目标是在 Stage 10-C/D 已经完成的 **WikiText-2 corpus-backed semi-real multi-session workload** 基础上，继续优化 idle KV swap-out / resume swap-in 带来的恢复延迟。

具体目标包括：

1. 在不引入异步线程的前提下，优化同步预取策略；
2. 减少 request resume 时的 first-token latency；
3. 保持 idle KV swap + madvise 带来的 RSS 下降；
4. 识别性能回退来源，区分 trace 开销和机制本身开销；
5. 得到一个稳定、低日志、可作为当前提交版本的同步方案。

本阶段最终验证的方案为：

```text
S5_final12_no_core_trace
```

即：

```text
tail/lazy reclaim 开启；
paged KV 开启；
idle swap-out + madvise 开启；
during-active incremental prefetch 开启；
final limited sync prefetch 开启；
resume defer 开启；
KV_PAGED_TRACE 关闭；
KV_PAGED_IDLE_TRACE 关闭。
```

---

## 2. 实验背景

前序 Stage 10-C/D 已经证明：

```text
在 WikiText-2 真实文本驱动的半真实多会话 workload 下，
idle KV swap + prefetch + defer 可以稳定降低进程 RSS，
但性能代价明显高于 Stage 9 固定 prompt workload。
```

Stage 10-D-C 中，当前最优的 incremental prefetch 参数为：

```text
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

对应配置简称：

```text
S5_e2_b2_defer
```

Stage 10-D-C 3-run median 结果为：

| case             | RSS drop vs S0 | active TPS delta | resume delta |
| ---------------- | -------------: | ---------------: | -----------: |
| `S5_e2_b2_defer` |    363.176 MiB |         -17.945% |   +27.564 ms |

该结果说明：

```text
RSS 收益成立，但 resume first-token latency 仍然偏高。
```

因此 Stage 10-E 进一步引入 **final limited sync prefetch**，即在 request 真正 resume 前，对该 seq 进行有限数量的同步预取，尽量降低 resume 首 token 等待时间。

---

## 3. Workload 与实验设置

### 3.1 Workload 类型

本阶段使用：

```text
WikiText-2 corpus-backed semi-real multi-session workload
```

含义是：

1. prompt 内容来自 WikiText-2 真实文本语料；
2. A/B/C/D 四个 session 的 prompt 从真实文本中切分得到；
3. 多会话生命周期、pause/resume 时间线、target decode token 数仍由 driver 构造；
4. 因此该 workload 不是纯 synthetic prompt，但也不是完整真实 server workload。

更准确的中文表述为：

```text
真实语料驱动的半真实多会话工作负载。
```

不应表述为：

```text
真实线上服务 workload；
完整真实请求流；
真实 server benchmark。
```

### 3.2 模型与运行参数

模型：

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

语料：

```text
wikitext-2-raw/wiki.test.raw
```

driver：

```text
build/bin/llama-kv-semi-real-multisession
```

通用运行参数：

```text
--ctx-size 2048
--batch-size 512
--ubatch-size 128
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 4
--seed 1
--temp 0
```

corpus env：

```text
LLAMA_KV_SEMI_CORPUS_FILE=wikitext-2-raw/wiki.test.raw
LLAMA_KV_SEMI_CORPUS_CHARS=1024
LLAMA_KV_SEMI_CORPUS_OFFSET=0
```

corpus token 分布稳定为：

```text
A_tokens=232
B_tokens=257
C_tokens=232
D_tokens=246
```

### 3.3 对比配置

本阶段主要对比：

| case                       | 含义                                                       |
| -------------------------- | -------------------------------------------------------- |
| `S0_baseline`              | 不启用 tail/lazy，不启用 paged idle swap                        |
| `S2_tail_lazy`             | 仅启用 tail/lazy reclaim                                    |
| `S5_*`                     | tail/lazy + paged idle swap + madvise + prefetch + defer |
| `S5_final12_no_core_trace` | 最终低日志完整同步方案                                              |

---

## 4. Stage 10-E-A：实现 final limited sync prefetch

### 4.1 修改目标

新增 env：

```text
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS
```

语义：

```text
默认 0，关闭 final limited sync prefetch；
>0 时，在 RESUME_PENDING session 真正 resume decode 前，对该 seq 最多同步恢复 N 个 block；
<0 回退 0 并 warning。
```

### 4.2 修改范围

修改文件：

```text
examples/kv-semi-real-multisession/semi-real-multisession.cpp
```

没有修改：

```text
src/
include/
common/
CMakeLists.txt
scripts/
docs/
public API
```

### 4.3 关键逻辑

在 `decode_some()` 中，当 session 处于：

```text
RESUME_PENDING
```

并即将切换到 resume decode 前，执行：

```text
llama_memory_prefetch_seq_step(...)
```

最多恢复：

```text
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS
```

个 block。

新增 telemetry：

```text
KV_SEMI_PREFETCH_FINAL phase=<N> seq=<N> requested=<N> restored=<N> calls=<N> blocks=<N> rss_kb=<N>
```

---

## 5. Stage 10-E-B：final limited sync prefetch 参数对比

Stage 10-E-B 对以下配置进行 3-run median 对比：

| case                     | 说明                                        |
| ------------------------ | ----------------------------------------- |
| `S0_baseline`            | baseline                                  |
| `S2_tail_lazy`           | tail/lazy only                            |
| `S5_e2_b2_final0`        | incremental prefetch + defer，无 final sync |
| `S5_e2_b2_final4`        | final sync 最多 4 blocks                    |
| `S5_e2_b2_final8`        | final sync 最多 8 blocks                    |
| `S5_e2_b2_final12`       | final sync 最多 12 blocks                   |
| `S5_sync_prefetch_defer` | full sync prefetch 对照                     |

### 5.1 3-run median 结果

| case                     | RSS drop vs S0 | active TPS delta | resume delta |   total wall |
| ------------------------ | -------------: | ---------------: | -----------: | -----------: |
| `S5_e2_b2_final0`        |    363.441 MiB |         -17.456% |   +28.193 ms | 18545.670 ms |
| `S5_e2_b2_final4`        |    363.520 MiB |         -17.288% |   +24.694 ms | 18491.657 ms |
| `S5_e2_b2_final8`        |    363.559 MiB |         -16.989% |   +21.142 ms | 18424.827 ms |
| `S5_e2_b2_final12`       |    363.539 MiB |         -16.621% |   +20.711 ms | 18429.768 ms |
| `S5_sync_prefetch_defer` |    363.625 MiB |         -17.550% |   +22.936 ms | 18510.016 ms |

### 5.2 结论

`S5_e2_b2_final12` 是当前同步预取参数中最优的稳定配置。

相比 `final0`：

```text
resume delta: +28.193 ms → +20.711 ms
TPS delta:    -17.456%  → -16.621%
```

说明：

```text
final limited sync prefetch 能明显降低 resume first-token latency；
同时基本不损害 RSS 收益。
```

但此时实验仍启用了部分 core trace，因此还需要进一步区分性能代价中有多少来自 trace 输出。

---

## 6. Stage 10-E-C/D：发现 IDLE_TRACE 与 idle swap 执行耦合

### 6.1 问题现象

Stage 10-E-C 尝试关闭 core trace 后发现：

| case                          |    RSS drop | idle swap 是否工作 |
| ----------------------------- | ----------: | -------------- |
| `S5_final12_full_trace`       | 362.633 MiB | 工作             |
| `S5_final12_idle_trace_only`  | 362.809 MiB | 工作             |
| `S5_final12_paged_trace_only` | 194.141 MiB | 不工作            |
| `S5_final12_no_core_trace`    | 194.117 MiB | 不工作            |

关键现象：

```text
关闭 LLAMA_KV_PAGED_IDLE_TRACE 后，
RSS drop 从约 363 MiB 降到约 194 MiB。
```

这说明：

```text
LLAMA_KV_PAGED_IDLE_TRACE 不只是日志开关；
它同时参与了 idle swap execution。
```

因此当时的 `no_core_trace` 不是低日志完整 S5，而是 idle swap 没有执行的退化版本。

### 6.2 源码审计结论

审计确认两处耦合：

第一处：`idle_swap_ready` 依赖 `paged_idle_trace_enabled`。

原逻辑等价于：

```text
idle_swap_ready =
    paged_idle_swap_requested &&
    paged_swap_enabled &&
    kv_swap_store &&
    nonidentity_probe &&
    paged_idle_trace_enabled
```

第二处：idle maintenance 主循环外层使用：

```text
if (paged_idle_trace_enabled) {
    ...
}
```

该 block 包住了：

```text
idle block analysis；
safe swap candidate 计算；
swap-out；
madvise；
相关统计更新。
```

因此 `LLAMA_KV_PAGED_IDLE_TRACE=0` 时，idle maintenance 整体不执行。

### 6.3 判断

这不是性能问题，而是功能/日志开关耦合问题。

在解决该问题前，无法测得：

```text
完整 S5 功能 + 关闭 core trace 打印
```

的真实性能。

---

## 7. Stage 10-E-F：解耦 idle swap execution 与 idle trace printing

### 7.1 修改目标

目标是拆分：

```text
功能开关：
  LLAMA_KV_PAGED_IDLE_SWAP=1
  LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

打印开关：
  LLAMA_KV_PAGED_IDLE_TRACE=1
```

解耦后应满足：

```text
IDLE_SWAP=1, IDLE_TRACE=0:
  idle swap / madvise 仍然执行，但不打印 KV_PAGED_IDLE_TRACE。

IDLE_SWAP=0, IDLE_TRACE=1:
  保留诊断打印模式，但不执行 idle swap。

IDLE_SWAP=1, IDLE_TRACE=1:
  保留原 full trace 行为。
```

### 7.2 修改范围

修改文件：

```text
src/llama-kv-cache.cpp
```

diff stat：

```text
src/llama-kv-cache.cpp | 22 +++++++++++++++++-----
1 file changed, 17 insertions(+), 5 deletions(-)
```

未修改：

```text
examples/
common/
include/
CMakeLists.txt
scripts/
docs/
public API
```

### 7.3 关键修改

第一，去掉 `idle_swap_ready` 对 `paged_idle_trace_enabled` 的依赖：

```text
idle_swap_ready =
    paged_idle_swap_requested &&
    paged_swap_enabled &&
    kv_swap_store &&
    nonidentity_probe
```

第二，引入 idle maintenance 执行 gate：

```text
idle_maintenance_active =
    paged_idle_trace_enabled || paged_idle_swap_requested
```

第三，仅将逐步 `KV_PAGED_IDLE_TRACE` 的 `fprintf` 放入：

```text
if (paged_idle_trace_enabled) {
    ...
}
```

而 idle analysis、safe candidate、swap-out、madvise、计数器更新仍留在执行路径中。

### 7.4 smoke 结果

解耦后 smoke 结果：

| case                         | RSS drop vs S0 | KV_PAGED_TRACE lines | KV_PAGED_IDLE_TRACE lines | 结论                 |
| ---------------------------- | -------------: | -------------------: | ------------------------: | ------------------ |
| `S5_final12_full_trace`      |    363.277 MiB |                   64 |                        64 | full trace 正常      |
| `S5_final12_idle_trace_only` |    363.387 MiB |                    0 |                        64 | idle trace only 正常 |
| `S5_final12_no_core_trace`   |    363.398 MiB |                    0 |                         0 | 低日志完整 S5 成功        |

核心结果：

```text
解耦前：
  no_core_trace RSS drop ≈ 194 MiB

解耦后：
  no_core_trace RSS drop ≈ 363 MiB
```

说明：

```text
idle swap / madvise 执行已经不再依赖 KV_PAGED_IDLE_TRACE 打印。
```

### 7.5 诊断模式说明

另测：

```text
S5_diag_idle_trace_no_swap
```

该模式下：

```text
LLAMA_KV_PAGED_IDLE_TRACE=1
LLAMA_KV_PAGED_IDLE_SWAP=0
```

结果显示：

```text
paged_idle_swap_enabled=0
paged_idle_swap_out_calls=0
```

说明 idle swap 本身未执行。

但该模式仍触发了其他 paged swap / redirect 相关统计，出现大量异常计数字段，并导致性能显著下降。因此该模式不作为最终性能路径，仅作为诊断边界记录。

最终方案不依赖该模式。

---

## 8. Stage 10-E-G：低日志完整 S5 3-run median

### 8.1 最终测试配置

最终低日志同步方案：

```text
S5_final12_no_core_trace
```

env 组合：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1

LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

显式关闭或不设置：

```text
LLAMA_KV_PAGED_TRACE
LLAMA_KV_PAGED_IDLE_TRACE
```

### 8.2 3-run 通过性

| case                       | exit_sum | real_abnormal_sum | all_finished_sum |
| -------------------------- | -------: | ----------------: | ---------------: |
| `S0_baseline`              |        0 |                 0 |                3 |
| `S2_tail_lazy`             |        0 |                 0 |                3 |
| `S5_final12_no_core_trace` |        0 |                 0 |                3 |

全部 9 次运行正常退出，无 real abnormal，四个 session 均完成。

### 8.3 3-run median 结果

| case                       |   RSS KiB | RSS drop vs S0 | active TPS | TPS delta vs S0 | resume avg | resume delta vs S0 |
| -------------------------- | --------: | -------------: | ---------: | --------------: | ---------: | -----------------: |
| `S0_baseline`              | 8,714,616 |      0.000 MiB |     12.545 |          0.000% |  82.045 ms |           0.000 ms |
| `S2_tail_lazy`             | 8,518,560 |    191.461 MiB |     12.476 |         -0.552% |  81.614 ms |          -0.431 ms |
| `S5_final12_no_core_trace` | 8,342,716 |    363.184 MiB |     10.199 |        -18.706% | 102.242 ms |         +20.198 ms |

### 8.4 低日志验证

| case                       | KV_PAGED_TRACE lines | KV_PAGED_IDLE_TRACE lines |
| -------------------------- | -------------------: | ------------------------: |
| `S0_baseline`              |                    0 |                         0 |
| `S2_tail_lazy`             |                    0 |                         0 |
| `S5_final12_no_core_trace` |                    0 |                         0 |

说明：

```text
最终 S5 结果不是依赖大量逐步 trace 打印得到的；
它是在关闭 KV_PAGED_TRACE / KV_PAGED_IDLE_TRACE 的低日志模式下得到的。
```

### 8.5 prefetch 执行情况

`S5_final12_no_core_trace` median：

```text
step_lines = 9
step_blocks = 18
final_lines = 2
final_calls = 9
final_blocks = 12
final_restored_sum = 12
```

说明：

```text
during-active incremental prefetch 与 final limited sync prefetch 均生效。
```

---

## 9. 关键结论

### 9.1 内存收益成立

最终低日志同步方案实现：

```text
RSS drop = 363.184 MiB
```

相对 baseline：

```text
8,714,616 KiB → 8,342,716 KiB
```

说明 idle KV swap + madvise 在真实语料驱动的半真实多会话 workload 下确实能降低进程 current RSS。

### 9.2 final limited sync prefetch 有效

从 Stage 10-D-C 到 Stage 10-E，resume 延迟明显改善。

对比：

| 阶段           | 配置                         | resume delta |
| ------------ | -------------------------- | -----------: |
| Stage 10-D-C | `S5_e2_b2_defer`           |   +27.564 ms |
| Stage 10-E-G | `S5_final12_no_core_trace` |   +20.198 ms |

说明：

```text
final limited sync prefetch 可以降低 resume first-token latency。
```

### 9.3 low-trace 完整 S5 跑通

解耦后，关闭：

```text
LLAMA_KV_PAGED_TRACE
LLAMA_KV_PAGED_IDLE_TRACE
```

仍能得到：

```text
RSS drop ≈ 363 MiB
```

说明：

```text
idle swap execution 与 idle trace printing 已经成功解耦。
```

这是 Stage 10-E 的一个关键工程修复点。

### 9.4 性能代价仍然偏大

最终低日志 S5：

```text
active TPS delta = -18.706%
resume delta = +20.198 ms
```

说明：

```text
性能回退主要来自 swap-out / madvise / prefetch / swap-in 机制本身，
而不是 core trace 打印。
```

因此不能声称该同步方案“几乎无损”。

更准确的口径是：

```text
当前同步方案以约 18.7% active TPS 回退和约 20.2 ms resume 首 token 额外延迟，
换取约 363.2 MiB 进程 RSS 下降。
```

---

## 10. 当前可提交方案定位

当前同步方案已经可以作为比赛的稳定提交版本，理由是：

1. 功能闭环完整；
2. RSS 下降稳定；
3. 使用真实文本语料；
4. 低日志模式已验证；
5. correctness smoke 通过；
6. 实现不引入异步线程，风险相对可控。

但提交时需要明确其代价：

```text
active TPS 下降约 18.7%；
resume first-token latency 增加约 20.2 ms。
```

该结果更适合描述为：

```text
运行时 KV 内存换性能机制。
```

不应描述为：

```text
无损优化；
几乎无性能下降；
真实 server workload 全面验证。
```

---

## 11. 与异步预取的关系

Stage 10-E 已经说明：

```text
同步预取可以降低 resume latency；
但 active TPS 回退仍然偏大。
```

由于低日志后 TPS 仍下降约 18.7%，下一步如果要继续降低性能代价，应进入异步预取方向。

异步预取的预期目标是：

```text
把部分 KV restore 工作从主推理路径移到后台线程；
在 request resume 前尽量提前完成 swap-in；
减少 active decode 被前台恢复阻塞的时间。
```

但异步预取会引入新的风险：

```text
线程安全；
KV 状态一致性；
后台 worker 生命周期；
和主推理线程的锁竞争；
context 销毁时的安全退出；
偶发竞态问题。
```

因此建议 Stage 11 先做：

```text
async prefetch feasibility audit
```

而不是直接实现异步线程。

---

## 12. 推荐最终表述

对外汇报可使用以下表述：

```text
我们在 llama.cpp 中实现了面向多会话 idle KV cache 的运行时 swap-out / madvise / resume swap-in 机制，并进一步加入 during-active incremental prefetch、resume defer 和 final limited sync prefetch。 在 WikiText-2 真实语料驱动的半真实多会话 workload 下，最终低日志同步方案在关闭 KV_PAGED_TRACE 与 KV_PAGED_IDLE_TRACE 的情况下，实现约 363.2 MiB 的进程 RSS 下降；所有 3-run 实验均正常完成，无 real abnormal。代价是 active TPS 下降约 18.7%，resume first-token 平均额外延迟约 20.2 ms。该结果证明运行时 idle KV 换出可以真实降低进程内存占用，同时也表明同步恢复路径仍存在明显性能代价，后续可通过异步预取进一步优化。
```

---

## 13. 后续工作

建议后续路线：

1. 将当前同步方案作为稳定版本提交；
2. 补充总体 `swap + prefetch` 设计说明文档；
3. 新开异步预取分支；
4. 先审计 `llama_memory_prefetch_seq_step` 等函数的线程安全；
5. 再实现最小 async worker 原型；
6. 使用同一 corpus-backed semi-real workload 对比同步和异步方案。

当前推荐分支策略：

```text
当前分支：
  kv-paged-stage4c-idle-telemetry
  作为同步稳定主线。

新分支：
  kv-paged-stage11-async-prefetch
  用于异步预取实验。
```
