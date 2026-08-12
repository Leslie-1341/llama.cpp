# 当前整体框架设计：Dense Flex、MoE Buffer、KV Cache 与全局 Memory Governor 协同

> 日期：2026-08-10
> 分支：`merge-test`
> 用途：用于论文/答辩配图前的系统架构梳理。
> 证据口径：本文基于当前工作区源码、当前 diff、`AGENTS.md` 与 `CLAUDE.md` 项目约束整理。凡未运行 benchmark 或未被测试闭环验证的内容，均按“已实现但证据不足”或“目标设计/尚未闭合”标注。

---

## 0. 一句话结论

当前分支已经形成一个以 **Server Memory Governor** 为中心的统一内存治理框架：

- **Dense Flex** 负责 dense 权重的显式层级流式加载、ring 驻留、balanced locking、运行期 clean reclaim、ring resize 与 delta re-pin。
- **MoE Buffer** 负责 MoE expert 权重的匿名 expert slice 缓存、LRU/热 group 驻留、预测预取、动态预算、sidecar/低比特与 fused expert 计算路径。
- **KV Cache / KV Memory Backend** 负责 KV resident/reclaimable 观测、pressure 下 release/offload，以及在 core backend 不可用时通过 slot-state offload fallback 保存 sequence state 并清理 slot。
- **Server Memory Governor** 统一采样 cgroup/RSS/KV pressure、Dense/MoE/KV stats，构造 reclaim/prefetch/grow 候选，按 pressure state、ROI、headroom、cooldown 与 reallocation credit 进行仲裁，并通过统一 observation marker 输出证据。

需要特别注意：**当前已经具备 Dense/MoE/KV 协同治理骨架和部分动作路径，但 KV lifecycle 的完整统一协议仍未完全实现**。`docs/kv_pressure_scheduler_contract.md` 明确标注其为目标契约，当前源码尚未完全满足其中关于 unified action request、block event、transaction terminal、active-required prefetch 等要求。

---

## 1. 总体框架分层

### 1.1 系统分层视图

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         llama-server / Server Scheduler              │
│                                                                     │
│  update_slots()                                                     │
│      │                                                              │
│      ▼                                                              │
│  maybe_sample_kv_pressure()                                         │
│      │                                                              │
│      ▼                                                              │
│  publish_memory_governor_observation()                              │
│      │                                                              │
│      ├─ 采样 cgroup/RSS/KV pressure                                  │
│      ├─ 采样 Dense Flex / MoE Buffer / KV stats                       │
│      ├─ 计算 effective pressure state                                 │
│      ├─ 构造 reclaim / prefetch / grow candidates                     │
│      ├─ ROI auction + pressure gate + cooldown                        │
│      ├─ unified prefetch budget                                       │
│      ├─ clean reclaim / KV release / KV offload                       │
│      ├─ MoE dynamic budget / Dense ring resize / delta re-pin          │
│      └─ 输出 memory_governor_observe marker                           │
└─────────────────────────────────────────────────────────────────────┘
              │                         │                         │
              ▼                         ▼                         ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│      Dense Flex       │      │      MoE Buffer       │      │       KV Cache        │
│ layer ring streaming  │      │ expert slice cache    │      │ resident/offload      │
│ balanced locking      │      │ LRU / hot pin         │      │ release/offload       │
│ delta re-pin          │      │ CLG/EAM/CCT prefetch  │      │ slot-state fallback   │
│ clean reclaim         │      │ dynamic budget        │      │ soft budget           │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
              ▲                         ▲                         ▲
              └────────────── stats / candidates / action results ─┘
```

### 1.2 两阶段治理

当前框架分为 **加载期规划** 与 **运行期调度** 两个阶段。

#### 阶段 A：加载期 memory planner / auto backends

加载期主要在模型加载过程中决定使用哪种权重管理后端，并根据 cgroup memory.max、模型大小、KV reserve、non-expert floor 等信息生成初始预算。

核心职责：

1. 检测是否处于 memory governor / cgroup 限制环境。
2. 判断模型是 dense 还是 MoE。
3. 对 dense 模型倾向启用 **Dense Flex**。
4. 对 MoE 模型倾向启用 **lazy window + MoE Buffer**。
5. 在预算中预留 KV Cache 与运行时临时内存。
6. 为 Dense Flex 规划 ring/ahead/lock；为 MoE Buffer 规划 expert budget / warm working set。

相关源码：

- `src/llama-model.cpp`：模型加载期 planner、Dense/MoE 后端选择、budget plan。
- `src/llama-model-loader.cpp` / `src/llama-model-loader.h`：loader 记录主模型路径，供 MoE sidecar 自动推断使用。
- `src/llama-flex.*`：Dense Flex 创建、注册、finalize。
- `src/llama-moe-buffer.*`：MoE expert tensor 注册、budget plan、warm working set 估算。

#### 阶段 A 的具体代码实现导读

加载期的关键入口是 `src/llama-model.cpp` 中的 `llama_model_base::load_tensors(ml)`。这一段不是单纯创建 tensor，而是在创建 backend buffer、初始化 mmap/lazy/flex 之前做了一次全局内存判定：

1. **读取 cgroup 与环境开关**：
   - `llama_read_cgroup_memory_snapshot()` 读取 cgroup `memory.max/current`。
   - `LLAMA_MEMORY_GOVERNOR` 或 cgroup 有效上限会使 `memory_governor_requested` 为真。
   - `LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS` 控制是否自动选择 Dense/MoE 后端。
2. **估算模型与 KV reserve**：
   - 遍历 `ml.weights_map` 得到 `model_weight_bytes` 与 `largest_weight_tensor_bytes`。
   - 遍历 `hparams.n_layer` 按 `n_head_kv * (head_k + head_v) * sizeof(uint16_t)` 估算 `governor_kv_bytes_per_token`。
   - 使用 `largest_weight_tensor_bytes + kv_bytes_per_token * n_layer` 形成 `governor_backend_reserve`。
3. **区分 Dense 与 MoE 模型**：
   - `model_has_moe = hparams.n_expert > 0`。
   - MoE 模型在 auto backend 下默认走 `lazy_v2/window + MoE Buffer`。
   - Dense 模型在 auto backend 下默认请求 `llama-flex`。
4. **互斥选择权重后端**：
   - `use_flex` 要求 `flex_requested && !use_mlock && !ml.check_tensors && ml.use_mmap && !vocab_only`。
   - `use_lazy_window` 显式排除 `use_flex`，因此 Dense Flex 与 lazy mmap window 互斥。
   - 当 `use_lazy_window || use_flex` 时，`make_cpu_buft_list()` 禁用 `params.use_extra_bufts`，避免额外 backend buffer 与用户态 streaming 后端冲突。
5. **初始化映射策略**：
   - `ml.init_mappings(!(use_lazy_window || use_flex), ...)` 表示 Flex/lazy 模式不做传统 eager mmap prefetch。
   - `ml.load_all_data(..., use_lazy_window)` 在 lazy window 场景允许仅按 mmap/window 需要处理；MoE Buffer 会接管 expert tensor。
6. **MoE Buffer 注册和预算规划**：
   - `use_lazy_window` 块中解析 `LLAMA_LAZY_MOE_*` 系列环境变量，构造 `llama_moe_buffer_params`。
   - `LLAMA_LAZY_MOE_SIDECAR` 未设置时，通过 `llama_auto_moe_sidecar_path(ml.fname_model)` 从主模型路径推断 sidecar。
   - 遍历 `ml.weights_map`，识别 3D 且名称包含 `_exps` 的 expert tensor。
   - 对 expert tensor 调用 `llama_moe_buffer_register()`，成功后 `continue`，跳过 lazy window region，即 expert 权重由匿名 buffer 管理。
   - auto budget 下调用 `llama_moe_buffer_expert_bytes()`、`llama_moe_buffer_warm_working_set_bytes()`，再用 non-expert floor、KV reserve、hard guard、working-set floor 推出 safe budget，最后 `llama_moe_buffer_set_budget_plan()`。
7. **Dense Flex 注册和预算规划**：
   - Dense path 由 `llama_flex_params` 承载 ring/ahead/io_threads/lock_bytes/read_cost 等规划结果。
   - 创建 `llama_flex_context` 后，将 layer tensor 元数据注册进去，最后 `llama_flex_finalize()` 计算 slot size、分配 ring 并启动 IO worker。

这部分适合画成“加载期 Planner”配图：左侧输入是 cgroup/model/KV/env，中央是 Dense/MoE 分流和 budget planner，右侧输出分别是 `Flex context` 与 `MoE Buffer context`。

#### 阶段 B：运行期 server memory governor

运行期由 server scheduler 的单线程 update path 周期触发。它既是观测层，也是策略层和部分动作触发层。

核心职责：

1. 采样 pressure：cgroup memory.current/max/high、RSS、KV pressure runtime。
2. 收集模块状态：Dense Flex stats、MoE Buffer stats、KV budget / slot-state budget。
3. 计算 effective pressure state：`NORMAL / PRESSURE / CRITICAL`。
4. 构造候选动作：
   - Dense clean reclaim / prefetch / grow / re-pin；
   - MoE clean reclaim / prefetch / grow / budget shrink；
   - KV global release / sequence offload / resume accounting；
5. 进行 ROI/auction 选择和 pressure gate。
6. 下发统一 speculative prefetch budget。
7. 在 pressure 下触发 clean reclaim、KV release/offload 或 slot-state offload。
8. 在 normal/headroom 足够时执行 MoE budget grow 或 Dense delta re-pin。
9. 输出 `memory_governor_observe` marker，供实验脚本和 parser 归因。

相关源码：

- `tools/server/server-context.cpp`：Server memory governor 主体。
- `tests/test-memory-governor-observe-static.py`：静态 schema/token presence 测试。
- `scripts/profile-global-memory-sweep.sh`：未跟踪的 sweep/profile 脚本，用于提取 marker 与 cgroup 结果。

---

## 2. 模块一：Dense Flex 设计

### 2.1 模块目标

Dense Flex 是 dense 权重侧的显式驻留控制模块。它的目标不是依赖 mmap page cache 与 `madvise` hint，而是：

1. 不把全部 dense 权重常驻在 RSS/page cache 中。
2. 按 decoder layer 组织权重。
3. 用一个小型 per-layer ring buffer 保存当前和未来若干层。
4. 后台 IO 线程从 GGUF 文件 direct/sequential `pread()` 层权重。
5. compute 消费完一层后释放该层 ring slot。
6. 通过 balanced locking / delta pin 保留高价值 tensor，减少每 token 重复读取。
7. 向 Server Memory Governor 暴露 resident、reclaimable、stream-per-token、effective-ahead、prefetch dropped 等统计。

头文件注释已经明确其设计：Dense Flex 不再 mmap weights，而是维护小 ring 的 per-layer host buffers，使权重 RSS 主要受 ring size 约束。

### 2.2 核心数据结构

#### `llama_flex_tensor`

语义：描述一个被 Flex 管理的 tensor 在模型文件和 buffer 中的位置。

关键字段：

- `name`：tensor 名称，供 callback 根据 ggml op 找回对应权重。
- `file_idx` / `file_offset` / `size`：原始 GGUF/split 文件定位信息。
- `buf_offset`：tensor 在 ring slot 或 lock buffer 中的位置。
- `locked`：加载期 balanced locking 后永久驻留。
- `delta_locked` / `delta_buf_index` / `delta_buf_offset`：运行期 delta re-pin 后进入 delta lock buffer。

#### `llama_flex_params`

语义：Dense Flex 的加载期和运行期行为配置。

关键字段：

- `enabled` / `direct_io` / `debug_log`：基础开关。
- `sched_auto` / `planner_applied`：是否由 cap-aware planner 自动规划。
- `adaptive_ahead` / `ring_layers` / `prefetch_ahead` / `prefetch_ahead_max`：ring 和预取窗口。
- `io_threads`：后台流式读取线程数。
- `lock_bytes`：balanced locking 预算。
- `memory_budget_bytes` / `fixed_bytes`：总预算和不可 Flex 化的固定预留。
- `read_cost_bytes` / `read_cost_auto`：将读开销转化为 pinning 价值。
- `global_rebalance`：跨层使用 leftover pin budget。
- `pin_policy`：`small-first`、`large-first`、`attn-first`、`ffn-first`、`cost-aware`、`none` 等。

#### `llama_flex_stats`

语义：向 server governor 和实验 marker 暴露 Dense Flex 运行状态。

关键字段类别：

- IO 与等待：`layer_loads`、`layer_hits`、`wait_events`、`bytes_streamed`、`bytes_read_phys`、`read_ops`、`total_io_us`、`total_wait_us`、`ewma_io_us`、`ewma_compute_us`、`ewma_wait_us`。
- 预取：`demand_loads`、`prefetch_queued`、`prefetch_budget_dropped`、`effective_ahead`。
- 驻留：`ring_bytes`、`slot_bytes`、`locked_bytes`、`delta_locked_bytes`、`stream_per_token`。
- re-pin：`delta_pin_attempts`、`delta_pin_failures`、`delta_pin_saved_per_token`、`delta_pin_async_*`。
- 全局预算：`prefetch_budget_bytes`、`prefetch_budget_available_bytes`。

### 2.3 内部机制

#### 2.3.1 Layer ring streaming

Dense Flex 的基础访问流程：

```text
llama_flex_graph_begin()
    ├─ reset per-graph state
    └─ request initial layers [0, ahead]

llama_flex_stream_callback(op, ith)
    ├─ 识别 op 所属 layer
    ├─ request / wait 当前 layer
    ├─ 将 tensor->data 指向 ring / lock / delta buffer
    ├─ 在 layer boundary release 前一层
    └─ prefetch 后续层
```

关键设计点：

- `ith == 0` 负责 stream 与 repoint，CPU backend barrier 保证其他 worker 看到一致数据。
- 当前 compute layer 不会被 reclaim。
- ring slot 优先复用 free slot；必要时回收已 `RELEASED` 层。
- demand load 不受 speculative prefetch budget 阻塞。

#### 2.3.2 Balanced locking

Balanced locking 是加载期优化：在给定 `lock_bytes` 预算内，把部分高价值 tensor 永久放入 lock buffer，避免每 token 重复流式读取。

候选排序可以按：

- 小 tensor 优先；
- 大 tensor 优先；
- attention tensor 优先；
- FFN tensor 优先；
- cost-aware；
- none。

其目标不是扩大 ring，而是降低每 token 必须读取的 unlocked bytes，即降低 `stream_per_token`。

#### 2.3.3 Adaptive ahead

Dense Flex 根据 IO 等待和 compute EWMA 动态调节预取 ahead：

- 如果 compute 经常等待 IO，可增加 ahead；
- 如果 ring 空间紧张或等待不明显，可降低 ahead；
- effective ahead 还受 ring room 限制。

#### 2.3.4 Clean reclaim

`llama_flex_reclaim_released()` 只回收已经被 `llama_flex_release_layer()` 标记为 consumed/released 的 ring slot。

性质：

- 不触碰当前 compute layer。
- 不触发 demand load。
- 通过 `MADV_DONTNEED` 释放已释放层的匿名/显式 buffer 驻留页。
- 适合作为 pressure 下低风险回收动作。

#### 2.3.5 Runtime ring resize

`llama_flex_resize_ring()` 支持运行时调整 ring slot 数量：

- grow：立即增加 slot；
- shrink：只移除 free slot，不能使 active/resident layer 失效；
- 如果 slot busy，返回 `busy_slots` 等 reason。

该能力允许 server governor 在 pressure 下缩小 Dense resident footprint，在 normal/headroom 足够时适当扩大 prefetch window。

#### 2.3.6 Delta pin / runtime re-pin

Delta pin 是运行期预算再投资机制：

1. Server governor 根据 headroom、reallocation credit、ROI 与 idle 状态决定是否触发。
2. Dense Flex 选择高价值 tensor，读入 delta lock buffer。
3. 后续 token 对这些 tensor 不再走 ring streaming。
4. 返回 `saved_per_token_bytes`、`pinned_bytes`、`roi`、`reason`。

这适合作为论文中的“回收收益再投资到权重常驻优化”的例子。

### 2.4 与全局 Governor 的协作

Dense Flex 向 Server Memory Governor 提供：

- resident 信息：`ring_bytes + locked_bytes + delta_locked_bytes`；
- reclaimable 信息：released layer slots；
- IO 压力：`stream_per_token`、`ewma_wait_us`、`demand_loads`；
- prefetch 信息：`effective_ahead`、`prefetch_budget_dropped`；
- re-pin 结果：`delta_pin_*`。

Server Governor 对 Dense Flex 可执行：

1. `llama_flex_set_prefetch_budget()`：下发本 tick speculative prefetch budget。
2. `llama_flex_reclaim_released()`：pressure 下清洁回收 released slots。
3. `llama_flex_resize_ring()`：运行期 ring shrink/grow。
4. `llama_flex_delta_pin()` / `llama_flex_delta_pin_async()`：normal/headroom 足够时运行期 re-pin。

### 2.5 Dense Flex 具体代码实现导读

#### 2.5.1 对外接口层：`src/llama-flex.h`

Dense Flex 的头文件已经把模块边界拆得很清楚：

- `llama_flex_create()`：创建 streaming context，内部会复制/重开文件描述符，必要时尝试 O_DIRECT。
- `llama_flex_register_tensor()`：加载期把某个 tensor 绑定到 decoder layer，记录文件偏移、大小、layer 内 offset 和 pin 状态。
- `llama_flex_finalize()`：注册完成后计算每个 layer slot 的最大尺寸，分配 ring slot、lock buffer，启动后台 worker。
- `llama_flex_request_layer()` / `llama_flex_wait_layer()` / `llama_flex_get_tensor()` / `llama_flex_release_layer()`：构成“请求层 → 等待常驻 → 获取 tensor 指针 → 释放层”的最小闭环。
- `llama_flex_set_prefetch_budget()`：由 server 下发每 tick speculative prefetch 预算。
- `llama_flex_reclaim_released()`：只回收已经 release 的 resident layer。
- `llama_flex_resize_ring()`：运行期调整 ring slots。
- `llama_flex_delta_pin()` / `llama_flex_delta_pin_async()`：运行期按 ROI 将 tensor 固定到 delta lock buffer。
- `llama_flex_graph_begin()` / `llama_flex_stream_callback()`：接入 ggml CPU compute path。

#### 2.5.2 内部上下文：`llama_flex_context`

`src/llama-flex.cpp` 中 `llama_flex_context` 是 Dense Flex 的状态中心，核心字段可以按功能分为：

| 字段类别 | 代表字段 | 作用 |
|---|---|---|
| 文件与 layer 元数据 | `fds`、`file_sizes`、`layers`、`name_layer` | 定位每个 tensor 的原始 GGUF 数据和所属 layer |
| ring 驻留 | `slot_bytes`、`slots`、`slot_layer` | 维护 k 个 per-layer host buffer |
| pinned buffer | `lock_buf`、`lock_size`、`delta_lock_bufs` | 加载期 balanced lock 与运行期 delta pin |
| worker/queue | `workers`、`queue`、`cv_work`、`cv_ready` | 后台 IO 线程和 layer stream 队列 |
| 运行时图状态 | `cur_compute_layer`、`graph_id`、`last_layer_enter_us` | 每个 decode graph 的 layer 边界与统计 |
| 自适应控制 | `adaptive_ahead`、`ewma_io_us`、`ewma_compute_us`、`ewma_wait_us` | 调整 prefetch ahead |
| 统一 prefetch budget | `prefetch_budget_enabled`、`prefetch_budget_bytes`、`prefetch_budget_available_bytes` | 与 Server Governor 的预算接口 |

#### 2.5.3 IO worker 与 ring slot 管理

代码实现的关键路径是 `flex_worker()`：

```text
flex_worker()
  ├─ 等待 queue 中的 layer id
  ├─ 如果 layer 已 resident/loading，跳过
  ├─ flex_acquire_slot()
  │    ├─ 优先找 free slot
  │    └─ 否则选择 released + resident 的 LRU victim
  ├─ 标记 layer 为 loading
  ├─ 对 layer 中未 locked / 未 delta_locked 的 tensor 执行 flex_read()
  ├─ 成功：state = resident，更新 stats 与 EWMA
  └─ 失败：释放 slot，state 回到 not_resident，通知等待线程
```

`flex_read()` 根据是否启用 O_DIRECT 分为两种路径：

- 非 O_DIRECT：普通 `pread()` loop，按逻辑大小读取。
- O_DIRECT：将文件 offset 对齐到 block 边界，先读入 per-thread bounce buffer，再拷贝到 ring slot，统计物理读取字节 `bytes_read_phys`。

`flex_acquire_slot()` 的语义也很适合配图：slot 不是任意抢占，而是只在 free slot 或已 release 的 resident layer 中找 victim；如果没有安全 victim，则 requeue 并短暂 backoff。

#### 2.5.4 Request / wait / get / release 实现闭环

Dense Flex 的同步语义由四个函数闭合：

1. `flex_request_layer(ctx, layer_id, prefetch)`：
   - 如果已 resident，记 `layer_hits`。
   - 如果 speculative prefetch 且 budget 不足，增加 `prefetch_budget_dropped` 并直接 drop。
   - demand load 不经过该 budget drop 逻辑。
   - 将 layer id 放入 queue 并唤醒 worker。
2. `flex_wait_layer_us(ctx, layer_id)`：
   - 如果当前 layer 还没有 resident，则把它 push 到队首作为 demand load。
   - 等待 `cv_ready`，直到 state 变成 resident 或 shutdown。
   - 统计 `wait_events` 与 `total_wait_us`。
3. `llama_flex_get_tensor()`：
   - 若 tensor 是 `locked`，返回 `lock_buf + buf_offset`。
   - 若 tensor 是 `delta_locked`，返回对应 `delta_lock_buf`。
   - 否则要求 layer resident，并返回 `slots[L.slot] + buf_offset`。
4. `llama_flex_release_layer()`：
   - 将 layer 标记为 `released = true`，更新 `last_use`。
   - 后续 clean reclaim 或 slot eviction 只能安全处理这类 released layer。

#### 2.5.5 Compute path 接入

`src/llama-context.cpp` 在每次 graph compute 前读取 `model.get_flex_context()`：

```text
if flex_active:
    llama_flex_graph_begin(*flex)
    ggml_cpu_set_weight_stream_callback(llama_flex_stream_callback, flex)
    ggml_cpu_set_op_override_callback(nullptr, nullptr)
```

计算结束后调用 `ggml_backend_sched_synchronize()`，并清空 weight stream callback。这里有两个重要实现语义：

- Dense Flex 与 MoE Buffer 是互斥接入：`moe_active = !flex_active && llama_moe_buffer_enabled(moe)`。
- Flex path 只设置 weight-stream callback，不设置 op override；MoE path 同时设置 stream callback 与 op override。

#### 2.5.6 Runtime resize / reclaim / delta pin

Server Governor 对 Dense Flex 的运行期动作最终落到以下 API：

- `llama_flex_resize_ring()`：
  - grow 时分配新 slot 并追加到 `slots/slot_layer`。
  - shrink 时只删除 `slot_layer[i] < 0` 的 free slot；busy/resident slot 不会被强制删除。
  - 返回 `grown/shrunk/busy/unchanged/alloc_failed` 等 reason。
- `llama_flex_reclaim_released()`：
  - 对已经 release 的 slot 执行 clean reclaim。
  - 目标是释放物理页，不改变模型文件作为 authoritative source 的语义。
- `llama_flex_delta_pin()` / `llama_flex_delta_pin_async()`：
  - 根据预算和 `min_roi` 选择 tensor 候选。
  - 将其读入 delta lock buffer 并更新 `delta_locked` 元数据。
  - async 版本通过内部 delta pin queue/worker 执行，server marker 记录 submitted/completed/rejected/pending。

### 2.6 Dense Flex 当前证据状态

| 设计点 | 当前状态 |
|---|---|
| Layer ring streaming API 与 callback | 已实现，源码可见 |
| Balanced locking / pin policy | 已实现，源码可见 |
| Adaptive ahead | 已实现，源码可见 |
| Clean reclaim | 已实现，源码可见；静态测试约束 observe path 不触发 forbidden demand/prefetch |
| Runtime ring resize | 已实现，源码可见 |
| Delta pin / async re-pin | 已实现，源码可见 |
| 与 Server Governor 的 marker 字段 | 已实现，静态测试覆盖 token/schema |
| 性能收益、RSS 降幅、combined 正收益 | 本轮未运行 benchmark，证据不足 |
| 并发 compute/prefetch 下 ring shrink 与 delta pin 安全性 | 需要运行期压力测试补证 |

---

## 3. 模块二：MoE Buffer 设计

### 3.1 模块目标

MoE Buffer 是 MoE expert 权重侧的显式缓存与专家激活优化模块。它替代的是 mmap expert window/page cache 式管理，核心思想是：

1. 每个 `ffn_*_exps` expert tensor 分配 full-size anonymous buffer。
2. 将 `tensor->data` repoint 到匿名 buffer。
3. 仍保持原始 `expert_id * stride` 寻址，不需要改变 `mul_mat_id` kernel 的 id remap。
4. 只有 router 实际选中的 expert slice 会被 `pread()` 读入。
5. 冷 expert slot 维持 zero-fill/unbacked，不占物理内存。
6. resident expert groups 受 byte budget 控制，通过 LRU/Belady-like/热度/predictor 选择 victim。
7. eviction 通过匿名内存 `MADV_DONTNEED` 释放物理页。
8. CLG/EAM/CCT 等 predictor 提前预取未来专家，降低 synchronous miss。
9. sidecar/MWQ/低比特和 fused FFN/SWIGLU 路径进一步降低 expert 读入和计算开销。

### 3.2 核心数据结构

#### `llama_moe_buffer_params`

关键配置类别：

- 基础开关：`enabled`、`debug_log`、`direct_io`。
- 预取调度：`hebf_schedule`、`n_workers`。
- 动态低比特：`dynamic_bits`、`dynamic_bits_real`、`strict_sidecar`、`native_hot`。
- sidecar 与 bit policy：`sidecar_path`、`base_bits`、`hot_bits`、`warm_bits`、`cold_bits`、`fixed_bits`、`gate_bits`、`up_bits`、`down_bits`。
- fused compute：`fuse_gate_up`、`fuse_swiglu`、`fuse_direct_swiglu`、`fuse_expert_ffn`、`prefetch_down_with_swiglu`。
- ISA 优化：`avx512_q2`、`avx512_q2_dot`、`vnni_q2`、`vnni_q2_down`、`vnni_q2_swiglu`。
- 预算：`budget_bytes`、`budget_unbounded`、`planner_safe_budget_bytes`、`planner_floor_bytes`。
- 热 group 管理：`hot_ratio`、`pinned_fraction`、`pinned_layer_fraction`。
- eviction 保护：`active_window`、`group_cooldown_tokens`、`pin_refresh_interval`。
- pressure-adaptive 保护收缩：`pressure_adaptive`、`pressure_scale_pin`、`pressure_scale_layer_pin`、`pressure_scale_window`、`pressure_scale_cooldown`、`pressure_scale_spec_guard`、`pressure_soft_ratio`、`pressure_hard_ratio`、`pressure_pin_floor`。
- warm-start：`warm_coverage`。

#### `llama_moe_buffer_stats`

对外统计包括：

- 驻留与预算：`resident_bytes`、`budget_bytes`、`budget_unbounded`、`planner_safe_budget_bytes`、`planner_floor_bytes`、`expert_bytes`。
- 流式读：`streams`、`hits`、`bytes_read`。
- cache 质量：`cache_hits`、`cache_misses`、`evictions`。
- prefetch 质量：`prefetch_hits`、`prefetch_late`、`prefetch_unused`、`prefetch_budget_dropped`。
- unified prefetch budget：`prefetch_budget_bytes`、`prefetch_budget_available_bytes`。
- warm working set：`warm_working_set_bytes`、`warm_working_set_groups`、`warm_working_set_coverage`。

#### `moe_group_state`

以 `(layer, expert)` group 为核心记录 expert 驻留和预测状态：

- reuse 估计：`reuse_ema`、`inter_token_gap_ema`、`reuse_within_1/4/16`。
- eviction 反馈：`last_evicted_token_epoch`、`last_evicted_reason`、`bad_reload_score`、`ghost_reload_risk_ema`。
- demand async/admission：`demand_async_pending`、`demand_admission_*`。
- predictor：`eam_replace_*`、`cct_conf`、`next_token_conf`、`next_token_protect_until`。

#### `llama_moe_buffer_context`

内部上下文维护：

- `by_name`：tensor name 到 managed expert tensor。
- `by_layer`：layer 到 expert tensors。
- `groups`：`(layer, expert)` 到 group state。
- `lru`：slice-level LRU。
- `group_lru`：group-level LRU。
- `resident_bytes`：当前实际 resident expert bytes。
- `expert_total`：所有 expert tensor 总 bytes。
- prefetch workers / queue / budget。
- EAM、CCT、next-token predictor 状态。
- trace 与大量 atomic stats。

### 3.3 内部机制

#### 3.3.1 Expert slice anonymous buffer

```text
GGUF / sidecar file
       │
       │ pread(selected expert slice)
       ▼
full-size anonymous expert tensor buffer
       │
       ├─ expert 0 slot
       ├─ expert 1 slot
       ├─ ...
       └─ expert N slot
       │
       ▼
GGML_OP_MUL_MAT_ID reads tensor->data + expert_id * stride
```

关键收益：

- 未访问 expert 不读入，不占物理页；
- 访问语义保持原 expert id layout；
- eviction 可以真正释放匿名页；
- 不依赖 kernel page cache 是否按预期回收。

#### 3.3.2 Group-level LRU 与 Belady-like eviction

MoE Buffer 不只做简单 slice LRU，而是以 `(layer, expert)` group 为主要 eviction 单元，避免 gate/up/down sibling tensor 分开抖动。

Victim 选择会综合：

- 是否 pinned hot group；
- 是否在 active window 或 cooldown 中；
- 是否有近未来预测使用；
- group LRU recency；
- bad reload / ghost feedback；
- reuse 估计；
- CCT confidence；
- layer locality；
- OLECAR online policy score。

#### 3.3.3 Hot group pin

MoE Buffer 会周期性刷新 hot group：

1. 按 group hot score 排序。
2. 受 `pinned_fraction` 总预算约束。
3. 受 `pinned_layer_fraction` 每层预算约束。
4. pinned group 在 eviction 中被保护。

#### 3.3.4 Warm working set estimator

新增 warm working set sizing 用于加载期和运行期预算规划：

1. 以 layer 为单位构造 expert group candidate。
2. group bytes 是同 layer 同 expert 的 gate/up/down 等 expert tensors stride 总和。
3. 如果已有访问统计，用 `seq_access/access/hot_score` 作为 prior。
4. 如果没有历史，使用 structural top-k/uniform prior。
5. 按 gain 排序，选择到给定 coverage。
6. 输出 warm working set bytes/groups/coverage。

这使 MoE budget 可以从“全 expert 常驻”转变为“覆盖高概率 expert working set”。

#### 3.3.5 CLG / EAM / CCT / next-token predictor

MoE Buffer 的预测体系包含多层：

- **CLG predictor**：从 window/路由历史预测下一层 expert，调用 `llama_moe_buffer_prefetch_ranked()`。
- **EAM transition**：学习 `P(next-layer expert | previous-layer expert)`。
- **CCT predictor**：branch-predictor 风格 saturating confidence，用于 prefetch/evict 信号。
- **same-layer next-token predictor**：retention-only，保护下一 token 可能复用的 resident group，不主动发 speculative read。

#### 3.3.6 Ranked prefetch 与 unified prefetch budget

MoE prefetch 接口：

- `llama_moe_buffer_prefetch()`：普通 expert list。
- `llama_moe_buffer_prefetch_ranked()`：带 rank/scores，用于优先级和 dynamic-bit accounting。

Server Governor 可以每 tick 调用 `llama_moe_buffer_set_prefetch_budget()`：

- budget > 0：speculative prefetch 消耗 budget；不足则 drop 并计数。
- budget = 0：关闭 gate。
- demand loads 不受此 API 限制。

#### 3.3.7 Pressure-adaptive expert cache protection

当前源码新增了 MoE pressure-adaptive 保护收缩机制。它不是直接改变 resident budget，而是在 MoE Buffer 内部根据 `resident_bytes / budget_bytes` 的压力比例，动态缩小“软保护”范围，使 eviction 在接近预算上限时更愿意释放本来受保护的 expert group。

核心配置位于 `llama_moe_buffer_params`：

- `pressure_adaptive`：总开关，默认开启。
- `pressure_soft_ratio`：resident/budget 超过该比例后开始收缩软保护，当前默认 `0.88`。
- `pressure_hard_ratio`：resident/budget 达到该比例后收缩到 floor，当前默认 `0.97`。
- `pressure_scale_pin`：缩小全局 pinned fraction。
- `pressure_scale_layer_pin`：缩小单层 pinned cap。
- `pressure_scale_window`：缩小 Belady-style active/future-use window。
- `pressure_scale_cooldown`：缩小 recent-use cooldown tokens。
- `pressure_scale_spec_guard`：降低 speculative-unused keep penalty。
- `pressure_pin_floor`：hard pressure 下 pinned fraction 的下限。

核心实现函数位于 `src/llama-moe-buffer.cpp`：

```text
moe_pressure_ratio_locked()
  └─ resident_bytes / budget_bytes

moe_pressure_scale_locked()
  └─ ratio <= soft → 0
  └─ ratio >= hard → 1
  └─ otherwise linear interpolation

moe_effective_pinned_fraction_locked()
moe_effective_pinned_layer_fraction_locked()
moe_effective_active_window_locked()
moe_effective_group_cooldown_tokens_locked()
````

这些 effective 值已经接入：

- future/active/recent 保护判断；
- hot group pin refresh；
- eviction protect result；
- speculative-unused penalty；
- debug stats 输出中的 `pressure={...}`。

配图建议：在 MoE Buffer eviction selector 前增加一个 **Pressure Adaptive Guard** 小模块，输入为 `resident_bytes / budget_bytes`，输出为 effective pin fraction、layer cap、active window、cooldown 和 speculative guard penalty。

证据边界：该机制当前有源码实现和 debug stats 输出；但是否能减少高压下 eviction 抖动、提升 TPOT 或降低 OOM，需要使用 MoE pressure sweep / ablation 实验补证。

#### 3.3.8 Demand async / admission

在 gate op 已知本层 routed experts 后，MoE Buffer 可以提前为 sibling tensors 做 demand async：

1. 统计本 expert 还需要哪些 tensor bytes。
2. bounded budget 下先做 admission / victim selection。
3. 必要时提前 eviction。
4. 将 demand group task 入队并等待完成。

这降低 gate/up/down 串行缺页式加载带来的 tail latency。

#### 3.3.9 Sidecar / dynamic bits / fused expert FFN

MoE Buffer 还提供专家激活优化路径：

- sidecar/MWQ 低比特 expert 数据源；
- 根据 rank/hotness 选择 hot/warm/cold bits；
- `strict_sidecar` 控制 sidecar 缺失是否 abort；
- gate/up pairing；
- fused SWIGLU；
- direct SWIGLU；
- fused expert FFN/down projection；
- AVX512/VNNI q2 path。

这些路径如果作为论文创新点，需要补充 correctness/精度对照实验，不能仅凭代码路径声称无损或性能收益。

### 3.4 与全局 Governor 的协作

MoE Buffer 向 Server Governor 提供：

- resident/budget/expert_total；
- evictions、bytes_read、cache hits/misses；
- prefetch hits/late/unused/dropped；
- warm working set；
- planner floor/safe budget。

Server Governor 对 MoE Buffer 可执行：

1. `llama_moe_buffer_set_budget()`：动态 grow/shrink resident budget。
2. `llama_moe_buffer_reclaim_clean()`：pressure 下回收 cold clean expert groups。
3. `llama_moe_buffer_set_prefetch_budget()`：限制 speculative expert prefetch。
4. reallocation credit：将 clean/KV 回收确认得到的 credit 再投资给 MoE working set。

### 3.5 MoE Buffer 具体代码实现导读

#### 3.5.1 对外接口层：`src/llama-moe-buffer.h`

MoE Buffer 的 API 边界可以分成五类：

1. **创建与启用**：`llama_moe_buffer_create()`、`llama_moe_buffer_enabled()`。
2. **加载期注册与预算**：`llama_moe_buffer_register()`、`llama_moe_buffer_expert_bytes()`、`llama_moe_buffer_warm_working_set_bytes()`、`llama_moe_buffer_set_budget_plan()`。
3. **运行期预算调整**：`llama_moe_buffer_set_budget()`。
4. **compute/prefetch 接入**：`llama_moe_buffer_stream_callback()`、`llama_moe_buffer_mul_mat_id_callback()`、`llama_moe_buffer_prefetch()`、`llama_moe_buffer_prefetch_ranked()`。
5. **观测与回收**：`llama_moe_buffer_get_stats()`、`llama_moe_buffer_reclaim_clean()`、`llama_moe_buffer_set_prefetch_budget()`。

这套接口使 MoE Buffer 同时承担“专家权重缓存”和“专家激活优化”的角色，但对 Server Governor 暴露的是统一的 budget/stats/reclaim/prefetch 接口。

#### 3.5.2 加载期注册：expert tensor repoint

加载期入口在 `src/llama-model.cpp` 的 lazy window 分支：

```text
use_lazy_window
  ├─ parse LLAMA_LAZY_MOE_* env
  ├─ sidecar_path = env 或 llama_auto_moe_sidecar_path(ml.fname_model)
  ├─ pimpl->moe_buffer = llama_moe_buffer_create(mp)
  ├─ 遍历 ml.weights_map
  │    ├─ is_exps = 3D tensor && name contains "_exps"
  │    ├─ llama_moe_buffer_register(...)
  │    └─ register 成功后 continue，跳过 window region
  └─ auto budget: expert_bytes / warm_working_set / safe_budget / set_budget_plan
```

`llama_moe_buffer_register()` 的核心语义是：为每个 `*_exps` tensor 分配 full-size anonymous buffer，重定向 `tensor->data`，并记录每个 expert slice 的文件偏移和 stride。因此后续 `mul_mat_id` 仍按原来的 `expert_id * stride` 访问，不需要改变专家 id 语义。

#### 3.5.3 内部上下文：`llama_moe_buffer_context`

`src/llama-moe-buffer.cpp` 的 `llama_moe_buffer_context` 字段很多，建议按论文配图抽象为七个子状态区：

| 子状态区 | 代表字段 | 作用 |
|---|---|---|
| tensor 管理 | `by_name`、`by_layer`、`groups` | 从 ggml tensor/layer/expert 找到 managed slice 与 group |
| 文件和 sidecar | `fds`、`sidecar_fd`、`sidecar_bits_mask`、`sidecar` | 原始 GGUF 与低比特 sidecar 数据源 |
| anonymous residency | `resident_bytes`、`expert_total`、`warm_working_set_bytes` | resident expert 页与总 expert 上界 |
| eviction 队列 | `lru`、`group_lru` | slice-level 与 group-level LRU |
| async prefetch | `workers`、`queue`、`prefetch_budget_*` | 后台 expert slice 读入与 budget gate |
| predictor | `eam_layer_transition`、`cct_transition`、`next_token_last_layer_experts`、`eamc_snapshots` | EAM/CCT/next-token/EAMC 预测与保留 |
| fused/sidecar compute | `pending_*`、`fused_*`、`direct_ensure_map`、`qx_cache_map`、`ffn_pipe_map` | gate/up pairing、direct SWIGLU、FFN pipeline 和 qx cache |

#### 3.5.4 Group state 与 eviction feedback

MoE Buffer 的核心不是单个 tensor slice，而是 `(layer, expert)` group。`moe_group_state` 中维护了多类在线反馈：

- cache/prefetch 统计：`seq_prefetch_hits`、`seq_prefetch_late`、`seq_cache_hits`、`seq_cache_misses`。
- reuse 估计：`reuse_ema`、`inter_token_gap_ema`、`reuse_within_1/4/16`。
- eviction 后悔信号：`bad_reload_score`、`ghost_reload_risk_ema`、`bad_reload_1/4/16`。
- demand admission：`demand_async_pending`、`demand_admission_*`、`admission_feedback_ema`、`admission_regret_ema`。
- predictor 信号：`eam_replace_*`、`cct_conf`、`next_token_conf`、`next_token_protect_until`。

Victim 选择时可以把这些信号组合为“是否应该继续保护这个 group”的分数。配图时可把它画成：LRU recency + future prediction + reload regret + hot pin + budget pressure → victim selector。

#### 3.5.5 Warm working set 与 budget plan

MoE auto budget 使用三层约束：

1. **不可回收 floor**：non-expert bytes、最大 tensor、KV reserve、当前 cgroup current/headroom guard。
2. **expert 上限**：`llama_moe_buffer_expert_bytes()` 返回所有 expert tensor 总量。
3. **工作集目标**：`llama_moe_buffer_warm_working_set_bytes(ctx, coverage)` 按 coverage 估算常用 expert group 集合。

加载期 planner 会在有限 `memory.max` 下计算：

```text
safe_budget = min(memory.max - hard_floor, expert_bytes)
working_set_floor = f(warm_working_set, min_expert_group, n_expert_used)
budget = min(warm_working_set, safe_budget)
```

如果 `safe_budget >= expert_bytes`，则 `budget_unbounded = true`，表示无需 eviction；否则通过 `llama_moe_buffer_set_budget_plan()` 明确写入 bounded budget、planner safe budget 与 planner floor，避免把 0 字节预算误解为 unbounded。

#### 3.5.6 Compute path：stream callback 与 op override

`src/llama-context.cpp` 在 MoE active 时设置两类 hook：

```text
if moe_active:
    ggml_cpu_set_weight_stream_callback(llama_moe_buffer_stream_callback, moe)
    ggml_cpu_set_op_override_callback(llama_moe_buffer_mul_mat_id_callback, moe)
```

两者分工：

- `llama_moe_buffer_stream_callback()`：在 managed `GGML_OP_MUL_MAT_ID` 执行前，确保本 op 路由到的 selected experts 已经 resident；miss 时同步或等待 async stream 完成。
- `llama_moe_buffer_mul_mat_id_callback()`：当 sidecar/MWQ/fused path 可用时，直接完成 managed op，绕过普通 kernel 的部分中间写入或使用低比特 expert slice。

由于 CPU backend 会在 weight-stream callback 返回 managed op 后设置 barrier，`ith == 0` 完成 expert residency 后，其他 worker thread 才会进入真正计算。

#### 3.5.7 Prefetch 与 demand async

MoE Buffer 的 prefetch 入口有两个：

- `llama_moe_buffer_prefetch()`：普通 expert list。
- `llama_moe_buffer_prefetch_ranked()`：带 rank 和 score，供 predictor 和 dynamic-bit policy 使用。

实现上，prefetch task 进入 `priority_queue<moe_prefetch_task>`，由多个 worker 并发 `pread()`。如果 `prefetch_budget_enabled`，speculative prefetch 会消耗 `prefetch_budget_available_bytes`，不足时 drop 并累计 `prefetch_budget_dropped`；但 demand miss/正确性必需加载不应被这个 speculative budget 阻断。

Demand async/admission 用于本层 gate 已知之后，为 sibling expert tensors 提前创建异步读取任务，同时在 bounded budget 下先评估是否需要 eviction。它更接近“确定性需求提前化”，而不是纯预测 prefetch。

#### 3.5.8 Clean reclaim 与 dynamic budget

Server Governor 的 MoE 动作主要通过两个 API：

- `llama_moe_buffer_reclaim_clean(ctx, target_bytes, max_groups)`：
  - 使用现有 victim selection 回收 cold resident group。
  - 对匿名 expert buffer 执行 `MADV_DONTNEED`，释放物理页。
  - 模型文件或 sidecar 仍是未来 reload 的权威数据源。
- `llama_moe_buffer_set_budget(ctx, budget_bytes)`：
  - 运行期 grow/shrink resident budget。
  - shrink 后可触发后续 reclaim；grow 可承接 reallocation credit 或 warm working set deficit。

### 3.6 MoE Buffer 当前证据状态

| 设计点 | 当前状态 |
|---|---|
| 匿名 full-size expert buffer 与 tensor repoint | 已实现，源码可见 |
| Expert slice streaming | 已实现，源码可见 |
| Group LRU / hot pin / eviction policy | 已实现，源码可见 |
| Pressure-adaptive soft protection shrink | 已实现，源码可见；运行收益未验证 |
| Warm working set estimator | 已实现，源码可见；静态测试仅检查字段存在 |
| CLG/EAM/CCT/next-token predictor 接口与状态 | 已实现，源码可见；运行效果未验证 |
| Ranked prefetch 与 prefetch budget | 已实现，源码可见 |
| Dynamic budget / budget_unbounded 语义 | 已实现，源码可见 |
| Sidecar/dynamic bits/fused compute | 已实现路径可见；需 correctness/精度/性能实验补证 |
| 与 Server Governor marker 字段 | 已实现，静态测试覆盖 token/schema |
| 命中率、eviction 质量、TPOT 改善 | 本轮未运行 benchmark，证据不足 |

---

## 4. 模块三：KV Cache / KV Memory Backend 设计

### 4.1 当前 KV 协作目标

KV Cache 是推理过程中随上下文增长而增长的核心 resident 内存。当前分支中，KV 与全局 governor 的协作目标包括：

1. 暴露 KV resident / reclaimable bytes。
2. 支持 pressure 下对全局可回收 KV block 做 release。
3. 支持按 sequence 粒度 offload idle KV。
4. 当 core memory backend 不支持 paged release/offload budget 时，使用 server slot-state fallback。
5. 将 KV resume 需要的预取/恢复使用量计入 unified prefetch budget。
6. 与 Dense/MoE 共同受 cgroup/headroom/pressure state 约束。

### 4.2 KV budget 来源

Server Governor 使用两类 KV budget 来源。

#### 来源 A：KV memory backend / paged release budget

优先调用：

```text
llama_get_memory(ctx_tgt)->sample_kv_release_budget()
```

如果有效：

- `kv_effective_budget_source = paged_release`
- 使用 backend 返回的 `resident_bytes`
- 使用 backend 返回的 `reclaimable_resident_bytes`

#### 来源 B：server slot-state fallback

如果 backend budget 无效，则遍历 server slots：

1. 用 `llama_state_seq_get_size_ext()` 估算每个 slot sequence state/KV bytes。
2. idle、非 shared、非 resume-protected 的 slot 计入 reclaimable。
3. 输出 slot-level resident/reclaimable/sequences。
4. `kv_effective_budget_source = slot_state`

这保证即使底层 KV backend 还不具备完整 paged release budget，server 仍能获得粗粒度 sequence offload fallback。

### 4.3 KV release

KV release 是 pressure 驱动的全局回收动作。

候选：

```text
kind   = kv_global
action = release
```

基本流程：

```text
pressure == PRESSURE / CRITICAL
    │
    ▼
构造 kv_global release candidate
    │
    ▼
llama_kv_action::evaluate
    │
    ├─ context_invalid / transaction_open / fail_stop / unsupported → skip/fail reason
    ▼
llama_kv_action::release
    │
    ▼
记录 blocks / bytes / relieved_bytes / shortfall / outcome / reason
```

### 4.4 KV offload

KV offload 是 sequence-scoped 的容量写回动作。

候选：

```text
kind   = kv_sequence
action = offload
id     = seq_id
```

设计约束：

- release 优先：如果 release disabled/cooldown 或 pressure 已被 release/clean reclaim 满足，则 offload 不应盲目执行。
- seq scoped：offload 针对选中的 sequence。
- 同步路径先 evaluate，再 execute offload。
- async 开启时提交到 memory governor async queue。
- IO class 使用 capacity write，体现这是为了容量释放而非普通后台写。

### 4.5 slot-state offload fallback

当 core KV backend budget 无效但 slot-level sequence state 可估算时，server 可使用 fallback：

```text
memory_governor_slot_state_offload(seq_id, target_bytes)
    ├─ 找 slot
    ├─ 拒绝 active / task 非空 / protected / empty prompt
    ├─ llama_state_seq_get_size_ext() 估算 before bytes
    ├─ 如果 prompt cache 未保存，保存 target/draft sequence state
    ├─ slot->prompt_clear(false)
    ├─ 再次测量 after bytes
    └─ 返回 relieved_bytes / reason
```

语义：它不是完整 KV lifecycle core offload，而是 server slot/prompt-cache 层面的 fallback，适合在配图中与正式 KV backend offload 区分表示。

### 4.6 KV soft budget

KV soft budget 用来保护 active/必要 KV，同时将 idle reclaimable KV 作为 pressure 下优先回收对象。

核心概念：

```text
kv_soft_protected_bytes = resident_bytes - reclaimable_bytes
kv_soft_target_bytes    = protected_bytes + configured_idle_bytes
kv_soft_excess_bytes    = max(0, resident_bytes - kv_soft_target_bytes)
```

release/offload target 会结合：

- pressure target；
- soft budget excess；
- max blocks；
- cooldown；
- effective pressure state。

### 4.7 KV 目标契约与当前实现差距

`docs/kv_pressure_scheduler_contract.md` 定义了更完整的目标协议：

```text
Server scheduler (policy only)
    │ logical action request / decision_id
    ▼
KV lifecycle core (state authority)
    │ logical→physical mapping
    │ ownership / transaction / quarantine checks
    │ backing IO + commit/rollback/fail-stop
    ▼
core action result + block events + transaction terminal
    ▼
server observation + runner observation + parser verdict
```

当前必须谨慎表述：

| 契约项 | 当前状态 |
|---|---|
| Server 只做策略，core 执行部分 release/offload | 部分实现 |
| `NOOP/EVALUATE/RELEASE/OFFLOAD/PREFETCH` 五种 action 统一 request | 尚未完全实现 |
| Active-required prefetch 优先级与 graph compute 前失败传播 | 尚未完全实现 |
| core result、server observation、runner observation 严格分层 | 尚未完全实现 |
| block event、transaction terminal、physical generation ID | 尚未完全实现 |
| quarantine / fail-closed / destructive recheck 完整闭环 | 尚未完全实现 |
| KV release/offload env gate、evaluate、execute_action、marker | 已实现部分路径 |
| slot-state offload fallback | 已实现路径；恢复语义需运行期补证 |

### 4.8 KV Cache 具体代码实现导读

#### 4.8.1 Server 侧 KV 观测入口

当前 KV 协作主要在 `tools/server/server-context.cpp` 的运行期 observation path 中完成。它不是单独由 KV core 主动调度，而是由 server 的周期性采样驱动：

```text
update_slots()
  └─ maybe_sample_kv_pressure(all_idle)
       ├─ pressure runtime sample
       └─ publish_memory_governor_observation()
            ├─ sample backend KV budget
            ├─ sample slot-state fallback budget
            ├─ compute kv soft budget
            ├─ evaluate release/offload candidates
            └─ execute or enqueue action
```

`init_kv_pressure_sampler()` 会先调用 `init_memory_governor_observer_from_env()`，当 `LLAMA_MEMORY_GOVERNOR_KV_RELEASE/OFFLOAD` 启用时，还会拒绝与 legacy `LLAMA_KV_PAGED_RELEASE=1` 同时启用，避免两个治理路径同时修改 KV 状态。

#### 4.8.2 Backend budget 与 slot-state fallback

KV budget 的代码实现上有两层：

1. **KV memory backend budget**：
   - 通过 `llama_get_memory(ctx_tgt)` 获取 memory backend。
   - 如果 backend 能返回有效 `sample_kv_release_budget()`，则直接使用其中的 resident/reclaimable 信息。
2. **slot-state fallback budget**：
   - 当 backend budget 无效时，server 遍历 slots。
   - 使用 `llama_state_seq_get_size_ext()` 估算每个 sequence state 的可迁移大小。
   - 排除 active、shared、resume-protected、任务非空或 prompt 不可安全保存的 slot。
   - 将结果作为 coarse-grained `kv_slot_resident_bytes` / `kv_slot_reclaimable_resident_bytes`。

配图时建议把 backend budget 画成主路径，把 slot-state fallback 画成 server 侧旁路路径，并标注“fallback，不等价于完整 KV lifecycle core”。

#### 4.8.3 KV release 代码路径

KV release 的同步路径使用 `llama_kv_action::evaluate` 先做能力检查，再执行 release。典型判断包括：

- `decision_id` 是否一致；
- `context_invalid`；
- `write_transaction_open`；
- `fail_stop`；
- backend 是否 `can_release`；
- evaluate outcome/reason 是否允许继续。

执行 release 后记录：

- `bytes` / `blocks`；
- `relieved_bytes` / `shortfall_bytes`；
- `state_changed`；
- `io_failure` / `fail_stop` / `io_errno`；
- `outcome` / `action_reason` / `reason`。

这些字段最终进入 `memory_governor_observe` marker 的 `kv_release_*` 字段。

#### 4.8.4 KV offload 代码路径

KV offload 是 sequence-scoped action。当前实现中 selected candidate 会携带 `seq_id`、`score`、`bytes` 和 target。同步路径：

```text
selected kv_sequence candidate
  ├─ decision_id = next_kv_decision_id()
  ├─ mem->execute_action(evaluate, claimant=kv, io_class=capacity_write)
  ├─ 检查 context / transaction / fail_stop / can_offload
  └─ mem->execute_action(offload, seq_id, target, max_blocks, priority, target)
```

关键实现语义：

- `llama_kv_io_class::capacity_write` 明确该 IO 是为了释放容量。
- offload action 带 `seq_id`，不是全局盲目写回。
- async 开启时，server 可以将 `kv_sequence_offload` 提交到 memory governor async queue；marker 中 `kv_offload_outcome=enqueued`、`kv_offload_reason=async_enqueued/async_queue_full` 用于区分。

#### 4.8.5 Slot-state offload fallback 代码语义

slot-state fallback 的目标是在 backend offload 不可用时，尽可能释放 server slot 中可保存的 sequence/prompt 状态：

```text
memory_governor_slot_state_offload(seq_id, target_bytes)
  ├─ 找到对应 slot
  ├─ 检查 active/task/shared/protected 状态
  ├─ before = llama_state_seq_get_size_ext(...)
  ├─ 必要时保存 target/draft prompt cache
  ├─ slot->prompt_clear(false)
  ├─ after = llama_state_seq_get_size_ext(...)
  └─ relieved = before - after
```

需要注意：该路径更偏 server slot/prompt-cache 层面的 fallback。它能作为 pressure 下的工程兜底，但不能在论文中宣称为完整 KV block lifecycle offload，除非后续补齐恢复正确性和 transaction/fail-stop 证据。

#### 4.8.6 KV soft budget 与 unified prefetch accounting

KV soft budget 的实现把 KV resident 拆成：

```text
protected = resident - reclaimable
target    = protected + configured_idle
excess    = max(0, resident - target)
```

这样 release/offload target 不只看全局 pressure target，也看 idle KV 是否超过 soft target。与此同时，KV resume 或恢复相关的预算会通过 `memory_governor_prefetch_budget_reserve(bytes)` 计入 unified prefetch budget：

- reserve 成功：减少本 tick 可给 Dense/MoE 的 speculative budget。
- reserve 失败：记录 `prefetch_budget_kv_resume_overruns`。

这体现 KV 不只是被回收对象，也参与 prefetch/IO 预算竞争。

因此论文配图建议用：

- **实线**：当前已实现的 Server Governor → KV release/offload/slot fallback 协同路径。
- **虚线**：目标 KV lifecycle protocol 尚未完全闭合的路径。

---

## 5. 模块四：Server Memory Governor 设计

### 5.1 模块定位

Server Memory Governor 是当前框架的运行期协调中心。它不属于单一 Dense/MoE/KV 模块，而是跨模块策略层。

它的输入是：

- cgroup memory.current / memory.max / memory.high；
- RSS / pressure telemetry；
- KV pressure sample；
- Dense Flex stats；
- MoE Buffer stats；
- KV budget / slot-state budget；
- 历史 action result / cooldown / reallocation credit。

它的输出是：

- `memory_governor_observe` marker；
- Dense prefetch budget / clean reclaim / ring resize / delta re-pin；
- MoE prefetch budget / budget grow/shrink / clean reclaim；
- KV release / offload / slot-state offload；
- async action queue tasks；
- reallocation credit accounting。

### 5.2 关键状态字段

Server context 中的 memory governor 状态大致可分为：

#### 5.2.1 Observation 与 auto backend

- `memory_governor_auto_backends_enabled`
- `memory_governor_observe_enabled`
- `memory_governor_observe_interval_ms`

#### 5.2.2 Clean reclaim

- `memory_governor_clean_reclaim_enabled`
- `memory_governor_clean_reclaim_target_bytes`
- `memory_governor_clean_reclaim_max_objects`
- `memory_governor_clean_reclaim_ranked_enabled`
- `memory_governor_clean_reclaim_max_passes`

#### 5.2.3 KV release/offload

- `memory_governor_kv_release_enabled`
- `memory_governor_kv_release_target_bytes`
- `memory_governor_kv_release_max_blocks`
- `memory_governor_kv_release_cooldown_samples`
- `memory_governor_kv_offload_enabled`
- `memory_governor_kv_offload_target_bytes`
- `memory_governor_kv_offload_max_blocks`
- `memory_governor_kv_offload_cooldown_samples`

#### 5.2.4 KV soft budget

- `memory_governor_kv_soft_budget_enabled`
- `memory_governor_kv_soft_target_bytes`
- `memory_governor_kv_soft_idle_bytes`

#### 5.2.5 Unified prefetch budget

- `memory_governor_prefetch_budget_enabled`
- `memory_governor_prefetch_budget_auto`
- `memory_governor_prefetch_budget_runtime_auto`
- `memory_governor_prefetch_budget_bytes_per_tick`
- `memory_governor_prefetch_budget_min_bytes_per_tick`
- `memory_governor_prefetch_budget_max_bytes_per_tick`
- `memory_governor_prefetch_budget_headroom_bytes`
- `memory_governor_prefetch_budget_kv_resume_used_bytes`
- `memory_governor_prefetch_budget_kv_resume_overruns`

#### 5.2.6 Async action executor

Action kinds：

- `dense_clean_reclaim`
- `moe_clean_reclaim`
- `kv_global_release`
- `kv_sequence_offload`
- `kv_sequence_slot_state_offload`

队列排序优先级：

1. pressure state；
2. ROI；
3. score；
4. target bytes；
5. decision id。

#### 5.2.7 Global optimizer

- `memory_governor_global_optimizer_enabled`
- `memory_governor_global_hard_headroom_bytes`
- `memory_governor_global_roi_threshold`
- `memory_governor_global_moe_eviction_weight`
- `memory_governor_global_moe_read_weight`
- `memory_governor_global_moe_prefetch_weight`
- `memory_governor_global_kv_reclaim_weight`
- `memory_governor_global_kv_resident_weight`

#### 5.2.8 Reallocation credit

- `memory_governor_reallocation_enabled`
- `memory_governor_reallocation_apply_moe`
- `memory_governor_reallocation_confirm_enabled`
- `memory_governor_reallocation_credit_bytes`
- `memory_governor_reallocation_pending_bytes`
- `memory_governor_reallocation_credit_cap_bytes`
- `memory_governor_reallocation_max_grant_bytes`
- `memory_governor_reallocation_min_grant_bytes`
- `memory_governor_reallocation_hard_guard_bytes`
- `memory_governor_reallocation_decay`

#### 5.2.9 Dense runtime controls

- `memory_governor_dense_repin_enabled`
- `memory_governor_dense_repin_async_enabled`
- `memory_governor_dense_repin_idle_only`
- `memory_governor_dense_repin_cooldown_samples`
- `memory_governor_dense_repin_step_bytes`
- `memory_governor_dense_repin_max_bytes`
- `memory_governor_dense_runtime_ring_shrink_enabled`

#### 5.2.10 MoE dynamic budget controls

- `memory_governor_moe_budget_dynamic_enabled`
- `memory_governor_moe_budget_fast_start_enabled`
- `memory_governor_moe_budget_min_bytes`
- `memory_governor_moe_budget_warm_bytes`
- `memory_governor_moe_budget_max_bytes`
- `memory_governor_moe_budget_grow_bytes`
- `memory_governor_moe_budget_headroom_bytes`
- `memory_governor_moe_budget_pressure_shrink_pct`
- `memory_governor_moe_budget_pressure_shrink_max_bytes`
- `memory_governor_moe_budget_grow_samples`
- `memory_governor_moe_budget_cooldown_samples`

### 5.3 Effective pressure state

Governor 使用多种输入计算 effective pressure state：

```text
raw pressure telemetry
      │
      ├─ cgroup current/max/headroom
      ├─ hard headroom guard
      ├─ Dense Flex slot/reclaimable bytes
      ├─ async relieved pending bytes
      └─ pressure excess bytes
              │
              ▼
NORMAL / PRESSURE / CRITICAL
```

典型规则：

- headroom 小于 hard headroom 的一半 → `CRITICAL`；
- headroom 小于 hard headroom → `PRESSURE`；
- pressure excess 明显超过阈值 → 提升到 `PRESSURE/CRITICAL`；
- 否则为 `NORMAL`。

### 5.4 Candidate / ROI / auction

Governor 将模块动作抽象成统一 candidate：

```text
memory_governor_would_candidate:
    kind
    action
    id
    score
    roi
    bytes
    reason
```

候选类别：

| kind | action | 含义 |
|---|---|---|
| `dense_layer` | `reclaim_clean` | 回收 Dense 已 release layer slot |
| `dense_layer` | `prefetch` | Dense speculative layer prefetch |
| `dense_layer` | `grow` | Dense ring/growth/re-pin 方向 |
| `moe_expert` | `reclaim_clean` | 回收 MoE cold clean expert group |
| `moe_expert` | `prefetch` | MoE expert speculative prefetch |
| `moe_expert` | `grow` | 增长 MoE resident budget |
| `kv_global` | `release` | 全局 KV release |
| `kv_sequence` | `offload` | sequence-scoped KV offload |
| `kv_sequence` | `prefetch` | KV resume/resident 需求 accounting |

Pressure gate：

| state | 允许动作 |
|---|---|
| `CRITICAL` | 只允许 reclaim/release/offload 类回收动作 |
| `PRESSURE` | 允许 reclaim；prefetch 仅限必要 resume-protected 类需求 |
| `NORMAL` | 可允许 reclaim/prefetch/grow/re-pin |

Auction 输出：

- `auction_candidates`
- `auction_selected_allocation`
- `auction_allocation_reason`
- `auction_selected_reclaim`
- `auction_reclaim_reason`

### 5.5 Unified prefetch budget

统一 prefetch budget 的目标是避免 Dense/MoE/KV 同时进行 speculative prefetch，导致内存和 IO 突刺。

```text
prefetch_budget_tick_bytes
       │
       ├─ KV resume reserve / overrun accounting
       ├─ Dense Flex prefetch budget
       └─ MoE Buffer prefetch budget
```

原则：

- speculative prefetch 受 budget gate；
- demand/correctness-required load 不应被简单丢弃；
- pressure 下预算会按 divisor/headroom clamp；
- marker 记录：
  - `prefetch_budget_dense_bytes`
  - `prefetch_budget_moe_bytes`
  - `prefetch_budget_kv_resume_used_bytes`
  - `dense_prefetch_budget_dropped`
  - `moe_prefetch_budget_dropped`

### 5.6 Clean reclaim

Clean reclaim 是 pressure 下优先级较高、风险较低的动作集合：

- Dense：只回收已经 release 的 layer slot。
- MoE：只回收 cold clean expert group。
- KV：不属于 clean reclaim，而走 release/offload action path。

Clean reclaim 可以同步执行，也可以提交 async queue。

### 5.7 Reallocation credit

Reallocation credit 是“回收收益再投资”的闭环：

```text
clean reclaim / KV release / KV offload / MoE shrink
        │ relieved bytes
        ▼
pending credit
        │ optional confirm by cgroup current drop
        ▼
earned credit
        │ decay + cap + hard guard
        ▼
reinvest
    ├─ MoE budget grow / emergency working set
    └─ Dense prefetch window / delta re-pin accounting
```

设计意义：

- 避免刚释放出的内存立刻被不受控增长吞掉。
- 要求观察到 memory.current 下降或在 slack 内，才确认 credit。
- `CRITICAL` 下不给 credit，避免危险状态继续扩张。
- normal 下可以把回收收益投给高 ROI 模块。

### 5.8 Server Memory Governor 具体代码实现导读

#### 5.8.1 初始化：env gate 与默认 auto backend

`tools/server/server-context.cpp` 的 `init_memory_governor_observer_from_env()` 负责把环境变量转成 server context 内的控制字段。实现上有几个层次：

1. **基础开关**：
   - `LLAMA_MEMORY_GOVERNOR`：开启全局 memory governor。
   - `LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS`：允许按 cgroup 和模型类型自动启用 Dense/MoE/KV 后端。
   - `LLAMA_MEMORY_GOVERNOR_OBSERVE_MS`：设置 observe tick 间隔。
2. **回收类动作**：
   - `LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM` / `*_RANKED` / target/max_objects/max_passes。
   - `LLAMA_MEMORY_GOVERNOR_KV_RELEASE` / target/max_blocks/cooldown。
   - `LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD` / target/max_blocks/cooldown。
3. **KV soft budget**：
   - `LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET`。
   - `LLAMA_MEMORY_GOVERNOR_KV_SOFT_TARGET_MB`、`LLAMA_MEMORY_GOVERNOR_KV_SOFT_IDLE_MB`。
4. **Unified prefetch budget**：
   - `LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MB_PER_TICK`。
   - `LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO`。
   - min/max/headroom/pressure divisor。
5. **Async action executor**：
   - `LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS`。
   - `LLAMA_MEMORY_GOVERNOR_ASYNC_QUEUE_DEPTH`。
6. **Global optimizer 与 reallocation**：
   - `LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER`。
   - ROI/weight/headroom 参数。
   - `LLAMA_MEMORY_GOVERNOR_REALLOCATION` 及 credit cap/max grant/min grant/confirm/decay。
7. **Dense/MoE runtime controls**：
   - Dense re-pin、ring shrink。
   - MoE dynamic budget、fast start、pressure shrink、grow/cooldown。

初始化结束会打印一行配置日志，列出 clean/KV/prefetch/async/global/reallocation/MoE budget 等关键参数。该日志可用于确认实验环境是否真的启用了相应路径。

#### 5.8.2 Observation tick 主体

`publish_memory_governor_observation()` 是运行期中心。可以按以下顺序理解：

```text
publish_memory_governor_observation()
  ├─ 读取 pressure/cgroup/RSS/KV pressure runtime
  ├─ 获取 Dense Flex stats / MoE Buffer stats / window stats
  ├─ 采样 KV backend budget 或 slot-state budget
  ├─ 计算 pressure_excess 与 effective_pressure_state
  ├─ 计算 Dense/MoE/KV soft target、reclaimable、headroom
  ├─ 构造 reclaim / prefetch / grow candidates
  ├─ global optimizer 计算 Dense/MoE/KV/prefetch utility 与 ROI
  ├─ pressure gate + auction 选出 allocation/reclaim
  ├─ 执行或提交 clean reclaim / KV release / KV offload
  ├─ 更新 MoE dynamic budget / Dense ring resize / delta re-pin
  ├─ 计算 reallocation credit 并可能投给 MoE/Dense
  └─ 输出 memory_governor_observe marker
```

需要注意：当前代码中的注释可能仍保留“read-only observation marker”之类历史说法，但现状已经包含动作路径，例如 clean reclaim、KV release/offload、MoE budget set、Dense resize/re-pin 和 async action queue。因此文档和配图应以当前源码实际行为为准，而不是旧注释。

#### 5.8.3 Candidate 表示与 auction

Governor 把跨模块动作压缩成 `memory_governor_would_candidate`：

```text
kind:   dense_layer / moe_expert / kv_global / kv_sequence / ...
action: reclaim_clean / prefetch / grow / release / offload / ...
id:     layer id、expert group id 或 seq id
score:  整数效用分
roi:    单位收益
bytes:  预期释放或申请的字节数
reason: 生成/跳过原因
```

随后把 reclaim、prefetch、grow 三类 candidate 合并到 `auction_candidates`，分别用 `memory_governor_auction_select()` 选择：

- allocation candidate：偏 prefetch/grow/re-pin。
- reclaim candidate：偏 reclaim/release/offload。

Pressure gate 是关键约束：

- `CRITICAL`：只允许释放类动作。
- `PRESSURE`：以释放为主，限制 speculative prefetch/grow。
- `NORMAL`：允许增长、prefetch 和 re-pin，但仍受 headroom/ROI/cooldown 限制。

#### 5.8.4 Async action executor

当前 async action enum 覆盖：

- `dense_clean_reclaim`
- `moe_clean_reclaim`
- `kv_global_release`
- `kv_sequence_offload`
- `kv_sequence_slot_state_offload`

Async submit 会进入 priority queue，排序大致考虑 pressure state、ROI、score、target bytes、decision id。执行结果会累计到 async relieved/drained 字段，并在后续 observe tick 中进入 reallocation credit 的 raw relieved 来源。

配图建议把 async executor 画成 Server Governor 内部的“动作执行队列”，而不是独立模块；它服务于 Dense/MoE/KV 三类后端。

#### 5.8.5 Unified prefetch budget 下发

每个 observe tick 会计算有效 prefetch budget：

```text
configured/auto budget
  ├─ 根据 cgroup headroom clamp
  ├─ 根据 NORMAL/PRESSURE/CRITICAL divisor 调整
  ├─ 先扣 KV resume reserve
  ├─ 分配 Dense budget
  └─ 分配 MoE budget
```

下发点：

- Dense：`llama_flex_set_prefetch_budget(*flex, dense_budget)`。
- MoE：`llama_moe_buffer_set_prefetch_budget(*moe, moe_budget)`。
- KV：通过 `memory_governor_prefetch_budget_reserve()` 记录 resume 使用或 overrun。

这部分的工程意义是把“预取”从单模块局部优化变成全局受控资源，避免 Dense layer ahead、MoE expert prefetch 和 KV resume 同时制造 IO/RSS 峰值。

#### 5.8.6 Reallocation credit 的具体实现

代码中 reallocation 的 raw relieved 来源包括：

- Dense/MoE clean reclaim 的 `released_bytes`；
- `kv_release_result.relieved_bytes`；
- `kv_offload_result.relieved_bytes`；
- MoE budget shrink/reclaim；
- async action drained relieved bytes。

实现步骤：

1. 根据 pressure state 打折：`CRITICAL` 下 discount 为 0，`PRESSURE` 使用 pressure discount，`NORMAL` 使用 normal discount。
2. 将打折后的 relieved bytes 加入 pending credit，受 cap 限制。
3. 如果开启 confirm，则要求 `memory.current` 下降量加 slack 能覆盖 pending，才转成 earned credit。
4. 对已有 credit 做 decay，再加 earned credit。
5. 按策略投给：
   - MoE resident budget grow / emergency working set。
   - Dense prefetch window accounting。
   - Dense delta re-pin grant。

`llama_moe_buffer_set_budget()` 与 `llama_flex_delta_pin(_async)` 是最终把 credit 转成模块动作的关键 API。

#### 5.8.7 Marker 输出即实验证据总线

`memory_governor_observe` marker 是一行长字段输出，包含：

- planner：`planner_auto_enabled`、`planner_model_kind`、`planner_floor_bytes`、`planner_dynamic_budget_bytes`。
- pressure：`pressure_state`、`effective_pressure_state`、`pressure_current_bytes`、`pressure_excess_bytes`、`rss_observed_bytes`。
- Dense：ring/lock/delta pin/stream/read/resize/reclaim/prebudget。
- MoE：resident/budget/expert/cache/prefetch/warm working set/dynamic budget。
- KV：backend budget、slot budget、soft budget、release/offload result。
- global optimizer：utility、ROI、headroom barrier、selected candidates。
- reallocation：pending/earned/spent/grant/reason。
- async：queue depth、submitted/completed/rejected/dropped/drained bytes。

论文实验脚本应优先从 marker 中取字段进行归因，而不是只看最终延迟数字。延迟/RSS 是结果，marker 才能说明“为什么这个结果发生”。

---

## 6. 模块协同关系

### 6.1 Dense Flex 与 MoE Buffer：权重侧协同

| 维度 | Dense Flex | MoE Buffer |
|---|---|---|
| 管理对象 | dense/普通层权重 | MoE expert 权重 |
| 粒度 | layer slot / tensor pin | `(layer, expert)` group / expert slice |
| 数据源 | GGUF direct/sequential read | GGUF 或 sidecar expert slice pread |
| 常驻控制 | ring + lock + delta pin | budget + LRU + hot pin |
| 回收动作 | released layer slot clean reclaim | cold clean expert group reclaim |
| 预取 | layer ahead prefetch | CLG/EAM ranked expert prefetch |
| 预算接口 | `set_prefetch_budget`、`resize_ring`、`delta_pin` | `set_prefetch_budget`、`set_budget`、`reclaim_clean` |

二者没有直接互相调用，而是通过 Server Memory Governor 协作：

1. 统一上报 stats。
2. 统一进入 candidate auction。
3. 共享 speculative prefetch budget。
4. pressure 下共同贡献 clean reclaim。
5. normal 下按 ROI 竞争增长预算。

### 6.2 Weight side 与 KV Cache：内存竞争协同

权重侧和 KV Cache 争用同一个 cgroup/RSS headroom：

```text
Dense ring / lock / delta pin
MoE expert resident budget
KV resident blocks / sequence states
        │
        ▼
shared memory.max / memory.high / RSS headroom
```

典型冲突：

- Dense prefetch ahead 增大会占用更多 host memory 和 IO；
- MoE expert prefetch/grow 会增加 resident expert pages；
- KV Cache 随上下文和并发增长，占用不可忽略；
- KV offload/release 释放容量，但可能影响 resume latency 或 correctness path；
- speculative prefetch 与 KV resume 可能同时触发 IO 峰值。

Governor 的调解方式：

1. pressure 下优先 clean reclaim。
2. clean reclaim 不足时触发 KV release/offload 或 MoE budget shrink。
3. normal 下使用 reallocation credit 增长 MoE budget 或 Dense pin。
4. unified prefetch budget 限制 Dense/MoE/KV resume 的 speculative 竞争。
5. ROI/utility 将“释放多少内存”“减少多少未来 IO/等待”“是否接近 hard headroom”合并排序。

### 6.3 KV 与 MoE：expert resident vs KV resident

MoE 与 KV 的核心矛盾：

- MoE resident budget 越大，expert miss 越少，MoE 读 IO 越低；
- KV resident 越大，长上下文和 resume 越稳，offload/reload 越少；
- memory pressure 下二者都不能无限增长。

Governor 输入：

- MoE：`resident_bytes`、`budget_bytes`、`evictions`、`bytes_read`、`prefetch_late`、`warm_working_set_bytes`。
- KV：`resident_bytes`、`reclaimable_resident_bytes`、`soft_excess_bytes`、release/offload relieved bytes。

Governor 输出：

- MoE budget grow/shrink。
- MoE clean reclaim。
- KV release/offload。
- prefetch budget 切分。

### 6.4 Dense 与 KV：streaming window vs KV pressure

Dense Flex 与 KV 的主要矛盾：

- Dense ring/ahead 越大，层权重等待越少，但 resident footprint 越大；
- KV pressure 越高，越需要释放 headroom；
- ring shrink 可以快速降低 Dense footprint，但可能增加后续 streaming wait；
- delta pin 可以降低 per-token read，但会增加常驻内存。

Governor 策略：

- pressure 下 shrink/reclaim Dense ring released slots。
- normal/idle/headroom 足够时执行 delta re-pin。
- re-pin 需要 cooldown、idle-only、headroom guard 与 ROI gate。

---

## 7. 端到端调用链

### 7.1 加载期调用链

```text
llama_model_load_from_file
  └─ llama_model_loader(...)
       └─ 保存 fname_model，用于 sidecar path 推断

llama_model_base::load_tensors(ml)
  ├─ 读取 cgroup / memory governor env
  ├─ 统计模型权重、最大 tensor、KV reserve
  ├─ 判断 dense / MoE
  │
  ├─ MoE path
  │    ├─ use lazy window / MoE Buffer
  │    ├─ 创建 llama_moe_buffer_context
  │    ├─ 遍历 *_exps expert tensor
  │    ├─ llama_moe_buffer_register()
  │    ├─ warm_working_set_bytes()
  │    ├─ 计算 non_expert floor + KV reserve + safe budget
  │    └─ llama_moe_buffer_set_budget_plan()
  │
  └─ Dense path
       ├─ use Dense Flex
       ├─ 统计 per-layer bytes / non-layer bytes
       ├─ 计算 runtime guard + KV reserve
       ├─ planner 选择 ring / ahead / lock
       ├─ llama_flex_create()
       ├─ llama_flex_register_tensor()
       └─ llama_flex_finalize()
```

### 7.2 推理 compute 调用链

```text
llama_context graph compute
  ├─ 获取 model flex context / moe buffer context
  │
  ├─ if flex active
  │    ├─ llama_flex_graph_begin()
  │    └─ ggml_cpu_set_weight_stream_callback(llama_flex_stream_callback)
  │
  └─ else if moe active
       ├─ ggml_cpu_set_weight_stream_callback(llama_moe_buffer_stream_callback)
       └─ ggml_cpu_set_op_override_callback(llama_moe_buffer_mul_mat_id_callback)

CPU backend executes graph
  ├─ Dense Flex: layer request/wait/repoint/prefetch/release
  └─ MoE Buffer: selected expert ensure resident / optional fused override
```

### 7.3 运行期 governor 调用链

```text
server_context_impl::update_slots()
  └─ maybe_sample_kv_pressure(all_idle)
       ├─ sample pressure if due
       └─ publish_memory_governor_observation()
            ├─ sample KV budget or slot-state fallback
            ├─ sample Dense Flex stats
            ├─ sample MoE Buffer stats
            ├─ compute effective pressure
            ├─ compute Dense/MoE/KV candidates
            ├─ global optimizer / ROI auction
            ├─ MoE budget dynamic grow/shrink
            ├─ Dense ring shrink/grow / delta re-pin
            ├─ unified prefetch budget allocation
            ├─ clean reclaim
            ├─ KV release
            ├─ KV offload / slot-state offload
            ├─ reallocation credit accounting
            └─ emit memory_governor_observe marker
```

### 7.4 Async action 调用链

```text
publish_memory_governor_observation()
  └─ memory_governor_async_submit(action)
       └─ priority_queue
            └─ memory_governor_async_worker_loop()
                 └─ memory_governor_async_execute()
                      ├─ dense_clean_reclaim → llama_flex_reclaim_released()
                      ├─ moe_clean_reclaim   → llama_moe_buffer_reclaim_clean()
                      ├─ kv_global_release   → mem->execute_action(evaluate/release)
                      ├─ kv_sequence_offload → mem->execute_action(evaluate/offload)
                      └─ slot fallback       → memory_governor_slot_state_offload()
```

---

## 8. Observation marker 设计

`memory_governor_observe` 是当前框架中最重要的实验证据出口。它是单行 marker，包含 planner、pressure、Dense、MoE、KV、auction、prefetch、async、reallocation 和 action result 字段。

### 8.1 Dense 字段

代表字段：

- `dense_flex_enabled`
- `dense_resident_bytes`
- `dense_reclaimable_bytes`
- `dense_ring_bytes`
- `dense_slot_bytes`
- `dense_locked_bytes`
- `dense_delta_locked_bytes`
- `dense_stream_per_token_bytes`
- `dense_effective_ahead`
- `dense_prefetch_budget_dropped`
- `dense_repin_*`
- `dense_resize_*`

### 8.2 MoE 字段

代表字段：

- `moe_enabled`
- `moe_resident_bytes`
- `moe_budget_bytes`
- `moe_budget_unbounded`
- `moe_planner_safe_budget_bytes`
- `moe_planner_floor_bytes`
- `moe_expert_bytes`
- `moe_evictions`
- `moe_bytes_read`
- `moe_cache_hits`
- `moe_cache_misses`
- `moe_prefetch_hits`
- `moe_prefetch_late`
- `moe_prefetch_unused`
- `moe_budget_action`
- `moe_budget_reason`
- `moe_warm_working_set_bytes`
- `moe_warm_working_set_groups`
- `moe_prefetch_budget_dropped`

### 8.3 KV 字段

代表字段：

- `kv_memory_present`
- `kv_release_budget_valid`
- `kv_resident_bytes`
- `kv_reclaimable_resident_bytes`
- `kv_slot_budget_valid`
- `kv_slot_resident_bytes`
- `kv_slot_reclaimable_resident_bytes`
- `kv_effective_budget_source`
- `kv_effective_resident_bytes`
- `kv_effective_reclaimable_resident_bytes`
- `kv_soft_budget_enabled`
- `kv_soft_target_bytes`
- `kv_soft_excess_bytes`
- `kv_release_*`
- `kv_offload_*`

### 8.4 Global optimizer / reallocation 字段

代表字段：

- `global_optimizer_enabled`
- `global_optimizer_decision`
- `global_dense_utility`
- `global_dense_roi`
- `global_moe_utility`
- `global_moe_roi`
- `global_kv_utility`
- `global_prefetch_utility`
- `global_headroom_barrier`
- `reallocation_enabled`
- `reallocation_pending_added_bytes`
- `reallocation_credit_earned_bytes`
- `reallocation_credit_spent_bytes`
- `reallocation_moe_grant_bytes`
- `reallocation_dense_grant_bytes`
- `reallocation_reason`

### 8.5 Prefetch / auction / async 字段

代表字段：

- `auction_candidates`
- `auction_selected_allocation`
- `auction_selected_reclaim`
- `would_reclaim_candidates`
- `would_prefetch_candidates`
- `prefetch_budget_enabled`
- `prefetch_budget_tick_bytes`
- `prefetch_budget_dense_bytes`
- `prefetch_budget_moe_bytes`
- `prefetch_budget_kv_resume_used_bytes`
- `governor_async_actions_enabled`
- `governor_async_queue_depth`
- `governor_async_submitted`
- `governor_async_completed`
- `governor_async_rejected`
- `governor_async_dropped`

### 8.6 Memory behavior profiling 脚本

当前工作区新增了 `scripts/run-memory-behavior-profile.py` 和 `tests/test-memory-behavior-profile-parser.py`，用于把 server runtime marker 与 cgroup samples 汇总成证据文件。它与 `scripts/profile-global-memory-sweep.sh` 的定位不同：

- `profile-global-memory-sweep.sh`：偏批量 sweep，在不同 `memory.max/high` 下跑 dense/MoE case，输出 `summary.tsv`。
- `run-memory-behavior-profile.py`：偏单次或少量实验的 evidence capture，解析 `memory_governor_observe`、`kv_pressure_telemetry`、`kv_pressure_dry_run`、`kv_pressure_bounded_release`、`kv_pressure_unified_action` 等 marker，同时采样 cgroup `memory.current/peak/high/max/events`。
- `tests/test-memory-behavior-profile-parser.py`：验证 parser 能解析带 server 前缀的 marker、计算 Dense/MoE/KV delta、保留未知字段并在缺失 marker 时给出 warning。

该脚本明确记录 `evidence_limits`，包括：

- `memory_governor_observe` 是采样 marker，不是 per-token complete trace。
- latency 受 server HTTP/streaming 与采样粒度影响。
- Dense 参数加载由 Flex counters 推断，不是硬件 PMU 级内存追踪。
- 脚本记录证据，不直接声称性能提升。

因此，文档和论文中可以把该脚本作为“实验取证与 parser 管线”，但不能把它本身当成性能结论。

### 8.7 测试覆盖性质

`tests/test-memory-governor-observe-static.py` 目前是源码字符串级静态测试：

- 能确认 marker schema、关键 API token、gate token 存在；
- 能约束 clean reclaim observe path 不调用某些 forbidden demand load/prefetch API；
- 不能验证运行期字段值正确；
- 不能验证真实 RSS 下降；
- 不能验证 async 并发安全；
- 不能验证 KV release/offload correctness；
- 不能验证 Dense/MoE/KV combined 性能收益。

---

## 9. 论文配图建议

### 9.1 总体框架图：全局内存治理闭环

建议标题：**Global Memory Governor for Coordinated Weight–KV Memory Management**

```text
                    ┌──────────────────────────────┐
                    │ cgroup / RSS / KV pressure   │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │    Server Memory Governor    │
                    │ observe → score → auction    │
                    │ pressure gate + ROI + credit │
                    └───────┬────────┬────────┬────┘
                            │        │        │
         ┌──────────────────┘        │        └──────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   Dense Flex      │       │   MoE Buffer      │       │    KV Cache       │
│ layer ring        │       │ expert slice LRU  │       │ release/offload   │
│ lock / delta pin  │       │ warm budget       │       │ slot fallback     │
│ clean reclaim     │       │ predictor prefetch│       │ soft budget       │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │ stats/action result       │ stats/action result       │ stats/action result
         └───────────────────────────┴───────────────────────────┘
```

图中建议用颜色区分：

- 蓝色：观测路径；
- 橙色：回收/释放动作；
- 绿色：预算增长/re-pin；
- 紫色：prefetch budget；
- 红色虚线：pressure/headroom guard。

### 9.2 Dense Flex 单模块图

建议标题：**Layer-wise Dense Weight Streaming with Runtime Re-pinning**

```text
GGUF weight file
      │ direct pread / O_DIRECT
      ▼
┌───────────────────────┐
│ Dense Flex IO workers │
└──────────┬────────────┘
           ▼
┌───────────────────────────────┐
│ Layer Ring Buffer              │
│ slot 0 | slot 1 | ... | slot k │
└───────┬───────────────┬───────┘
        │               │
        ▼               ▼
 current layer      prefetch ahead
        │
        ▼
ggml tensor->data repoint
        │
        ▼
compute consumes layer
        │
        ▼
release slot → clean reclaim

Side buffers:
  ├─ Balanced Lock Buffer  (load-time pinned tensors)
  └─ Delta Pin Buffer      (runtime ROI re-pin)
```

要强调：

- ring 控制 dense 权重 resident 上限；
- lock/delta pin 降低每 token 重复读；
- clean reclaim 只回收 released layer；
- prefetch budget 防止 speculative layer reads 抢占内存和 IO。

### 9.3 MoE Buffer 单模块图

建议标题：**Expert-slice Anonymous Cache with Predictor-guided Prefetch**

```text
Router / CLG / EAM / CCT predictors
        │ selected experts / ranked future experts
        ▼
┌──────────────────────────────┐
│       MoE Buffer Controller  │
│ LRU + hot pin + admission    │
│ warm working set + budget    │
└──────────────┬───────────────┘
               │ pread selected slices
               ▼
     GGUF / sidecar expert data
               │
               ▼
┌────────────────────────────────────────────┐
│ full-size anonymous expert tensor buffer   │
│ expert0 | expert1 | ... | expertN         │
└──────────────┬─────────────────────────────┘
               │ tensor->data + expert_id*stride
               ▼
       MUL_MAT_ID / fused SWIGLU / down

Eviction path:
 cold clean expert group → MADV_DONTNEED
```

要强调：

- 冷 expert zero-fill/unbacked，不占物理页；
- group-level eviction 避免 gate/up/down 抖动；
- warm working set 用于预算目标；
- predictor prefetch 与 demand load 分离；
- sidecar/fused path 需要 correctness/精度实验支撑。

### 9.4 KV 协议图

建议标题：**KV Pressure Actions: Current Path and Target Lifecycle Contract**

```text
Current implemented coordination path:

Server Governor
  ├─ sample KV budget / slot-state budget
  ├─ pressure gate + cooldown
  ├─ evaluate
  ├─ release / offload
  └─ marker: outcome, relieved bytes, reason

Target lifecycle contract (mark as partially implemented):

Server scheduler (policy only)
        │ logical action request + decision_id
        ▼
KV lifecycle core (state authority)
        │ ownership / transaction / quarantine checks
        │ backing IO / commit / rollback / fail-stop
        ▼
block events + transaction terminal + core result
        ▼
server observation + runner observation + parser verdict
```

图中应明确：

- 当前已实现路径画实线；
- unified lifecycle contract 未闭合部分画虚线；
- 不把目标契约画成已验证实现。

### 9.5 Reallocation credit 图

建议标题：**Reclaim-to-Reinvestment Feedback Loop**

```text
Pressure actions
  ├─ Dense clean reclaim
  ├─ MoE clean reclaim / shrink
  ├─ KV release
  └─ KV offload
        │ relieved bytes
        ▼
Pending credit
        │ confirm by memory.current drop
        ▼
Earned credit
        │ cap / decay / hard guard
        ▼
Reinvestment
  ├─ MoE budget grow / emergency warm working set
  └─ Dense delta re-pin / prefetch window
```

适合表达“模块不是各自贪心优化，而是通过全局 credit 闭环协调”。

---

## 10. 当前实现边界与风险清单

### 10.1 已实现并有源码/静态测试证据

- Dense Flex ring streaming、lock、delta pin、clean reclaim、ring resize API。
- MoE Buffer anonymous expert slice cache、LRU/group eviction、warm working set、prefetch budget、stats/reclaim API。
- Server Memory Governor 的 Dense/MoE/KV/global/reallocation/prefetch marker schema。
- KV release/offload env gate、candidate、evaluate、execute action、marker 字段。
- Async action queue 覆盖 Dense/MoE/KV 回收/迁移动作。
- Unified prefetch budget 字段与 Dense/MoE API 接入。
- `tests/test-memory-governor-observe-static.py` 对关键 token/schema 进行静态检查。

### 10.2 已实现但运行证据不足

- Dense ring shrink 在并发 compute/prefetch 下的安全性。
- Dense delta pin async 与 stream callback 之间是否完全无竞态。
- MoE warm working set estimator 的覆盖率和 budget 质量。
- MoE predictor 命中率、prefetch late/unused 对 TPOT 的影响。
- MoE dynamic budget grow/shrink 是否稳定、不振荡。
- KV release/offload 在不同 KV backend 下的实际 relieved bytes 和恢复路径。
- slot-state offload fallback 的 prompt cache 保存/恢复完整性。
- unified prefetch budget 是否真正降低 Dense/MoE/KV IO 抢占。
- reallocation credit 是否在真实 cgroup pressure 下稳定。
- combined 相比 KV-only/weight-only 是否有正收益。

### 10.3 尚未完全实现 / 目标契约差距

根据 `docs/kv_pressure_scheduler_contract.md` 当前状态：

- 五种 action `NOOP/EVALUATE/RELEASE/OFFLOAD/PREFETCH` 尚未完全统一 request/result。
- OFFLOAD/PREFETCH 尚未完全由统一 scheduler 仲裁。
- Active-required prefetch 优先级与 graph compute 前失败传播尚未完整实现。
- core action result、server observation、runner observation 尚未严格分层。
- block event、transaction terminal、physical block generation ID 尚未完整实现。
- quarantine/fail-closed/逐候选 destructive recheck 尚有 P0/P1 差距。

---

## 11. 建议实验矩阵

为了让论文配图和答辩可信，建议至少补齐以下实验矩阵。

### 11.1 四组主对照

| 组别 | KV | Dense/MoE 权重治理 | Governor | 用途 |
|---|---|---|---|---|
| baseline | off | off | off | 原始 llama.cpp 或当前关闭所有治理 |
| KV-only | on | off | partial | 证明 KV release/offload 独立收益 |
| weight-only | off | Dense Flex/MoE Buffer on | partial | 证明权重侧独立收益 |
| combined | on | on | on | 证明协同后不退化且收益更稳 |

### 11.2 Dense 模型消融

- ring size sweep；
- ahead sweep；
- balanced locking on/off；
- delta re-pin on/off；
- clean reclaim on/off；
- unified prefetch budget on/off；
- cgroup memory.max sweep。

指标：

- RSS peak / memory.current peak；
- OOM/oom_kill；
- TTFT；
- TPOT；
- throughput；
- `dense_stream_per_token_bytes`；
- `dense_wait_us` / `ewma_wait_us`；
- `dense_prefetch_budget_dropped`。

### 11.3 MoE 模型消融

- MoE Buffer off/on；
- unbounded vs bounded budget；
- warm working set coverage sweep；
- CLG/EAM/CCT prefetch on/off；
- dynamic budget on/off；
- pressure-adaptive protection on/off；
- clean reclaim on/off；
- sidecar/dynamic bits on/off；
- fused expert FFN/SWIGLU on/off。

指标：

- RSS peak / memory.current peak；
- `moe_resident_bytes`；
- `moe_budget_bytes`；
- `moe_evictions`；
- `moe_bytes_read`；
- `moe_cache_hits/misses`；
- `moe_prefetch_hits/late/unused`；
- pressure-adaptive debug stats：pressure ratio、scale、effective pin/window/cooldown；
- TTFT/TPOT/throughput；
- 正确性/精度对照，尤其 dynamic bits/sidecar/fused path。

### 11.4 KV 与协同实验

- KV release on/off；
- KV offload on/off；
- slot-state fallback on/off；
- KV soft budget on/off；
- unified prefetch budget on/off；
- reallocation credit on/off；
- async actions on/off。

指标：

- `kv_resident_bytes`；
- `kv_reclaimable_resident_bytes`；
- `kv_release_relieved_bytes`；
- `kv_offload_relieved_bytes`；
- `kv_offload_backend`；
- `prefetch_budget_kv_resume_used_bytes`；
- request latency；
- resume correctness；
- long context stability。

---

## 12. 最适合写进论文的设计亮点

在有实验支撑后，可以将以下点作为创新或工程亮点候选。

### 12.1 显式权重驻留替代 page-cache hint

Dense Flex 与 MoE Buffer 都避免单纯依赖 mmap/page cache，而是使用用户态可观测、可预算、可回收的显式 buffer。

### 12.2 Dense 与 MoE 使用不同粒度的权重治理

- Dense：layer ring + tensor pin。
- MoE：expert slice + group LRU + warm working set。

这体现模型结构感知的内存管理。

### 12.3 统一 Memory Governor 协调 Weight 与 KV

系统不是分别优化 KV 和权重，而是在 server 层统一观测、统一候选、统一 budget、统一 pressure gate。

### 12.4 Reclaim-to-reinvestment 闭环

把释放出来的内存经过 cgroup/current 确认后，再投给 MoE warm working set 或 Dense delta pin，形成稳定闭环。

### 12.5 Unified prefetch budget

将 Dense layer prefetch、MoE expert prefetch、KV resume accounting 放到同一个 tick budget 下，针对 memory/IO 突刺进行控制。

### 12.6 MoE warm working set 与 predictor-guided expert cache

MoE budget 不按全 expert 常驻估算，而按 coverage、历史访问、结构先验、predictor 信号估计 warm working set。

---

## 13. 最需要避免的表述

在没有补充实验之前，不建议写：

- “性能提升 X%”；
- “显著降低 TPOT/TTFT”；
- “完全解决 OOM”；
- “KV lifecycle 已完整协议化”；
- “combined 一定优于 KV-only/weight-only”；
- “dynamic bits 无精度损失”；
- “prefetch predictor 命中率已验证”；
- “async action 无并发风险”；
- “slot-state offload 恢复路径已完整验证”。

建议改写为：

- “当前源码已实现该路径，运行收益需通过实验验证”；
- “该机制用于降低/约束……，实际效果由后续 benchmark 给出”；
- “目标契约已定义，当前实现尚有 P0/P1 差距”；
- “本文实验将通过 baseline、KV-only、weight-only、combined 四组对照验证协同效果”。

---

## 14. 文件索引

当前梳理主要依据以下文件：

- `src/llama-flex.h`：Dense Flex 对外 API、参数、stats、设计注释。
- `src/llama-flex.cpp`：Dense Flex ring streaming、IO worker、lock、delta pin、clean reclaim、resize、callback。
- `src/llama-moe-buffer.h`：MoE Buffer 参数、stats、注册、stream callback、prefetch、reclaim、budget API。
- `src/llama-moe-buffer.cpp`：MoE expert anonymous buffer、LRU/group eviction、predictor、sidecar/fused path、stats/reclaim 实现。
- `src/llama-model.cpp`：加载期 auto backend、Dense/MoE planner、MoE sidecar、Flex/MoE context 持有与注册。
- `src/llama-model-loader.h` / `src/llama-model-loader.cpp`：模型路径记录、lazy/flex 与 mmap prefetch/mlock 关系。
- `tools/server/server-context.cpp`：Server Memory Governor、KV pressure sampler、candidate/auction、prefetch budget、clean reclaim、KV release/offload、reallocation、marker。
- `tests/test-memory-governor-observe-static.py`：静态 schema/token presence 测试。
- `docs/kv_pressure_scheduler_contract.md`：KV pressure scheduler 目标契约与当前实现差距。
- `scripts/profile-global-memory-sweep.sh`：profile 脚本，用于 dense/MoE cgroup memory sweep 与 marker 汇总。
- `scripts/run-memory-behavior-profile.py`：memory behavior profile 脚本，采集 server marker、HTTP streaming timing 与 cgroup samples，输出 JSON/TSV/Markdown summary。
- `tests/test-memory-behavior-profile-parser.py`：memory behavior profile parser 的单元测试，覆盖 marker parsing、delta aggregation、未知字段保留和 missing-marker warning。

---

## 15. 最终总结

当前系统可以概括为：

```text
模型结构感知权重驻留
    Dense: layer ring + pin
    MoE: expert slice cache + predictor
          │
          ▼
统一 Server Memory Governor
    pressure-aware observation
    ROI auction
    prefetch budget
    clean reclaim
    KV release/offload
    reallocation credit
          │
          ▼
Weight–KV 协同内存治理
    降低权重常驻峰值
    限制 expert working set
    控制 KV resident/offload
    避免 prefetch/IO 抢占
    在 cgroup pressure 下保持稳定
```

但论文和答辩中必须保持证据边界：当前源码已经形成较完整的协同框架和多个动作路径；静态测试确认 schema 与关键 gate 存在；真正的性能、稳定性、correctness 与 combined 协同收益仍需要通过 baseline、KV-only、weight-only、combined 四组实验和必要消融来证明。
---

## 16. 关键代码完整实现附录

> 本附录按“论文配图/答辩时最可能被追问的实现路径”粘贴关键源码。
> 这里不是伪代码，而是当前工作区对应文件中的完整关键结构体或完整关键函数。
> 每次源码发生变化时，应重新生成本附录，避免文档中的代码片段滞后于当前分支。

### 加载期总入口：llama_model_base::load_tensors()

来源：`src/llama-model.cpp:1278-2784`

```cpp
bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer      = hparams.n_layer;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;
    const auto planner_cgroup = llama_read_cgroup_memory_snapshot();
    const bool memory_governor_auto_by_cgroup =
            planner_cgroup.valid && planner_cgroup.max_bytes > 0;
    const bool memory_governor_requested =
            llama_env_flag("LLAMA_MEMORY_GOVERNOR") || memory_governor_auto_by_cgroup;
    const bool memory_governor_auto_backends =
            llama_env_flag("LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS", memory_governor_requested);
    size_t model_weight_bytes = 0;
    size_t largest_weight_tensor_bytes = 0;
    for (const auto & it : ml.weights_map) {
        const size_t nb = ggml_nbytes(it.second.tensor);
        model_weight_bytes += nb;
        largest_weight_tensor_bytes = std::max(largest_weight_tensor_bytes, nb);
    }
    size_t governor_kv_bytes_per_token = 0;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        governor_kv_bytes_per_token +=
            ((size_t) hparams.n_head_kv(il) *
             ((size_t) hparams.n_embd_head_k(il) + (size_t) hparams.n_embd_head_v(il)) *
             sizeof(uint16_t));
    }
    const size_t detected_avail = memory_governor_auto_backends
            ? llama_detect_available_memory()
            : SIZE_MAX;
    const size_t governor_backend_reserve =
        largest_weight_tensor_bytes +
        governor_kv_bytes_per_token * (size_t) std::max<uint32_t>(1, hparams.n_layer);
    const bool governor_low_memory =
            memory_governor_auto_backends &&
            detected_avail != SIZE_MAX &&
            detected_avail < model_weight_bytes + governor_backend_reserve;
    const bool model_has_moe = hparams.n_expert > 0;

    const bool lazy_v2_requested =
            llama_env_flag("LLAMA_LAZY_V2") ||
            (memory_governor_auto_backends && model_has_moe && !llama_env_is_set("LLAMA_LAZY_V2"));
    const bool lazy_window_requested =
            lazy_v2_requested ||
            llama_env_flag("LLAMA_LAZY_LOADING") ||
            (params.vm_layer_schedule && params.vm_dontneed);
    // FlexInfer-style streaming (llama-flex): mutually exclusive with the mmap
    // window. Requires the same loader conditions (mmap home, no mlock/check).
    const bool flex_requested = llama_env_flag("LLAMA_FLEX") ||
            (memory_governor_auto_backends && !model_has_moe && !llama_env_is_set("LLAMA_FLEX"));
    const bool use_flex =
            flex_requested &&
            !use_mlock &&
            !ml.check_tensors &&
            ml.use_mmap &&
            !params.vocab_only;

    const bool use_lazy_window =
            !use_flex &&
            lazy_window_requested &&
            !use_mlock &&
            !ml.check_tensors &&
            ml.use_mmap &&
            !params.vocab_only;

    if (lazy_window_requested && !use_lazy_window && params.vm_debug_log) {
        LLAMA_LOG_WARN(
                "%s: lazy window disabled for this loader pass "
                "(mmap=%d, mlock=%d, check_tensors=%d, vocab_only=%d)\n",
                __func__,
                ml.use_mmap ? 1 : 0,
                use_mlock ? 1 : 0,
                ml.check_tensors ? 1 : 0,
                params.vocab_only ? 1 : 0);
    }
    if (memory_governor_requested && (governor_low_memory || use_flex || use_lazy_window)) {
        LLAMA_LOG_INFO("%s: memory governor auto backend: available=%.0f MiB weights=%.0f MiB "
                "reserve=%.0f MiB flex=%d lazy_window=%d moe=%d\n",
                __func__,
                detected_avail / 1048576.0,
                model_weight_bytes / 1048576.0,
                governor_backend_reserve / 1048576.0,
                use_flex ? 1 : 0,
                use_lazy_window ? 1 : 0,
                model_has_moe ? 1 : 0);
    }

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s, direct_io = %s)\n",
        __func__, ml.use_mmap ? "true" : "false", ml.use_direct_io ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(
            devices,
            (use_lazy_window || use_flex) ? false : params.use_extra_bufts,
            params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < int(hparams.n_layer) && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;

        if (n_expert > 0 && n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output && output->type == GGML_TYPE_NVFP4) {
            // weight scale
            if (!output_s) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }
    ml.done_getting_tensors();

    GGML_ASSERT(!(output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            output->type == GGML_TYPE_NVFP4));
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(!(use_lazy_window || use_flex), use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ml.ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL,
                    params.progress_callback, params.progress_callback_user_data, use_lazy_window)) {
            return false;
        }
    }

    llama_rss_stage_log("A", "model_mmap_done");

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    if (use_lazy_window) {
        // MoE expert streaming via explicit anon buffers (per-expert-slice repoint).
        // When enabled, expert tensors are managed by llama-moe-buffer instead of
        // the mmap window: their data is repointed to anon buffers and only the
        // routed experts are streamed in. Requires LLAMA_LAZY_V2 (this block).
        const char * moe_buf_env = std::getenv("LLAMA_LAZY_MOE_BUFFER");
        const bool   use_moe_buffer =
                (moe_buf_env != nullptr && std::atoi(moe_buf_env) > 0) ||
                (memory_governor_auto_backends && model_has_moe && moe_buf_env == nullptr);
        const bool   moe_auto_budget =
                llama_env_flag("LLAMA_LAZY_MOE_BUFFER_AUTO") ||
                (memory_governor_auto_backends && model_has_moe &&
                 !llama_env_is_set("LLAMA_LAZY_MOE_BUFFER_AUTO"));
        double moe_warm_coverage = 0.95;
        std::string moe_sidecar_source = "none";
        if (use_moe_buffer) {
            llama_moe_buffer_params mp;
            mp.enabled   = true;
            mp.debug_log = std::getenv("LLAMA_LAZY_DEBUG") != nullptr;
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_HEBF")) {
                mp.hebf_schedule = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_DYNBITS")) {
                mp.dynamic_bits = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_DYNBITS_REAL")) {
                mp.dynamic_bits_real = std::atoi(v) > 0;
                mp.dynamic_bits = mp.dynamic_bits || mp.dynamic_bits_real;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_STRICT_SIDECAR")) {
                mp.strict_sidecar = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_NATIVE_HOT")) {
                mp.native_hot = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_FUSE_GATE_UP")) {
                mp.fuse_gate_up = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_FUSE_SWIGLU")) {
                mp.fuse_swiglu = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_FUSE_DIRECT_SWIGLU")) {
                mp.fuse_direct_swiglu = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_FUSE_FFN")) {
                mp.fuse_expert_ffn = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PREFETCH_DOWN")) {
                mp.prefetch_down_with_swiglu = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_AVX512_Q2")) {
                mp.avx512_q2 = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_AVX512_Q2_DOT")) {
                mp.avx512_q2_dot = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_VNNI_Q2")) {
                mp.vnni_q2 = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_VNNI_Q2_DOWN")) {
                mp.vnni_q2_down = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_VNNI_Q2_SWIGLU")) {
                mp.vnni_q2_swiglu = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_VNNI_BLOCK")) {
                mp.vnni_block = std::max(64, std::min(256, std::atoi(v)));
                mp.vnni_block = (mp.vnni_block / 64) * 64;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_AVX512_PREFETCH")) {
                mp.avx512_prefetch = std::max(0, std::min(16, std::atoi(v)));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_BUFFER_MB")) {
                const int mb = std::max(0, std::atoi(v));
                mp.budget_bytes = (size_t) mb * 1024ull * 1024ull;
                mp.budget_unbounded = mb == 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_BUFFER_WORKERS")) {
                mp.n_workers = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_BASE_BITS")) {
                mp.base_bits = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_HOT_BITS")) {
                mp.hot_bits = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_WARM_BITS")) {
                mp.warm_bits = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_COLD_BITS")) {
                mp.cold_bits = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_GATE_BITS")) {
                mp.gate_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_UP_BITS")) {
                mp.up_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_DOWN_BITS")) {
                mp.down_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_FIXED_BITS")) {
                mp.fixed_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_GATE_MIN_BITS")) {
                mp.gate_min_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_UP_MIN_BITS")) {
                mp.up_min_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_DOWN_MIN_BITS")) {
                mp.down_min_bits = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_TOP_K")) {
                mp.sync_top_k = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_BUFFER_HOT_RATIO")) {
                mp.hot_ratio = std::max(0.0f, (float) std::atof(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PINNED_FRACTION")) {
                mp.pinned_fraction = std::max(0.0f, std::min(0.90f, (float) std::atof(v)));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PINNED_LAYER_FRACTION")) {
                mp.pinned_layer_fraction = std::max(0.0f, std::min(1.0f, (float) std::atof(v)));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_ACTIVE_WINDOW")) {
                mp.active_window = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_GROUP_COOLDOWN_TOKENS")) {
                mp.group_cooldown_tokens = std::max(0, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_ADAPTIVE")) {
                mp.pressure_adaptive = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SCALE_PIN")) {
                mp.pressure_scale_pin = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SCALE_LAYER_PIN")) {
                mp.pressure_scale_layer_pin = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SCALE_WINDOW")) {
                mp.pressure_scale_window = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SCALE_COOLDOWN")) {
                mp.pressure_scale_cooldown = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SCALE_SPEC_GUARD")) {
                mp.pressure_scale_spec_guard = std::atoi(v) > 0;
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_SOFT_RATIO")) {
                mp.pressure_soft_ratio = std::max(0.50, std::min(0.99, std::atof(v)));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_HARD_RATIO")) {
                mp.pressure_hard_ratio = std::max(mp.pressure_soft_ratio + 0.01, std::min(1.20, std::atof(v)));
            }
            if (mp.pressure_hard_ratio <= mp.pressure_soft_ratio) {
                mp.pressure_hard_ratio = std::min(1.20, mp.pressure_soft_ratio + 0.01);
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PRESSURE_PIN_FLOOR")) {
                mp.pressure_pin_floor = std::max(0.0f, std::min(mp.pinned_fraction, (float) std::atof(v)));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_PIN_REFRESH")) {
                mp.pin_refresh_interval = std::max(1, std::atoi(v));
            }
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_WARM_COVERAGE")) {
                mp.warm_coverage = std::max(0.0, std::min(1.0, std::atof(v)));
            }
            moe_warm_coverage = mp.warm_coverage;
            if (const char * v = std::getenv("LLAMA_LAZY_MOE_SIDECAR")) {
                mp.sidecar_path = v;
                moe_sidecar_source = "env";
            } else {
                mp.sidecar_path = llama_auto_moe_sidecar_path(ml.fname_model);
                if (!mp.sidecar_path.empty()) {
                    moe_sidecar_source = "auto";
                }
            }
            pimpl->moe_buffer = llama_moe_buffer_create(mp);
        }

        std::vector<llama_window_region_input> inputs;
        inputs.reserve(ml.weights_map.size());

        int moe_registered = 0;
        size_t model_total_bytes = 0;  // sum of all mapped weight bytes (for adaptive budget)
        std::vector<size_t> moe_layer_group_bytes(hparams.n_layer, 0);
        for (const auto & it : ml.weights_map) {
            const auto & weight = it.second;
            if (weight.idx >= pimpl->mappings.size() || !pimpl->mappings[weight.idx]) {
                continue;
            }
            model_total_bytes += ggml_nbytes(weight.tensor);

            // Detect MoE expert weight tensors: 3-D tensors whose name contains
            // "_exps" (e.g. blk.N.ffn_gate_exps.weight, blk.N.ffn_down_exps.weight).
            const ggml_tensor * t = weight.tensor;
            const bool is_exps = ggml_n_dims(t) == 3 && t->ne[2] > 1 &&
                    it.first.find("_exps") != std::string::npos;
            if (is_exps) {
                int layer = -1;
                if (std::sscanf(it.first.c_str(), "blk.%d.", &layer) == 1 &&
                        layer >= 0 && layer < (int) hparams.n_layer) {
                    moe_layer_group_bytes[layer] += (size_t) t->nb[2];
                }
            }

            // Buffer mode owns the expert tensors: repoint them and skip the window.
            if (is_exps && use_moe_buffer && weight.idx < ml.files.size()) {
                if (llama_moe_buffer_register(*pimpl->moe_buffer,
                            const_cast<ggml_tensor *>(t), ml.files[weight.idx]->file_id(),
                            weight.offs, t->nb[2], (int) t->ne[2])) {
                    ++moe_registered;
                    continue;
                }
            }

            llama_window_region_input input;
            input.name = it.first;
            input.addr = (uint8_t *) pimpl->mappings[weight.idx]->addr() + weight.offs;
            input.size = ggml_nbytes(weight.tensor);
            input.file_idx = weight.idx;
            input.file_offset = weight.offs;
            // For these we also store the per-expert byte stride so the window
            // controller can slice them at expert granularity.
            if (is_exps) {
                input.n_expert      = (int) t->ne[2];
                input.expert_stride = t->nb[2];
            }
            inputs.emplace_back(std::move(input));
        }
        if (use_moe_buffer && pimpl->moe_buffer && std::getenv("LLAMA_LAZY_DEBUG")) {
            LLAMA_LOG_INFO("%s: llama-moe-buffer managing %d expert tensors\n", __func__, moe_registered);
        }

        // Adaptive expert budget: pick the resident-expert byte budget from the
        // memory actually available, so we evict only as much as needed to fit.
        //   available >= model      -> no eviction (budget = 0, unbounded)
        //   available <  model      -> budget = available - non_expert - reserve
        // The non-expert weights stay mmap-resident and the KV/compute scratch is
        // not evictable, so they are subtracted first (they are the OOM floor).
        if (moe_auto_budget && use_moe_buffer && pimpl->moe_buffer) {
            const size_t expert_bytes = llama_moe_buffer_expert_bytes(pimpl->moe_buffer.get());
            const size_t warm_working_set = llama_moe_buffer_warm_working_set_bytes(
                    pimpl->moe_buffer.get(), moe_warm_coverage);
            const size_t non_expert   = model_total_bytes > expert_bytes
                    ? model_total_bytes - expert_bytes : 0;

            const size_t kv_reserve =
                governor_kv_bytes_per_token * (size_t) std::max<uint32_t>(1, hparams.n_layer);
            const size_t reserve = largest_weight_tensor_bytes + kv_reserve;
            size_t min_expert_group = 0;
            for (const size_t b : moe_layer_group_bytes) {
                min_expert_group = std::max(min_expert_group, b);
            }
            const size_t floor = non_expert + reserve;
            const bool finite_limit = planner_cgroup.valid && planner_cgroup.max_bytes > 0;
            size_t safe_budget = 0;
            bool budget_unbounded = false;
            size_t moe_hard_floor = floor;
            size_t moe_soft_guard = reserve;
            size_t moe_working_set_floor = 0;
            const char * moe_budget_reason = "legacy_safe_budget";
            bool moe_budget_clamped_by_headroom = false;
            if (finite_limit) {
                // V2: the old floor treats all non-expert mmap bytes as a hard RSS
                // floor. In practice a large part of that floor is file-backed and
                // reclaimable, while too small an expert cache causes massive rereads.
                // Keep a hard cgroup guard, but let a model-structure working-set
                // floor compete with the conservative legacy safe budget.
                const size_t min_guard = 64ull * 1024ull * 1024ull;
                moe_soft_guard = std::max({ min_guard, largest_weight_tensor_bytes, kv_reserve, min_expert_group });
                const size_t current_floor = planner_cgroup.current_bytes + moe_soft_guard;
                moe_hard_floor = std::max(current_floor, reserve);
                if (planner_cgroup.max_bytes <= moe_hard_floor) {
                    throw std::runtime_error(format(
                            "%s: memory planner capacity failure model=moe memory_max=%.0f MiB "
                            "hard_floor=%.0f MiB legacy_floor=%.0f MiB non_expert=%.0f MiB reserve=%.0f MiB "
                            "largest_tensor=%.0f MiB kv_reserve=%.0f MiB",
                            __func__,
                            planner_cgroup.max_bytes / 1048576.0,
                            moe_hard_floor / 1048576.0,
                            floor / 1048576.0,
                            non_expert / 1048576.0,
                            reserve / 1048576.0,
                            largest_weight_tensor_bytes / 1048576.0,
                            kv_reserve / 1048576.0));
                }
                const size_t legacy_safe_budget = planner_cgroup.max_bytes > floor
                    ? std::min(planner_cgroup.max_bytes - floor, expert_bytes)
                    : 0;
                const size_t hard_safe_budget = std::min(planner_cgroup.max_bytes - moe_hard_floor, expert_bytes);
                moe_working_set_floor = warm_working_set / 3;
                if (warm_working_set > 0) {
                    moe_working_set_floor = std::max(moe_working_set_floor, warm_working_set / 4);
                }
                if (min_expert_group > 0) {
                    moe_working_set_floor = std::max(moe_working_set_floor, min_expert_group * (size_t) std::max<uint32_t>(1, hparams.n_expert_used));
                    moe_working_set_floor = std::max(moe_working_set_floor, min_expert_group);
                }
                moe_working_set_floor = std::min(moe_working_set_floor, warm_working_set);
                moe_working_set_floor = std::min(moe_working_set_floor, expert_bytes);
                safe_budget = legacy_safe_budget;
                if (safe_budget < moe_working_set_floor && hard_safe_budget > safe_budget) {
                    const size_t target = std::min(moe_working_set_floor, hard_safe_budget);
                    safe_budget = std::max(safe_budget, target);
                    moe_budget_reason = target < moe_working_set_floor ? "working_set_headroom_clamped" : "working_set_floor";
                    moe_budget_clamped_by_headroom = target < moe_working_set_floor;
                }
                safe_budget = std::min(safe_budget, expert_bytes);
                if (safe_budget > 0 && min_expert_group > 0 && safe_budget < min_expert_group) {
                    throw std::runtime_error(format(
                            "%s: memory planner capacity failure model=moe memory_max=%.0f MiB "
                            "hard_floor=%.0f MiB safe_budget=%.0f MiB min_expert_group=%.0f MiB",
                            __func__,
                            planner_cgroup.max_bytes / 1048576.0,
                            moe_hard_floor / 1048576.0,
                            safe_budget / 1048576.0,
                            min_expert_group / 1048576.0));
                }
                budget_unbounded = safe_budget >= expert_bytes;
            } else {
                const size_t avail = llama_detect_available_memory();
                safe_budget = (avail == SIZE_MAX || avail >= floor + expert_bytes)
                    ? expert_bytes
                    : (avail > floor ? avail - floor : 0);
                safe_budget = std::min(safe_budget, expert_bytes);
                budget_unbounded = safe_budget >= expert_bytes;
            }
            const size_t budget = budget_unbounded
                ? 0
                : std::min(warm_working_set, safe_budget);
            llama_moe_buffer_set_budget_plan(
                    pimpl->moe_buffer.get(),
                    budget,
                    budget_unbounded,
                    safe_budget,
                    moe_hard_floor);
            std::fprintf(stderr, "%s: memory planner model=moe auto=%d planner_v2=%d memory_max=%.0f MiB "
                    "memory_current=%.0f MiB floor=%.0f MiB hard_floor=%.0f MiB soft_guard=%.0f MiB "
                    "available_for_dynamic=%.0f MiB expert=%.0f MiB warm=%.0f MiB min_expert_group=%.0f MiB "
                    "moe_working_set_floor=%.0f MiB moe_budget=%.0f MiB moe_budget_unbounded=%d "
                    "moe_budget_reason=%s moe_budget_clamped_by_headroom=%d sidecar=%s sidecar_source=%s\n",
                    __func__,
                    memory_governor_auto_backends ? 1 : 0,
                    finite_limit ? 1 : 0,
                    finite_limit ? planner_cgroup.max_bytes / 1048576.0 : -1.0,
                    finite_limit ? planner_cgroup.current_bytes / 1048576.0 : -1.0,
                    floor / 1048576.0,
                    moe_hard_floor / 1048576.0,
                    moe_soft_guard / 1048576.0,
                    safe_budget / 1048576.0,
                    expert_bytes / 1048576.0,
                    warm_working_set / 1048576.0,
                    min_expert_group / 1048576.0,
                    moe_working_set_floor / 1048576.0,
                    budget / 1048576.0,
                    budget_unbounded ? 1 : 0,
                    moe_budget_reason,
                    moe_budget_clamped_by_headroom ? 1 : 0,
                    moe_sidecar_source == "none" ? "gguf_streaming" : "sidecar",
                    moe_sidecar_source.c_str());
            LLAMA_LOG_INFO("%s: memory planner model=moe auto=%d planner_v2=%d memory_max=%.0f MiB "
                    "memory_current=%.0f MiB floor=%.0f MiB hard_floor=%.0f MiB soft_guard=%.0f MiB "
                    "available_for_dynamic=%.0f MiB expert=%.0f MiB warm=%.0f MiB min_expert_group=%.0f MiB "
                    "moe_working_set_floor=%.0f MiB moe_budget=%.0f MiB moe_budget_unbounded=%d "
                    "moe_budget_reason=%s moe_budget_clamped_by_headroom=%d sidecar=%s sidecar_source=%s\n",
                    __func__,
                    memory_governor_auto_backends ? 1 : 0,
                    finite_limit ? 1 : 0,
                    finite_limit ? planner_cgroup.max_bytes / 1048576.0 : -1.0,
                    finite_limit ? planner_cgroup.current_bytes / 1048576.0 : -1.0,
                    floor / 1048576.0,
                    moe_hard_floor / 1048576.0,
                    moe_soft_guard / 1048576.0,
                    safe_budget / 1048576.0,
                    expert_bytes / 1048576.0,
                    warm_working_set / 1048576.0,
                    min_expert_group / 1048576.0,
                    moe_working_set_floor / 1048576.0,
                    budget / 1048576.0,
                    budget_unbounded ? 1 : 0,
                    moe_budget_reason,
                    moe_budget_clamped_by_headroom ? 1 : 0,
                    moe_sidecar_source == "none" ? "gguf_streaming" : "sidecar",
                    moe_sidecar_source.c_str());
        }

        llama_window_params window_params;
        window_params.enabled = true;
        const char * lazy_dontneed = std::getenv("LLAMA_LAZY_DONTNEED");
        const char * lazy_window_size = std::getenv("LLAMA_LAZY_WINDOW_SIZE");
        const char * lazy_prefetch_ahead = std::getenv("LLAMA_LAZY_PREFETCH_AHEAD");
        const char * lazy_worker_threads = std::getenv("LLAMA_LAZY_WORKER_THREADS");
        const char * lazy_auto_tune = std::getenv("LLAMA_LAZY_AUTO_TUNE");
        const char * lazy_memory_limit = std::getenv("LLAMA_LAZY_MEMORY_LIMIT");
        const char * lazy_prefill_aggressive = std::getenv("LLAMA_LAZY_PREFILL_AGGRESSIVE");

        // When LLAMA_LAZY_V2 is active, also check its namespace env vars.
        // LLAMA_LAZY_V2_* take precedence over LLAMA_LAZY_* when lazy_v2_requested.
        const char * lazy_v2_window   = lazy_v2_requested ? std::getenv("LLAMA_LAZY_V2_WINDOW")    : nullptr;
        const char * lazy_v2_prefetch = lazy_v2_requested ? std::getenv("LLAMA_LAZY_V2_PREFETCH")  : nullptr;
        const char * lazy_v2_workers  = lazy_v2_requested ? std::getenv("LLAMA_LAZY_V2_WORKERS")   : nullptr;
        const char * lazy_v2_memory   = lazy_v2_requested ? std::getenv("LLAMA_LAZY_V2_MEMORY_GB") : nullptr;

        // Effective env vars: V2-namespace overrides base namespace when both are set
        if (lazy_v2_window)   { lazy_window_size    = lazy_v2_window;   }
        if (lazy_v2_prefetch) { lazy_prefetch_ahead = lazy_v2_prefetch; }
        if (lazy_v2_workers)  { lazy_worker_threads = lazy_v2_workers;  }

        // memory_limit: LAZY_V2_MEMORY_GB is in GB (same unit as LAZY_MEMORY_LIMIT)
        if (lazy_v2_memory)   { lazy_memory_limit   = lazy_v2_memory;   }

        window_params.memory_limit = lazy_memory_limit != nullptr
                ? size_t(std::max(0.0, std::atof(lazy_memory_limit)) * 1024.0 * 1024.0 * 1024.0)
                : 0;
        window_params.use_dontneed =
                params.vm_dontneed ||
                (lazy_dontneed != nullptr && std::atoi(lazy_dontneed) > 0) ||
                (lazy_v2_requested && window_params.memory_limit > 0);
        window_params.auto_tune = lazy_auto_tune == nullptr ||
                std::atoi(lazy_auto_tune) > 0;
        window_params.prefill_aggressive =
                lazy_prefill_aggressive != nullptr &&
                std::atoi(lazy_prefill_aggressive) > 0;
        window_params.debug_log = params.vm_debug_log ||
                (std::getenv("LLAMA_LAZY_DEBUG") != nullptr) ||
                (lazy_v2_requested && std::getenv("LLAMA_LAZY_V2_DEBUG") != nullptr);
        window_params.window_size = lazy_window_size != nullptr
                ? std::max(4, std::atoi(lazy_window_size))
                : 12;
        window_params.prefetch_ahead = lazy_prefetch_ahead != nullptr
                ? std::max(1, std::atoi(lazy_prefetch_ahead))
                : (lazy_v2_requested ? 4 : std::max(1, params.vm_pipeline_layers));
        window_params.keep_behind = std::max(0, params.vm_keep_behind_layers);
        window_params.worker_threads = lazy_worker_threads != nullptr
                ? std::max(1, std::atoi(lazy_worker_threads))
                : (lazy_v2_requested ? 2 : 1);

        // prefetch_budget: use explicit param, or derive from memory_limit for V2
        if (params.vm_prefetch_budget_mb > 0) {
            window_params.prefetch_budget = size_t(params.vm_prefetch_budget_mb) * 1024ull * 1024ull;
        } else if (lazy_v2_requested && window_params.memory_limit > 0) {
            // Reserve up to 25% of the memory limit for prefetch buffering
            window_params.prefetch_budget = window_params.memory_limit / 4;
        } else {
            window_params.prefetch_budget = 0;
        }

        // reclaim_budget: use explicit param, or derive from memory_limit for V2
        if (window_params.use_dontneed) {
            if (params.vm_reclaim_budget_mb > 0) {
                window_params.reclaim_budget = size_t(params.vm_reclaim_budget_mb) * 1024ull * 1024ull;
            } else if (lazy_v2_requested && window_params.memory_limit > 0) {
                // Allow reclaiming up to 25% of the limit in a single pass
                window_params.reclaim_budget = window_params.memory_limit / 4;
            } else {
                window_params.reclaim_budget = 0;
            }
        } else {
            window_params.reclaim_budget = 0;
        }

        // MoE expert sliding window (LLAMA_LAZY_MOE_WINDOW).
        // Enabled independently of the layer-level window; can be combined with
        // LLAMA_LAZY_V2 for full layer+expert windowing on MoE models, or used
        // alone to reduce RSS from expert weights while dense layer weights stay
        // fully resident.
        {
            const char * moe_win   = std::getenv("LLAMA_LAZY_MOE_WINDOW");
            const char * moe_tok   = std::getenv("LLAMA_LAZY_MOE_WINDOW_TOKENS");
            const char * moe_evict = std::getenv("LLAMA_LAZY_MOE_DONTNEED");
            const char * clg_env   = std::getenv("LLAMA_LAZY_CLG");
            const char * clg_delta = std::getenv("LLAMA_LAZY_CLG_DELTA");
            const char * clg_pthr  = std::getenv("LLAMA_LAZY_CLG_PREFILL_THR");
            const char * clg_hot   = std::getenv("LLAMA_LAZY_CLG_HOT");
            const char * clg_warm  = std::getenv("LLAMA_LAZY_CLG_HOT_WARMUP");
            window_params.expert_window =
                    (moe_win != nullptr && std::atoi(moe_win) > 0);
            window_params.expert_window_tokens = moe_tok != nullptr
                    ? std::max(1, std::atoi(moe_tok))
                    : 4;
            window_params.expert_dontneed = moe_evict != nullptr &&
                    std::atoi(moe_evict) != 0;
            window_params.clg_predict =
                    (clg_env != nullptr && std::atoi(clg_env) > 0) ||
                    (governor_low_memory && model_has_moe && clg_env == nullptr);
            window_params.clg_delta = clg_delta != nullptr
                    ? std::max(0, std::atoi(clg_delta)) : 1;
            window_params.clg_prefill_threshold = clg_pthr != nullptr
                    ? std::max(1, std::atoi(clg_pthr)) : 4;
            // Hot-expert protection: experts activated more than clg_hot_thr_pct% of
            // decode tokens are always kept resident.  0 disables the feature.
            // LLAMA_LAZY_CLG_HOT=0  → disable hot protection
            // LLAMA_LAZY_CLG_HOT=20 → protect experts active >20% of tokens (default)
            window_params.clg_hot_thr_pct = clg_hot != nullptr
                    ? std::max(0, std::atoi(clg_hot)) : 20;
            window_params.clg_hot_warmup  = clg_warm != nullptr
                    ? std::max(1, std::atoi(clg_warm)) : 16;
            // CLG implies expert_window (needs expert_slots to be indexed)
            if (window_params.clg_predict) {
                window_params.expert_window = true;
            }
        }

        // Collect CLG gate inputs: use the model's actual layer tensors (not
        // ml.weights_map which holds GGUF context tensors whose data pointers
        // may be NULL or stale before load_all_data has committed them).
        // layers[il].ffn_norm and ffn_gate_inp are guaranteed valid after load.
        std::vector<llama_window_gate_input> gate_inputs;
        if (window_params.clg_predict) {
            gate_inputs.reserve(hparams.n_layer);
            for (int il = 0; il < (int) hparams.n_layer; ++il) {
                const auto & layer = layers[il];
                if (!layer.ffn_norm || !layer.ffn_gate_inp) {
                    continue;  // not a MoE layer or tensors not present
                }
                if (!layer.ffn_norm->data || !layer.ffn_gate_inp->data) {
                    continue;  // data not yet loaded (shouldn't happen after load_all_data)
                }
                llama_window_gate_input gi;
                gi.layer         = il;
                gi.n_expert_used = hparams.n_expert_used;
                gi.norm_eps      = hparams.f_norm_rms_eps;
                gi.norm_tensor   = layer.ffn_norm;
                gi.gate_tensor   = layer.ffn_gate_inp;
                gate_inputs.push_back(gi);
            }
        }

        pimpl->window = llama_window_create(inputs, hparams.n_layer, window_params, gate_inputs);

        // Connect CLG prediction to the explicit-buffer expert streamer: in buffer
        // mode the CLG predictor feeds its predicted experts to the moe-buffer's
        // async prefetch (giving layer L+1 lead time) instead of the mmap window.
        if (pimpl->window && pimpl->moe_buffer && window_params.clg_predict) {
            llama_window_set_moe_buffer(*pimpl->window, pimpl->moe_buffer.get());
        }
    }

    if (use_flex) {
        llama_flex_params fp;
        fp.enabled = true;
        fp.debug_log   = std::getenv("LLAMA_FLEX_DEBUG") != nullptr;
        fp.direct_io   = std::getenv("LLAMA_FLEX_BUFFERED") == nullptr; // default O_DIRECT
        const bool governor_respect_flex_env =
            !memory_governor_requested ||
            llama_env_flag("LLAMA_MEMORY_GOVERNOR_RESPECT_FLEX_ENV");
        const bool flex_ring_explicit    = governor_respect_flex_env && std::getenv("LLAMA_FLEX_RING")    != nullptr;
        const bool flex_ahead_explicit   = governor_respect_flex_env && std::getenv("LLAMA_FLEX_AHEAD")   != nullptr;
        const bool flex_max_ahead_explicit = governor_respect_flex_env && std::getenv("LLAMA_FLEX_MAX_AHEAD") != nullptr;
        const bool flex_threads_explicit = governor_respect_flex_env && std::getenv("LLAMA_FLEX_THREADS") != nullptr;
        const bool flex_lock_explicit    = governor_respect_flex_env && std::getenv("LLAMA_FLEX_LOCK_GB") != nullptr;
        const bool flex_pin_explicit     = governor_respect_flex_env && std::getenv("LLAMA_FLEX_PIN_POLICY") != nullptr;
        const bool flex_rebalance_explicit = governor_respect_flex_env && std::getenv("LLAMA_FLEX_GLOBAL_REBALANCE") != nullptr;
        const bool flex_sched_explicit   = governor_respect_flex_env && std::getenv("LLAMA_FLEX_SCHED") != nullptr;
        const bool flex_auto             = std::getenv("LLAMA_FLEX_AUTO") &&
                std::atoi(std::getenv("LLAMA_FLEX_AUTO")) > 0;
        const bool flex_sched            = flex_sched_explicit
                ? std::atoi(std::getenv("LLAMA_FLEX_SCHED")) > 0
                : !model_has_moe;
        LLAMA_LOG_INFO("%s: flex planner mode sched=%d auto=%d governor=%d moe=%d\n",
                __func__,
                flex_sched ? 1 : 0,
                flex_auto ? 1 : 0,
                memory_governor_requested ? 1 : 0,
                model_has_moe ? 1 : 0);

        if (flex_ring_explicit)    { const char * v = std::getenv("LLAMA_FLEX_RING"); fp.ring_layers    = std::max(2, std::atoi(v)); }
        if (flex_ahead_explicit)   { const char * v = std::getenv("LLAMA_FLEX_AHEAD"); fp.prefetch_ahead = std::max(1, std::atoi(v)); }
        if (flex_max_ahead_explicit) {
            const char * v = std::getenv("LLAMA_FLEX_MAX_AHEAD");
            fp.prefetch_ahead_max = std::max(1, std::atoi(v));
        }
        if (governor_respect_flex_env && std::getenv("LLAMA_FLEX_ADAPTIVE_AHEAD")) {
            const char * v = std::getenv("LLAMA_FLEX_ADAPTIVE_AHEAD");
            fp.adaptive_ahead = std::atoi(v) > 0;
        }
        if (flex_threads_explicit) { const char * v = std::getenv("LLAMA_FLEX_THREADS"); fp.io_threads = std::max(1, std::atoi(v)); }
        if (flex_lock_explicit) {
            const char * v = std::getenv("LLAMA_FLEX_LOCK_GB");
            fp.lock_bytes = (size_t)(std::max(0.0, std::atof(v)) * 1024.0 * 1024.0 * 1024.0);
        }

        // Adaptive ring: size the resident layer ring from available memory, so we
        // stream only as much as needed to fit (dense analogue of moe-buffer AUTO).
        //   available >= model    -> ring = n_layers (all resident, no streaming)
        //   available <  model    -> ring = (available - non_layer - reserve) / max_layer
        // A pre-scan sums layer-tensor bytes (largest layer sizes the ring slot) and
        // non-layer bytes (embedding/output/norm: stay mmap-resident, the OOM floor).
        if (flex_auto || flex_sched) {
            struct flex_plan_tensor {
                size_t size = 0;
                size_t value = 0;
            };
            std::vector<size_t> layer_bytes(hparams.n_layer, 0);
            std::vector<std::vector<flex_plan_tensor>> layer_tensors(hparams.n_layer);
            size_t non_layer = 0, layer_total = 0;
            for (const auto & it : ml.weights_map) {
                const size_t nb = ggml_nbytes(it.second.tensor);
                int layer = -1;
                if (std::sscanf(it.first.c_str(), "blk.%d.", &layer) == 1 &&
                        layer >= 0 && layer < (int) hparams.n_layer) {
                    layer_bytes[layer] += nb;
                    layer_total        += nb;
                    layer_tensors[layer].push_back({ nb, nb });
                } else {
                    non_layer += nb;
                }
            }
            size_t max_layer = 1;
            for (size_t b : layer_bytes) max_layer = std::max(max_layer, b);

            const size_t avail   = llama_detect_available_memory();
            uint64_t kv_tokens = std::max<uint32_t>(1, hparams.n_layer);
            if (const char * tokens = std::getenv("LLAMA_MEMORY_PLANNER_KV_TOKENS")) {
                kv_tokens = (uint64_t) std::max(1.0, std::atof(tokens));
            }
            double kv_factor = 1.0;
            if (const char * factor = std::getenv("LLAMA_MEMORY_PLANNER_KV_FACTOR")) {
                kv_factor = std::max(0.0, std::atof(factor));
            }
            size_t kv_bytes_per_token = 0;
            for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                kv_bytes_per_token +=
                    ((size_t) hparams.n_head_kv(il) *
                     ((size_t) hparams.n_embd_head_k(il) + (size_t) hparams.n_embd_head_v(il)) *
                     sizeof(uint16_t));
            }
            size_t kv_reserve = (size_t) ((double) kv_bytes_per_token * (double) kv_tokens * kv_factor);
            if (const char * kv_mb = std::getenv("LLAMA_MEMORY_PLANNER_KV_RESERVE_MB")) {
                kv_reserve = (size_t) (std::max(0.0, std::atof(kv_mb)) * 1024.0 * 1024.0);
            }
            size_t runtime_guard = max_layer;
            if (const char * guard_mb = std::getenv("LLAMA_MEMORY_PLANNER_RUNTIME_GUARD_MB")) {
                runtime_guard = (size_t) (std::max(0.0, std::atof(guard_mb)) * 1024.0 * 1024.0);
            }
            size_t reserve = runtime_guard + kv_reserve;
            if (const char * guard_mb = std::getenv("LLAMA_FLEX_HARD_GUARD_MB")) {
                reserve = (size_t) (std::max(0.0, std::atof(guard_mb)) * 1024.0 * 1024.0);
            }
            if (flex_sched) {
                if (!flex_threads_explicit) {
                    fp.io_threads = 4;
                }
                if (!flex_pin_explicit) {
                    fp.pin_policy = "cost-aware";
                }
                if (!flex_rebalance_explicit) {
                    fp.global_rebalance = true;
                }
                fp.memory_budget_bytes = avail == SIZE_MAX ? 0 : avail;
                fp.fixed_bytes = non_layer + reserve;
                fp.sched_auto = false;

                auto round_page = [](size_t v) {
                    const size_t page = 4096;
                    return v == 0 ? page : ((v + page - 1) / page) * page;
                };
                auto solve_layer_lock = [](const std::vector<flex_plan_tensor> & tensors, size_t budget) {
                    if (budget == 0 || tensors.empty()) {
                        return (size_t) 0;
                    }
                    if (tensors.size() > 22) {
                        std::vector<flex_plan_tensor> sorted = tensors;
                        std::sort(sorted.begin(), sorted.end(),
                                [](const flex_plan_tensor & a, const flex_plan_tensor & b) {
                                    if (a.value != b.value) {
                                        return a.value > b.value;
                                    }
                                    return a.size < b.size;
                                });
                        size_t used = 0;
                        for (const auto & t : sorted) {
                            if (used + t.size <= budget) {
                                used += t.size;
                            }
                        }
                        return used;
                    }

                    const uint64_t masks = 1ull << tensors.size();
                    size_t best_value = 0;
                    size_t best_used = 0;
                    for (uint64_t mask = 1; mask < masks; ++mask) {
                        size_t used = 0;
                        size_t value = 0;
                        bool ok = true;
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            if ((mask & (1ull << i)) == 0) {
                                continue;
                            }
                            used += tensors[i].size;
                            if (used > budget) {
                                ok = false;
                                break;
                            }
                            value += tensors[i].value;
                        }
                        if (ok && (value > best_value || (value == best_value && used > best_used))) {
                            best_value = value;
                            best_used = used;
                        }
                    }
                    return best_used;
                };
                if (avail != SIZE_MAX && hparams.n_layer > 0 && max_layer > 0) {
                    const size_t fixed_bytes = non_layer + reserve;
                    double bw_mib_s = 0.0;
                    if (const char * bw = std::getenv("LLAMA_FLEX_PLANNER_BW_MBPS")) {
                        bw_mib_s = std::max(1.0, std::atof(bw));
                    }
#if defined(__unix__) || defined(__APPLE__)
                    if (bw_mib_s <= 0.0 && !ml.files.empty()) {
                        const size_t probe = std::min<size_t>(4ull * 1024ull * 1024ull, model_weight_bytes);
                        void * buf = nullptr;
                        if (probe > 0 && posix_memalign(&buf, 4096, round_page(probe)) == 0 && buf != nullptr) {
                            const uint64_t t0 = ggml_time_us();
                            const ssize_t got = pread(ml.files[0]->file_id(), buf, probe, 0);
                            const uint64_t dt = ggml_time_us() - t0;
                            if (got > 0 && dt > 0) {
                                bw_mib_s = ((double) got / 1048576.0) / ((double) dt / 1000000.0);
                            }
                            free(buf);
                        }
                    }
#endif
                    double compute_ms = 0.0;
                    if (const char * compute = std::getenv("LLAMA_FLEX_PLANNER_COMPUTE_MS")) {
                        compute_ms = std::max(0.1, std::atof(compute));
                    }
                    if (compute_ms <= 0.0) {
                        const double ops_proxy =
                            (double) std::max<size_t>(1, layer_total) /
                            (double) std::max<uint32_t>(1, hparams.n_layer);
                        compute_ms = std::max(0.1, ops_proxy / (1024.0 * 1024.0 * 1024.0));
                    }

                    const double io_ms =
                        bw_mib_s > 0.0
                            ? ((double) layer_total / 1048576.0) / bw_mib_s * 1000.0
                            : 0.0;
                    const double rho = compute_ms > 0.0 ? io_ms / compute_ms : 0.0;
                    int planned_ahead = 1;
                    int planned_ring = 2;
                    if (rho < 0.5) {
                        planned_ahead = 3;
                        planned_ring = 5;
                    } else if (rho <= 2.0) {
                        planned_ahead = 2;
                        planned_ring = 4;
                    }
                    planned_ring = std::min<int>(planned_ring, (int) hparams.n_layer);
                    planned_ahead = std::min<int>(planned_ahead, std::max(1, planned_ring - 2));

                    size_t slot = round_page(max_layer);
                    size_t lock_budget = 0;
                    size_t locked = 0;
                    size_t stream = layer_total;
                    size_t dense_lock_auto_old = 0;
                    size_t dense_lock_candidate_count = 0;
                    double dense_lock_risk_ratio = 0.0;
                    size_t dense_expected_stream_saved = 0;
                    const char * dense_lock_budget_reason = "legacy_safe_budget";
                    const char * dense_lock_reject_reason = "none";
                    auto estimate_locked_stream = [&](size_t candidate_lock, size_t & out_locked, size_t & out_stream, size_t & out_slot) {
                        const size_t per_layer =
                            candidate_lock / (size_t) std::max<uint32_t>(1, hparams.n_layer);
                        out_locked = 0;
                        out_stream = 0;
                        size_t max_stream = 0;
                        for (size_t il = 0; il < layer_tensors.size(); ++il) {
                            const size_t layer_locked = solve_layer_lock(layer_tensors[il], per_layer);
                            const size_t layer_stream = layer_bytes[il] > layer_locked
                                ? layer_bytes[il] - layer_locked : 0;
                            out_locked += layer_locked;
                            out_stream += layer_stream;
                            max_stream = std::max(max_stream, layer_stream);
                        }
                        out_slot = round_page(max_stream);
                    };
                    for (int iter = 0; iter < 4; ++iter) {
                        const size_t ring_bytes = slot * (size_t) planned_ring;
                        const size_t available_lock =
                            avail > fixed_bytes + ring_bytes
                                ? avail - fixed_bytes - ring_bytes
                                : 0;
                        lock_budget = flex_lock_explicit
                            ? std::min(fp.lock_bytes, available_lock)
                            : available_lock;
                        const size_t per_layer =
                            lock_budget / (size_t) std::max<uint32_t>(1, hparams.n_layer);
                        locked = 0;
                        stream = 0;
                        size_t max_stream = 0;
                        for (size_t il = 0; il < layer_tensors.size(); ++il) {
                            const size_t layer_locked = solve_layer_lock(layer_tensors[il], per_layer);
                            const size_t layer_stream = layer_bytes[il] > layer_locked
                                ? layer_bytes[il] - layer_locked : 0;
                            locked += layer_locked;
                            stream += layer_stream;
                            max_stream = std::max(max_stream, layer_stream);
                        }
                        slot = round_page(max_stream);
                    }
                    dense_lock_auto_old = lock_budget;
                    dense_lock_candidate_count = 1;
                    if (!flex_lock_explicit && planner_cgroup.valid && planner_cgroup.max_bytes > 0 &&
                            layer_total > 0 && stream > layer_total / 3) {
                        const double risk_limit = 0.88;
                        const size_t cgroup_cap = (size_t) ((double) planner_cgroup.max_bytes * risk_limit);
                        const size_t kv_guard = kv_reserve;
                        const size_t soft_non_layer = non_layer / 4;
                        const size_t hard_guard = std::max({ runtime_guard, kv_guard, max_layer });
                        const size_t soft_fixed = soft_non_layer + hard_guard;
                        const size_t ring_bytes = slot * (size_t) planned_ring;
                        size_t risk_lock_cap = 0;
                        if (cgroup_cap > planner_cgroup.current_bytes + soft_fixed + ring_bytes) {
                            risk_lock_cap = cgroup_cap - planner_cgroup.current_bytes - soft_fixed - ring_bytes;
                        }
                        risk_lock_cap = std::min(risk_lock_cap, layer_total);
                        dense_lock_candidate_count = 2;
                        if (risk_lock_cap > lock_budget) {
                            size_t cand_locked = 0;
                            size_t cand_stream = layer_total;
                            size_t cand_slot = slot;
                            estimate_locked_stream(risk_lock_cap, cand_locked, cand_stream, cand_slot);
                            const size_t saved = stream > cand_stream ? stream - cand_stream : 0;
                            const size_t added = risk_lock_cap - lock_budget;
                            const bool useful = saved > 0 && added > 0 && saved * 4 >= added;
                            if (useful) {
                                lock_budget = risk_lock_cap;
                                locked = cand_locked;
                                stream = cand_stream;
                                slot = cand_slot;
                                dense_expected_stream_saved = saved;
                                dense_lock_budget_reason = "risk_aware_stream_roi";
                            } else {
                                dense_lock_reject_reason = "low_stream_roi";
                            }
                        } else {
                            dense_lock_reject_reason = "risk_cap";
                        }
                        const size_t resident_estimate = planner_cgroup.current_bytes + soft_fixed +
                            slot * (size_t) planned_ring + lock_budget;
                        dense_lock_risk_ratio = planner_cgroup.max_bytes > 0
                            ? (double) resident_estimate / (double) planner_cgroup.max_bytes
                            : 0.0;
                    }
                    if (!flex_ring_explicit) {
                        fp.ring_layers = planned_ring;
                    }
                    if (!flex_ahead_explicit) {
                        fp.prefetch_ahead = planned_ahead;
                    }
                    if (!flex_max_ahead_explicit) {
                        fp.prefetch_ahead_max = planned_ahead;
                    }
                    if (!flex_lock_explicit) {
                        fp.lock_bytes = lock_budget;
                    }
                    fp.planner_applied = true;
                    std::fprintf(stderr, "%s: memory planner model=dense auto=%d planner_v2=%d memory_max=%.0f MiB "
                            "memory_current=%.0f MiB floor=%.0f MiB available_for_dynamic=%.0f MiB "
                            "ring=%d ahead=%d lock=%.0f MiB locked=%.0f MiB "
                            "stream_per_token=%.0f MiB slot=%.0f MiB rho=%.3f bw=%.1f MiB/s "
                            "dense_lock_auto_old=%.0f MiB dense_lock_candidate_count=%zu "
                            "dense_lock_risk_ratio=%.3f dense_expected_stream_saved=%.0f MiB "
                            "dense_lock_budget_reason=%s dense_lock_reject_reason=%s\n",
                            __func__,
                            memory_governor_auto_backends ? 1 : 0,
                            planner_cgroup.valid ? 1 : 0,
                            planner_cgroup.valid ? planner_cgroup.max_bytes / 1048576.0 : -1.0,
                            planner_cgroup.valid ? planner_cgroup.current_bytes / 1048576.0 : -1.0,
                            fixed_bytes / 1048576.0,
                            avail == SIZE_MAX ? -1.0 : (avail > fixed_bytes ? (avail - fixed_bytes) / 1048576.0 : 0.0),
                            fp.ring_layers,
                            fp.prefetch_ahead,
                            fp.lock_bytes / 1048576.0,
                            locked / 1048576.0,
                            stream / 1048576.0,
                            slot / 1048576.0,
                            rho,
                            bw_mib_s,
                            dense_lock_auto_old / 1048576.0,
                            dense_lock_candidate_count,
                            dense_lock_risk_ratio,
                            dense_expected_stream_saved / 1048576.0,
                            dense_lock_budget_reason,
                            dense_lock_reject_reason);
                    LLAMA_LOG_INFO("%s: memory planner model=dense auto=%d planner_v2=%d memory_max=%.0f MiB "
                            "memory_current=%.0f MiB floor=%.0f MiB available_for_dynamic=%.0f MiB "
                            "ring=%d ahead=%d lock=%.0f MiB locked=%.0f MiB "
                            "stream_per_token=%.0f MiB slot=%.0f MiB rho=%.3f bw=%.1f MiB/s "
                            "dense_lock_auto_old=%.0f MiB dense_lock_candidate_count=%zu "
                            "dense_lock_risk_ratio=%.3f dense_expected_stream_saved=%.0f MiB "
                            "dense_lock_budget_reason=%s dense_lock_reject_reason=%s\n",
                            __func__,
                            memory_governor_auto_backends ? 1 : 0,
                            planner_cgroup.valid ? 1 : 0,
                            planner_cgroup.valid ? planner_cgroup.max_bytes / 1048576.0 : -1.0,
                            planner_cgroup.valid ? planner_cgroup.current_bytes / 1048576.0 : -1.0,
                            fixed_bytes / 1048576.0,
                            avail == SIZE_MAX ? -1.0 : (avail > fixed_bytes ? (avail - fixed_bytes) / 1048576.0 : 0.0),
                            fp.ring_layers,
                            fp.prefetch_ahead,
                            fp.lock_bytes / 1048576.0,
                            locked / 1048576.0,
                            stream / 1048576.0,
                            slot / 1048576.0,
                            rho,
                            bw_mib_s,
                            dense_lock_auto_old / 1048576.0,
                            dense_lock_candidate_count,
                            dense_lock_risk_ratio,
                            dense_expected_stream_saved / 1048576.0,
                            dense_lock_budget_reason,
                            dense_lock_reject_reason);
                    if (fp.debug_log) {
                        const double stream_mib = (double) stream / 1048576.0;
                        const double pred_io_ms =
                            bw_mib_s > 0.0 ? stream_mib / bw_mib_s * 1000.0 : 0.0;
                        std::fprintf(stderr,
                                "%s: flex HMB dense ring=%d ahead=%d lock=%.0f MiB "
                                "locked=%.0f MiB stream/token=%.0f MiB slot=%.0f MiB "
                                "rho=%.3f bw=%.1f MiB/s io=%.2f ms compute=%.2f ms "
                                "(avail=%.0f MiB fixed=%.0f MiB non_layer=%.0f MiB runtime_guard=%.0f MiB kv_reserve=%.0f MiB)\n", __func__,
                                fp.ring_layers, fp.prefetch_ahead,
                                fp.lock_bytes / 1048576.0,
                                locked / 1048576.0,
                                stream / 1048576.0,
                                slot / 1048576.0,
                                rho, bw_mib_s, pred_io_ms, compute_ms,
                                avail / 1048576.0,
                                fixed_bytes / 1048576.0,
                                non_layer / 1048576.0,
                                runtime_guard / 1048576.0,
                                kv_reserve / 1048576.0);
                        LLAMA_LOG_INFO("%s: flex HMB dense ring=%d ahead=%d lock=%.0f MiB "
                                "locked=%.0f MiB stream/token=%.0f MiB slot=%.0f MiB "
                                "rho=%.3f bw=%.1f MiB/s io=%.2f ms compute=%.2f ms "
                                "(avail=%.0f MiB fixed=%.0f MiB non_layer=%.0f MiB runtime_guard=%.0f MiB kv_reserve=%.0f MiB)\n", __func__,
                                fp.ring_layers, fp.prefetch_ahead,
                                fp.lock_bytes / 1048576.0,
                                locked / 1048576.0,
                                stream / 1048576.0,
                                slot / 1048576.0,
                                rho, bw_mib_s, pred_io_ms, compute_ms,
                                avail / 1048576.0,
                                fixed_bytes / 1048576.0,
                                non_layer / 1048576.0,
                                runtime_guard / 1048576.0,
                                kv_reserve / 1048576.0);
                        }
                }
            }

            int ring = fp.ring_layers;
            if (!fp.planner_applied && flex_auto && !fp.sched_auto && !flex_ring_explicit) {
                ring = (int) hparams.n_layer;             // default: all resident
            }
            if (!fp.planner_applied && flex_auto && !fp.sched_auto && !flex_ring_explicit &&
                    avail != SIZE_MAX && avail < non_layer + layer_total + reserve) {
                const size_t fixed = non_layer + reserve;
                const size_t room  = avail > fixed ? avail - fixed : 0;
                ring = (int) std::min<size_t>(hparams.n_layer,
                        std::max<size_t>(fp.prefetch_ahead + 2, room / max_layer));
                fp.ring_layers = ring;
            }
            if (fp.debug_log) {
                LLAMA_LOG_INFO("%s: flex adaptive ring=%d/%d sched=%d (avail=%.0f MiB, "
                        "max_layer=%.0f MiB, non_layer=%.0f MiB, lock=%.0f MiB, ahead=%d, "
                        "adaptive_ahead=%d, max_ahead=%d, threads=%d, pin=%s)\n", __func__,
                        fp.ring_layers, (int) hparams.n_layer, fp.sched_auto ? 1 : 0,
                        avail == SIZE_MAX ? -1.0 : avail / 1048576.0,
                        max_layer / 1048576.0, non_layer / 1048576.0,
                        fp.lock_bytes / 1048576.0, fp.prefetch_ahead,
                        fp.adaptive_ahead ? 1 : 0, fp.prefetch_ahead_max,
                        fp.io_threads, fp.pin_policy.c_str());
            }
        }

        std::vector<int> fds;
        for (const auto & f : ml.files) {
            fds.push_back(f->file_id());
        }

        pimpl->flex = llama_flex_create(fds, hparams.n_layer, fp);
        if (llama_flex_enabled(pimpl->flex.get())) {
            int registered = 0;
            for (const auto & it : ml.weights_map) {
                int layer = -1;
                if (std::sscanf(it.first.c_str(), "blk.%d.", &layer) != 1 ||
                        layer < 0 || layer >= hparams.n_layer) {
                    continue; // only stream per-decoder-layer weights
                }
                llama_flex_tensor t;
                t.name        = it.first;
                t.file_idx    = it.second.idx;
                t.file_offset = it.second.offs;
                t.size        = ggml_nbytes(it.second.tensor);
                llama_flex_register_tensor(*pimpl->flex, layer, t);
                ++registered;
            }
            llama_flex_finalize(*pimpl->flex);
            if (fp.debug_log) {
                LLAMA_LOG_INFO("%s: llama-flex enabled, streaming %d layer tensors\n",
                        __func__, registered);
            }
        }
    }

    return true;
}
```

### 推理期 CPU hook：Dense Flex 与 MoE Buffer 互斥接入

来源：`src/llama-context.cpp:2407-2432`

```cpp
    // Per-node weight-stream hook on the CPU backend for the duration of this
    // graph. Used by FlexInfer-style dense streaming (flex) or MoE expert
    // streaming (moe-buffer); the two are mutually exclusive.
    auto * flex = model.get_flex_context();
    auto * moe  = model.get_moe_buffer_context();
    const bool flex_active = backend_cpu != nullptr && llama_flex_enabled(flex);
    const bool moe_active  = backend_cpu != nullptr && !flex_active && llama_moe_buffer_enabled(moe);
    if (flex_active) {
        llama_flex_graph_begin(*flex);
        ggml_cpu_set_weight_stream_callback(llama_flex_stream_callback, flex);
        ggml_cpu_set_op_override_callback(nullptr, nullptr);
    } else if (moe_active) {
        ggml_cpu_set_weight_stream_callback(llama_moe_buffer_stream_callback, moe);
        ggml_cpu_set_op_override_callback(llama_moe_buffer_mul_mat_id_callback, moe);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    if (flex_active || moe_active) {
        ggml_backend_sched_synchronize(sched.get());
        ggml_cpu_set_weight_stream_callback(nullptr, nullptr);
        ggml_cpu_set_op_override_callback(nullptr, nullptr);
    }
```

## 16.1 Dense Flex 完整关键代码

### Dense Flex 对外参数/统计/API 声明

来源：`src/llama-flex.h:1-218`

```cpp
#pragma once

// FlexInfer-style streaming weight loader.
//
// Unlike the mmap-based llama-window (which only advises the kernel and leaves
// residency to the page cache), llama-flex never mmaps weights. It keeps a small
// ring of per-layer host buffers and streams each layer's tensors in from the
// file with direct, large, sequential reads on background IO threads, releasing
// a slot as soon as the compute has consumed the layer. The resident weight
// footprint is therefore bounded by the ring size (k layers), independent of the
// page cache:  RSS_weights ~= (k / n_layers) * model_size.
//
// This module (stage 2a) is self-contained and testable without ggml: it owns
// the file descriptors, the layer/tensor metadata and the ring buffer, and
// exposes a "request layer -> wait ready -> get tensor ptr -> release layer"
// API. Integration with the compute path is layered on top separately.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_flex_tensor {
    std::string name;
    uint16_t    file_idx    = 0;
    size_t      file_offset = 0;  // byte offset of tensor data in the file
    size_t      size        = 0;  // tensor data size in bytes
    size_t      buf_offset  = 0;  // offset within the ring slot (streamed) or lock buffer (locked)
    bool        locked      = false; // balanced-locked: resident permanently, never streamed
    bool        delta_locked = false; // runtime-pinned into a delta lock buffer
    size_t      delta_buf_index = 0;
    size_t      delta_buf_offset = 0;
};

struct llama_flex_params {
    bool   enabled        = false;
    bool   direct_io      = true; // use O_DIRECT for streaming reads when possible
    bool   debug_log      = false;
    bool   sched_auto     = false; // cap-aware scheduler: tune ring after pinning
    bool   planner_applied = false; // load-time global planner selected ring/ahead/lock
    bool   adaptive_ahead = true;  // runtime prefetch-depth controller
    int    ring_layers    = 4;    // k: number of layer slots kept resident
    int    prefetch_ahead = 2;    // initial/fixed layers ahead to stream
    int    prefetch_ahead_max = 8; // max runtime ahead when adaptive_ahead is enabled
    int    io_threads     = 4;    // background streaming threads
    size_t lock_bytes     = 0;    // balanced-locking budget (stage 2c); 0 = off
    size_t memory_budget_bytes = 0; // cgroup/MemAvailable budget for sched_auto
    size_t fixed_bytes    = 0;    // non-flex memory reserve for sched_auto
    size_t read_cost_bytes = 0;   // per-stream-read fixed cost in equivalent bytes for cost-aware pinning
    bool   read_cost_auto = false; // estimate read_cost_bytes from startup pread latency/bandwidth calibration
    bool   global_rebalance = false; // spend leftover per-layer lock budget on high stream-cost layers
    std::string pin_policy = "small-first"; // small-first, large-first, attn-first, ffn-first, cost-aware, none
};

struct llama_flex_stats {
    uint64_t layer_loads     = 0;
    uint64_t layer_hits      = 0;  // requested while already resident
    uint64_t wait_events     = 0;
    uint64_t bytes_streamed  = 0;  // logical tensor bytes delivered to slots
    uint64_t bytes_read_phys = 0;  // physical bytes pread() from disk (incl O_DIRECT alignment padding)
    uint64_t read_ops        = 0;  // number of pread() calls issued for streaming
    uint64_t total_io_us     = 0;
    uint64_t total_wait_us   = 0;
    double   ewma_io_us      = 0.0;
    double   ewma_compute_us = 0.0;
    double   ewma_wait_us    = 0.0;
    uint64_t demand_loads    = 0;  // layer was not already queued/loading when compute needed it
    uint64_t prefetch_queued = 0;
    uint64_t prefetch_budget_dropped = 0;
    uint64_t queue_requeues  = 0;  // IO worker could not acquire a slot
    uint64_t evictions       = 0;
    uint64_t releases        = 0;
    uint64_t graphs          = 0;
    uint64_t ahead_adjustments = 0;
    uint64_t locked_tensors  = 0;
    uint64_t streamed_tensors = 0;
    uint64_t delta_locked_tensors = 0;
    uint64_t delta_pin_attempts = 0;
    uint64_t delta_pin_failures = 0;
    uint64_t delta_pin_async_submitted = 0;
    uint64_t delta_pin_async_completed = 0;
    uint64_t delta_pin_async_rejected = 0;
    uint64_t delta_pin_async_pending = 0;
    uint64_t delta_pin_async_last_io_us = 0;
    uint64_t delta_pin_async_last_elapsed_us = 0;
    size_t   ring_bytes      = 0;  // total bytes held by the ring
    size_t   slot_bytes      = 0;  // bytes charged for one streamed layer slot
    size_t   locked_bytes    = 0;  // bytes pinned by balanced locking
    size_t   delta_locked_bytes = 0;
    size_t   delta_pin_saved_per_token = 0;
    size_t   lock_budget_unused = 0;
    size_t   stream_per_token = 0; // unlocked bytes that must be read each token
    int      effective_ahead = 0;
    int      min_effective_ahead = 0;
    int      max_effective_ahead = 0;
    size_t   sched_budget_bytes = 0;
    size_t   sched_fixed_bytes = 0;
    size_t   sched_ring_room = 0;
    size_t   read_cost_bytes = 0;
    size_t   global_rebalance_bytes = 0;
    uint64_t global_rebalance_tensors = 0;
    size_t   prefetch_budget_bytes = 0;
    size_t   prefetch_budget_available_bytes = 0;
};

struct llama_flex_reclaim_result {
    uint64_t released_bytes = 0;
    uint32_t released_layers = 0;
    bool target_satisfied = false;
};

struct llama_flex_resize_result {
    bool attempted = false;
    bool changed = false;
    int old_slots = 0;
    int new_slots = 0;
    uint64_t old_bytes = 0;
    uint64_t new_bytes = 0;
    const char * reason = "none";
};

struct llama_flex_delta_pin_result {
    bool attempted = false;
    bool changed = false;
    uint64_t requested_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t saved_per_token_bytes = 0;
    uint64_t candidates = 0;
    uint64_t pinned_tensors = 0;
    uint64_t io_us = 0;
    uint64_t elapsed_us = 0;
    double roi = 0.0;
    const char * reason = "none";
};

struct ggml_tensor;
struct llama_flex_context;

// Create a streaming context. Each fd is dup()'d and reopened for streaming
// reads (with O_DIRECT when params.direct_io and the fs supports it). n_layers
// is the number of decoder layers. Returns a disabled context
// (llama_flex_enabled()==false) when params.enabled is false.
std::shared_ptr<llama_flex_context> llama_flex_create(
        const std::vector<int> &  fds,
        int                       n_layers,
        const llama_flex_params & params);

bool llama_flex_enabled(const llama_flex_context * ctx);

// Register one tensor as belonging to a decoder layer. Must be called for every
// streamed tensor before any layer is requested. layer_id in [0, n_layers).
void llama_flex_register_tensor(
        llama_flex_context & ctx,
        int                  layer_id,
        const llama_flex_tensor & tensor);

// Finalize registration: sizes each layer slot to the largest layer and
// allocates the ring. Must be called once after all tensors are registered.
void llama_flex_finalize(llama_flex_context & ctx);

// Ask the IO threads to stream a layer into the ring (non-blocking).
void llama_flex_request_layer(llama_flex_context & ctx, int layer_id);

// Block until a layer is resident in the ring.
void llama_flex_wait_layer(llama_flex_context & ctx, int layer_id);

// Return a pointer to a tensor's data inside the ring. The layer must be
// resident (call llama_flex_wait_layer first). Returns nullptr on miss.
void * llama_flex_get_tensor(llama_flex_context & ctx, int layer_id, const std::string & name);

// Mark a layer as consumed; its slot may be reused for future prefetches.
void llama_flex_release_layer(llama_flex_context & ctx, int layer_id);

const llama_flex_stats & llama_flex_get_stats(const llama_flex_context & ctx);

// Set a per-tick speculative prefetch budget. A zero budget disables the gate.
// Demand loads are never gated by this API.
void llama_flex_set_prefetch_budget(
        llama_flex_context & ctx,
        uint64_t             budget_bytes);

// Release already-consumed resident layer slots back to the OS with
// MADV_DONTNEED. This never touches the current compute layer; it only reclaims
// layers that were previously marked released by llama_flex_release_layer().
llama_flex_reclaim_result llama_flex_reclaim_released(
        llama_flex_context & ctx,
        uint64_t             target_bytes,
        uint32_t             max_layers);

// Resize the streaming ring. Growing is immediate. Shrinking only removes free
// slots so active/resident layers are never invalidated.
llama_flex_resize_result llama_flex_resize_ring(
        llama_flex_context & ctx,
        int                 target_slots);

llama_flex_delta_pin_result llama_flex_delta_pin(
        llama_flex_context & ctx,
        uint64_t             budget_bytes,
        double               min_roi);

llama_flex_delta_pin_result llama_flex_delta_pin_async(
        llama_flex_context & ctx,
        uint64_t             budget_bytes,
        double               min_roi);

// --- compute-path integration ---------------------------------------------

// Reset per-graph state and kick off the initial prefetch (layers [0, ahead]).
// Call once before each decode graph.
void llama_flex_graph_begin(llama_flex_context & ctx);

// ggml_cpu_weight_stream_callback. user_data must be a llama_flex_context*.
// On ith==0 it streams the op's flex-managed weight(s) into the ring (waiting
// if necessary), repoints their ->data, and drives prefetch/release at layer
// boundaries. Returns true iff the op owns flex-managed weights (so the CPU
// backend issues a barrier before running the kernel).
bool llama_flex_stream_callback(struct ggml_tensor * op, int ith, void * user_data);
```

### Dense Flex 内部状态：llama_flex_context

来源：`src/llama-flex.cpp:72-185`

```cpp
struct llama_flex_context {
    llama_flex_params params;
    llama_flex_stats  stats;
    int n_layers = 0;

    std::vector<int>         fds;          // one fd per file
    std::vector<size_t>      file_sizes;   // size of each file (for EOF clamping)
    std::vector<flex_layer>  layers;
    std::unordered_map<std::string, int> name_layer;  // tensor name -> layer id

    bool   direct_io_active = false;       // O_DIRECT actually in effect
    size_t align            = 4096;        // direct-IO alignment (offset/buf/len)
    size_t max_tensor       = 0;           // largest single tensor (bounce sizing)

    // per-graph compute state (written only by ith==0 in the stream callback)
    int cur_compute_layer = -1;
    uint64_t graph_id = 0;
    uint64_t last_layer_enter_us = 0;

    // ring of slots
    size_t                   slot_bytes = 0;
    std::vector<void *>      slots;
    std::vector<int>         slot_layer;   // which layer occupies slot, or -1

    // persistent balanced-lock buffer (resident for the whole run)
    void *                   lock_buf  = nullptr;
    size_t                   lock_size = 0;
    std::vector<flex_delta_lock_buffer> delta_lock_bufs;

    std::vector<std::thread> workers;
    std::thread              delta_pin_worker;
    std::deque<int>          queue;        // layer ids to stream
    std::deque<flex_delta_pin_job> delta_pin_queue;
    std::mutex               mutex;
    std::condition_variable  cv_work;      // wakes IO threads
    std::condition_variable  cv_delta_pin; // wakes runtime delta-pin worker
    std::condition_variable  cv_ready;     // wakes waiters on layer-ready
    bool                     shutdown = false;
    FILE *                   trace = nullptr;

    // Adaptive prefetch-depth controller. IO EWMA is updated by workers under
    // mutex; compute/wait EWMAs are updated by ith==0 on layer transitions.
    int      adaptive_ahead = 0;
    uint64_t adaptive_transitions = 0;
    uint64_t adaptive_quiet = 0;
    uint64_t adaptive_last_requeues = 0;
    double   ewma_io_us = 0.0;
    double   ewma_compute_us = 0.0;
    double   ewma_wait_us = 0.0;

    bool     prefetch_budget_enabled = false;
    uint64_t prefetch_budget_bytes = 0;
    uint64_t prefetch_budget_available_bytes = 0;

    ~llama_flex_context() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv_work.notify_all();
        cv_delta_pin.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        if (delta_pin_worker.joinable()) {
            delta_pin_worker.join();
        }
        if (params.debug_log && stats.read_ops > 0) {
            const double phys_mib = stats.bytes_read_phys / 1048576.0;
            const double log_mib  = stats.bytes_streamed  / 1048576.0;
            const double io_s     = stats.total_io_us / 1e6;
            const double bw       = io_s > 0 ? phys_mib / io_s : 0.0;          // per-thread achieved MiB/s
            const double redun    = log_mib > 0 ? (phys_mib / log_mib - 1.0) * 100.0 : 0.0;
            const double avg_read = stats.read_ops > 0 ? phys_mib * 1024.0 / stats.read_ops : 0.0; // KiB/read
            const double wait_ms  = stats.total_wait_us / 1000.0;
            const double wait_avg = stats.wait_events > 0 ? wait_ms / (double) stats.wait_events : 0.0;
            std::fprintf(stderr,
                "llama_flex IO: loads=%llu reads=%llu avg_read=%.1f KiB  logical=%.0f MiB phys=%.0f MiB "
                "align_redundancy=%.2f%% achieved_bw=%.0f MiB/s waits=%llu wait=%.0f ms avg_wait=%.2f ms "
                "demand=%llu prefetch=%llu requeue=%llu evict=%llu release=%llu graphs=%llu ahead=%d "
                "ahead_adj=%llu ahead_range=[%d,%d] io_ewma=%.2f ms compute_ewma=%.2f ms\n",
                (unsigned long long) stats.layer_loads, (unsigned long long) stats.read_ops,
                avg_read, log_mib, phys_mib, redun, bw,
                (unsigned long long) stats.wait_events, wait_ms, wait_avg,
                (unsigned long long) stats.demand_loads,
                (unsigned long long) stats.prefetch_queued,
                (unsigned long long) stats.queue_requeues,
                (unsigned long long) stats.evictions,
                (unsigned long long) stats.releases,
                (unsigned long long) stats.graphs,
                stats.effective_ahead,
                (unsigned long long) stats.ahead_adjustments,
                stats.min_effective_ahead, stats.max_effective_ahead,
                ewma_io_us / 1000.0, ewma_compute_us / 1000.0);
        }
        if (trace) {
            std::fclose(trace);
        }
        for (void * p : slots) {
            free(p);
        }
        free(lock_buf);
        for (auto & b : delta_lock_bufs) {
            free(b.ptr);
        }
        for (int fd : fds) {
            if (fd >= 0) {
                close(fd);
            }
        }
    }
};
```

### slot 获取与 LRU released victim：flex_acquire_slot()

来源：`src/llama-flex.cpp:708-736`

```cpp
static int flex_acquire_slot(llama_flex_context & ctx, int layer) {
    for (size_t s = 0; s < ctx.slots.size(); ++s) {
        if (ctx.slot_layer[s] < 0) {
            ctx.slot_layer[s] = layer;
            return (int) s;
        }
    }
    // No free slot: evict the LRU released layer.
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int l = 0; l < ctx.n_layers; ++l) {
        auto & L = ctx.layers[l];
        if (L.slot >= 0 && L.released && L.state == layer_state::resident &&
                L.last_use < oldest) {
            oldest = L.last_use;
            victim = l;
        }
    }
    if (victim < 0) {
        return -1;
    }
    int slot = ctx.layers[victim].slot;
    ctx.layers[victim].slot  = -1;
    ctx.layers[victim].state = layer_state::not_resident;
    ctx.slot_layer[slot]     = layer;
    ctx.stats.evictions++;
    flex_trace_locked(ctx, "evict", victim, slot, 0, 0, 0);
    return slot;
}
```

### O_DIRECT / pread 读取实现：flex_read()

来源：`src/llama-flex.cpp:742-776`

```cpp
static bool flex_read(llama_flex_context * ctx,
                      uint8_t * dst, uint16_t file_idx, size_t foff, size_t size,
                      uint8_t * bounce, size_t bcap, size_t * phys_out = nullptr) {
    const int fd = ctx->fds[file_idx];
    if (!ctx->direct_io_active) {
        size_t left = size; off_t off = (off_t) foff; uint8_t * d = dst;
        while (left > 0) {
            ssize_t r = pread(fd, d, left, off);
            if (r <= 0) return false;
            d += r; off += r; left -= (size_t) r;
        }
        if (phys_out) *phys_out = size;
        return true;
    }
    const size_t A    = ctx->align;
    const size_t aoff = foff & ~(A - 1);
    const size_t head = foff - aoff;
    size_t alen = head + size;
    alen = (alen + A - 1) & ~(A - 1);
    const size_t fsz  = ctx->file_sizes[file_idx];
    size_t want = alen;
    if (aoff + want > fsz) {
        want = fsz - aoff;            // final read may be a short EOF block
    }
    if (head + size > bcap || want > bcap) {
        return false;
    }
    ssize_t r = pread(fd, bounce, want, (off_t) aoff);
    if (r < 0 || (size_t) r < head + size) {
        return false;
    }
    std::memcpy(dst, bounce + head, size);
    if (phys_out) *phys_out = want;
    return true;
}
```

### 后台 IO worker：flex_worker()

来源：`src/llama-flex.cpp:852-947`

```cpp
static void flex_worker(llama_flex_context * ctx) {
    // Per-thread aligned bounce buffer for O_DIRECT reads.
    uint8_t * bounce = nullptr;
    size_t    bcap   = 0;
    if (ctx->direct_io_active) {
        bcap = ctx->max_tensor + 2 * ctx->align;
        if (posix_memalign((void **) &bounce, ctx->align, bcap) != 0) {
            bounce = nullptr; bcap = 0;
        }
    }

    for (;;) {
        int layer = -1;
        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->cv_work.wait(lock, [&] {
                return ctx->shutdown || !ctx->queue.empty();
            });
            if (ctx->shutdown && ctx->queue.empty()) {
                free(bounce);
                return;
            }
            layer = ctx->queue.front();
            ctx->queue.pop_front();

            auto & L = ctx->layers[layer];
            if (L.state == layer_state::resident || L.state == layer_state::loading) {
                continue; // already handled
            }
            int slot = flex_acquire_slot(*ctx, layer);
            if (slot < 0) {
                // No slot available right now; requeue and back off.
                ctx->stats.queue_requeues++;
                flex_trace_locked(*ctx, "requeue", layer, -1, 0, 0, 0);
                ctx->queue.push_back(layer);
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            L.slot     = slot;
            L.state    = layer_state::loading;
            L.released = false;
        }

        // Stream tensors into the slot (outside the lock).
        auto & L = ctx->layers[layer];
        uint8_t * base = (uint8_t *) ctx->slots[L.slot];
        const uint64_t t0 = now_us();
        bool ok = true;
        size_t streamed = 0;
        size_t phys     = 0;
        size_t ops      = 0;
        for (const auto & t : L.tensors) {
            if (t.locked) {
                continue; // locked tensors live permanently in the lock buffer
            }
            if (t.delta_locked) {
                continue; // runtime-pinned tensors live permanently in delta lock buffers
            }
            size_t p = 0;
            if (!flex_read(ctx, base + t.buf_offset, t.file_idx, t.file_offset, t.size, bounce, bcap, &p)) {
                ok = false;
                break;
            }
            streamed += t.size;
            phys     += p;
            ops      += 1;
        }
        const uint64_t dt = now_us() - t0;

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            if (ok) {
                L.state    = layer_state::resident;
                L.last_use = now_us();
                ctx->stats.layer_loads++;
                ctx->stats.bytes_streamed  += streamed;
                ctx->stats.bytes_read_phys += phys;
                ctx->stats.read_ops        += ops;
                ctx->stats.total_io_us     += dt;
                flex_ewma(ctx->ewma_io_us, (double) std::max<uint64_t>(dt, 1));
                flex_trace_locked(*ctx, "load", layer, L.slot, streamed, phys, dt);
            } else {
                // Failed: drop the slot back.
                ctx->slot_layer[L.slot] = -1;
                L.slot  = -1;
                L.state = layer_state::not_resident;
                flex_trace_locked(*ctx, "load_fail", layer, -1, streamed, phys, dt);
                if (ctx->params.debug_log) {
                    std::fprintf(stderr, "llama_flex: stream failed for layer %d\n", layer);
                }
            }
        }
        ctx->cv_ready.notify_all();
    }
}
```

### 创建上下文：llama_flex_create()

来源：`src/llama-flex.cpp:949-1037`

```cpp
std::shared_ptr<llama_flex_context> llama_flex_create(
        const std::vector<int> &  fds,
        int                       n_layers,
        const llama_flex_params & params) {
    auto ctx = std::make_shared<llama_flex_context>();
    ctx->params   = params;
    ctx->n_layers = n_layers;
    ctx->layers.resize(std::max(0, n_layers));
    if (const char * v = std::getenv("LLAMA_FLEX_PIN_POLICY")) {
        ctx->params.pin_policy = v;
    }
    if (const char * v = std::getenv("LLAMA_FLEX_READ_COST_KB")) {
        if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {
            ctx->params.read_cost_auto = true;
        } else {
            const double kb = std::max(0.0, std::atof(v));
            ctx->params.read_cost_bytes = (size_t) (kb * 1024.0);
        }
    }
    if (const char * v = std::getenv("LLAMA_FLEX_READ_COST_AUTO")) {
        ctx->params.read_cost_auto = std::atoi(v) > 0;
    }
    if (const char * v = std::getenv("LLAMA_FLEX_GLOBAL_REBALANCE")) {
        ctx->params.global_rebalance = std::atoi(v) > 0;
    }
    if (!flex_pin_policy_valid(ctx->params.pin_policy)) {
        if (ctx->params.debug_log) {
            std::fprintf(stderr, "llama_flex: invalid pin policy '%s', using small-first\n",
                    ctx->params.pin_policy.c_str());
        }
        ctx->params.pin_policy = "small-first";
    }

    if (!params.enabled || n_layers <= 0) {
        return ctx;
    }

    bool all_direct = params.direct_io;
    for (int src_fd : fds) {
        // Reopen via /proc/self/fd to get an independent file position and,
        // optionally, an O_DIRECT description. Fall back to a plain dup().
        int fd = -1;
        bool got_direct = false;
#if defined(__linux__)
        char proc[64];
        std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", src_fd);
        int flags = O_RDONLY;
#if defined(O_DIRECT)
        if (params.direct_io) {
            fd = open(proc, flags | O_DIRECT);
            if (fd >= 0) {
                got_direct = true;
            }
        }
#endif
        if (fd < 0) {
            fd = open(proc, flags);
        }
#endif
        if (fd < 0) {
            fd = dup(src_fd);
        }
        if (fd < 0) {
            if (params.debug_log) {
                std::fprintf(stderr, "llama_flex: failed to reopen fd %d\n", src_fd);
            }
            ctx->params.enabled = false;
            return ctx;
        }
        if (!got_direct) {
            all_direct = false;
        }
        struct stat st;
        ctx->file_sizes.push_back(fstat(fd, &st) == 0 ? (size_t) st.st_size : SIZE_MAX);
        ctx->fds.push_back(fd);
    }
    ctx->direct_io_active = all_direct;
    if (const char * path = std::getenv("LLAMA_FLEX_TRACE")) {
        ctx->trace = std::fopen(path, "w");
        if (ctx->trace != nullptr) {
            std::fprintf(ctx->trace,
                    "time_us\tevent\tgraph\tlayer\tslot\tlogical_bytes\tphys_bytes\telapsed_us\tqueue_depth\tloads\twaits\tevictions\n");
        } else if (params.debug_log) {
            std::fprintf(stderr, "llama_flex: failed to open trace %s\n", path);
        }
    }

    return ctx;
}
```

### 注册 tensor：llama_flex_register_tensor()

来源：`src/llama-flex.cpp:1043-1059`

```cpp
void llama_flex_register_tensor(
        llama_flex_context & ctx,
        int                  layer_id,
        const llama_flex_tensor & tensor) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    auto & L = ctx.layers[layer_id];
    llama_flex_tensor t = tensor;
    t.locked     = false;
    t.delta_locked = false;
    t.buf_offset = 0; // assigned in finalize once locking is decided
    L.bytes += t.size;
    ctx.max_tensor = std::max(ctx.max_tensor, t.size);
    ctx.name_layer[t.name] = layer_id;
    L.tensors.push_back(std::move(t));
}
```

### finalize 分配 ring/lock 并启动 worker：llama_flex_finalize()

来源：`src/llama-flex.cpp:1061-1260`

```cpp
void llama_flex_finalize(llama_flex_context & ctx) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (ctx.params.read_cost_auto) {
        ctx.params.read_cost_bytes = flex_calibrate_read_cost_bytes(ctx);
    }

    // Balanced memory locking: give every layer the same lock budget so the
    // per-layer streamed IO stays uniform (avoids pipeline stalls). Within a
    // layer, lock tensors in registration order until the budget is hit; since
    // all layers share the same tensor structure this locks the same set per
    // layer. The remaining (unlocked) tensors are streamed through the ring.
    const bool pin_disabled = ctx.params.pin_policy == "none";
    const size_t per_layer_lock = !pin_disabled && ctx.n_layers > 0
            ? ctx.params.lock_bytes / (size_t) ctx.n_layers : 0;

    size_t balanced_locked_total = 0;
    for (auto & L : ctx.layers) {
        // Balanced pinning keeps the same byte budget per layer so streamed IO
        // remains uniform. Policies only change tensor order within that budget.
        std::vector<bool> cost_aware_pins;
        if (flex_policy_cost_aware(ctx.params.pin_policy)) {
            cost_aware_pins = flex_choose_cost_aware_pins(ctx, L.tensors, per_layer_lock);
        } else {
            flex_sort_for_pin(ctx, L.tensors, ctx.params.pin_policy);
        }
        size_t locked_here = 0;
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            auto & t = L.tensors[i];
            const bool should_lock = flex_policy_cost_aware(ctx.params.pin_policy)
                    ? cost_aware_pins[i] : locked_here + t.size <= per_layer_lock;
            if (should_lock) {
                t.locked     = true;
                locked_here += t.size;
            } else {
                t.locked     = false;
            }
        }
        if (per_layer_lock > locked_here) {
            ctx.stats.lock_budget_unused += per_layer_lock - locked_here;
        }
        balanced_locked_total += locked_here;
    }

    if (ctx.params.global_rebalance && flex_policy_cost_aware(ctx.params.pin_policy) &&
            ctx.params.lock_bytes > balanced_locked_total) {
        flex_apply_global_rebalance(ctx, ctx.params.lock_bytes - balanced_locked_total);
    }

    size_t lock_total = 0;
    ctx.stats.locked_tensors = 0;
    ctx.stats.streamed_tensors = 0;
    ctx.stats.lock_budget_unused = 0;
    ctx.stats.stream_per_token = 0;
    for (auto & L : ctx.layers) {
        size_t stream_off = 0;
        size_t locked_here = 0;
        for (auto & t : L.tensors) {
            if (t.locked) {
                t.buf_offset = lock_total;   // offset into the global lock buffer
                lock_total  += t.size;
                locked_here += t.size;
                ctx.stats.locked_tensors++;
            } else {
                t.buf_offset = stream_off;   // offset within this layer's slot
                stream_off  += t.size;
                ctx.stats.streamed_tensors++;
            }
        }
        L.stream_bytes    = stream_off;
        L.always_resident = (stream_off == 0);
    }
    if (ctx.params.lock_bytes > lock_total) {
        ctx.stats.lock_budget_unused = ctx.params.lock_bytes - lock_total;
    }

    size_t max_bytes = 0;
    for (const auto & L : ctx.layers) {
        max_bytes = std::max(max_bytes, L.stream_bytes);
    }
    // Round the slot up to a page so it can back O_DIRECT later.
    ctx.slot_bytes = max_bytes > 0 ? ((max_bytes + page - 1) / page) * page : page;

    // Allocate and fill the persistent lock buffer.
    if (lock_total > 0) {
        if (posix_memalign(&ctx.lock_buf, page, lock_total) != 0 || ctx.lock_buf == nullptr) {
            ctx.params.enabled = false;
            if (ctx.params.debug_log) {
                std::fprintf(stderr, "llama_flex: lock buffer alloc failed (%zu bytes)\n", lock_total);
            }
            return;
        }
        ctx.lock_size = lock_total;
        // Temp aligned bounce for direct-IO lock fill (workers not started yet).
        uint8_t * bounce = nullptr;
        size_t    bcap   = 0;
        if (ctx.direct_io_active) {
            bcap = ctx.max_tensor + 2 * ctx.align;
            if (posix_memalign((void **) &bounce, ctx.align, bcap) != 0) {
                bounce = nullptr; bcap = 0;
            }
        }
        for (auto & L : ctx.layers) {
            for (const auto & t : L.tensors) {
                if (!t.locked) {
                    continue;
                }
                if (!flex_read(&ctx, (uint8_t *) ctx.lock_buf + t.buf_offset,
                               t.file_idx, t.file_offset, t.size, bounce, bcap)) {
                    ctx.params.enabled = false;
                }
            }
        }
        free(bounce);
        ctx.stats.locked_bytes = lock_total;
    }
    for (const auto & L : ctx.layers) {
        ctx.stats.stream_per_token += L.stream_bytes;
    }

    if (ctx.params.sched_auto && ctx.params.memory_budget_bytes > 0 && ctx.slot_bytes > 0) {
        const size_t used_fixed = ctx.params.fixed_bytes + ctx.stats.locked_bytes;
        const size_t room = ctx.params.memory_budget_bytes > used_fixed
                ? ctx.params.memory_budget_bytes - used_fixed : 0;
        int auto_k = (int) std::min<size_t>(ctx.n_layers,
                std::max<size_t>(1, room / ctx.slot_bytes));
        const int min_k = std::min(ctx.n_layers, std::max(2, ctx.params.prefetch_ahead + 2));
        auto_k = std::max(auto_k, min_k);
        ctx.params.ring_layers = auto_k;
        ctx.stats.sched_budget_bytes = ctx.params.memory_budget_bytes;
        ctx.stats.sched_fixed_bytes  = ctx.params.fixed_bytes;
        ctx.stats.sched_ring_room    = room;
    }
    const int k = std::max(1, std::min(ctx.params.ring_layers, ctx.n_layers));
    ctx.slots.resize(k, nullptr);
    ctx.slot_layer.assign(k, -1);
    for (int s = 0; s < k; ++s) {
        void * p = nullptr;
        if (posix_memalign(&p, page, ctx.slot_bytes) != 0 || p == nullptr) {
            ctx.params.enabled = false;
            if (ctx.params.debug_log) {
                std::fprintf(stderr, "llama_flex: slot alloc failed (%zu bytes)\n", ctx.slot_bytes);
            }
            return;
        }
        ctx.slots[s] = p;
    }
    ctx.stats.ring_bytes = ctx.slot_bytes * (size_t) k;
    ctx.stats.slot_bytes = ctx.slot_bytes;
    ctx.stats.read_cost_bytes = ctx.params.read_cost_bytes;
    ctx.adaptive_ahead = std::max(1, std::min(ctx.params.prefetch_ahead, flex_ring_room_ahead(ctx)));
    ctx.stats.effective_ahead = flex_effective_ahead(ctx);
    ctx.stats.min_effective_ahead = ctx.stats.effective_ahead;
    ctx.stats.max_effective_ahead = ctx.stats.effective_ahead;

    // Fully-locked layers never need streaming: mark them permanently resident.
    for (auto & L : ctx.layers) {
        if (L.always_resident) {
            L.state    = layer_state::resident;
            L.released = false;
        }
    }

    const int nthreads = std::max(1, std::min(ctx.params.io_threads, 8));
    for (int i = 0; i < nthreads; ++i) {
        ctx.workers.emplace_back(flex_worker, &ctx);
    }
    ctx.delta_pin_worker = std::thread(flex_delta_pin_worker, &ctx);

    if (ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_flex: layers=%d ring=%d slot=%.2f MiB ring_total=%.2f MiB "
                "locked=%.2f MiB stream/token=%.2f MiB io_threads=%d direct_io=%d ahead=%d requested_ahead=%d "
                "adaptive_ahead=%d max_ahead=%d pin_policy=%s read_cost=%.1f KiB locked_tensors=%llu streamed_tensors=%llu lock_unused=%.2f MiB "
                "global_rebalance=%d global_locked=%.2f MiB global_tensors=%llu "
                "planner=%d sched=%d budget=%.0f MiB fixed=%.0f MiB ring_room=%.0f MiB\n",
                ctx.n_layers, k, ctx.slot_bytes / 1048576.0,
                ctx.stats.ring_bytes / 1048576.0,
                ctx.stats.locked_bytes / 1048576.0,
                ctx.stats.stream_per_token / 1048576.0,
                nthreads, ctx.direct_io_active ? 1 : 0,
                ctx.stats.effective_ahead, ctx.params.prefetch_ahead,
                ctx.params.adaptive_ahead ? 1 : 0, ctx.params.prefetch_ahead_max,
                ctx.params.pin_policy.c_str(),
                ctx.params.read_cost_bytes / 1024.0,
                (unsigned long long) ctx.stats.locked_tensors,
                (unsigned long long) ctx.stats.streamed_tensors,
                ctx.stats.lock_budget_unused / 1048576.0,
                ctx.params.global_rebalance ? 1 : 0,
                ctx.stats.global_rebalance_bytes / 1048576.0,
                (unsigned long long) ctx.stats.global_rebalance_tensors,
                ctx.params.planner_applied ? 1 : 0,
                ctx.params.sched_auto ? 1 : 0,
                ctx.stats.sched_budget_bytes / 1048576.0,
                ctx.stats.sched_fixed_bytes / 1048576.0,
                ctx.stats.sched_ring_room / 1048576.0);
    }
}
```

### request/wait/get/release 闭环

来源：`src/llama-flex.cpp:1262-1372`

```cpp
static void flex_request_layer(llama_flex_context & ctx, int layer_id, bool prefetch) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    if (L.state == layer_state::resident) {
        ctx.stats.layer_hits++;
        return;
    }
    if (L.state == layer_state::loading) {
        return;
    }
    if (prefetch && ctx.prefetch_budget_enabled) {
        const uint64_t charge = (uint64_t) ctx.slot_bytes;
        if (charge > ctx.prefetch_budget_available_bytes) {
            ctx.stats.prefetch_budget_dropped++;
            flex_trace_locked(ctx, "prefetch_budget_drop", layer_id, -1,
                    charge, ctx.prefetch_budget_available_bytes, 0);
            return;
        }
        ctx.prefetch_budget_available_bytes -= charge;
        ctx.stats.prefetch_budget_available_bytes =
            (size_t) ctx.prefetch_budget_available_bytes;
    }
    if (prefetch) {
        ctx.stats.prefetch_queued++;
    } else {
        ctx.stats.demand_loads++;
    }
    flex_trace_locked(ctx, prefetch ? "prefetch" : "demand", layer_id, -1, 0, 0, 0);
    ctx.queue.push_back(layer_id);
    ctx.cv_work.notify_one();
}

void llama_flex_request_layer(llama_flex_context & ctx, int layer_id) {
    flex_request_layer(ctx, layer_id, true);
}

static uint64_t flex_wait_layer_us(llama_flex_context & ctx, int layer_id) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    if (L.state == layer_state::resident) {
        return 0;
    }
    // Make sure it is at least queued.
    if (L.state == layer_state::not_resident) {
        ctx.stats.demand_loads++;
        flex_trace_locked(ctx, "demand_front", layer_id, -1, 0, 0, 0);
        ctx.queue.push_front(layer_id);
        ctx.cv_work.notify_one();
    }
    const uint64_t t0 = now_us();
    ctx.stats.wait_events++;
    ctx.cv_ready.wait(lock, [&] {
        return L.state == layer_state::resident || ctx.shutdown;
    });
    const uint64_t wait_us = now_us() - t0;
    ctx.stats.total_wait_us += wait_us;
    flex_trace_locked(ctx, "wait", layer_id, L.slot, 0, 0, wait_us);
    return wait_us;
}

void llama_flex_wait_layer(llama_flex_context & ctx, int layer_id) {
    (void) flex_wait_layer_us(ctx, layer_id);
}

void * llama_flex_get_tensor(llama_flex_context & ctx, int layer_id, const std::string & name) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    for (const auto & t : L.tensors) {
        if (t.name != name) {
            continue;
        }
        if (t.locked) {
            return (uint8_t *) ctx.lock_buf + t.buf_offset; // always resident
        }
        if (t.delta_locked &&
                t.delta_buf_index < ctx.delta_lock_bufs.size() &&
                ctx.delta_lock_bufs[t.delta_buf_index].ptr != nullptr) {
            return (uint8_t *) ctx.delta_lock_bufs[t.delta_buf_index].ptr + t.delta_buf_offset;
        }
        if (L.state != layer_state::resident || L.slot < 0) {
            return nullptr; // streamed tensor, layer not in a slot yet
        }
        return (uint8_t *) ctx.slots[L.slot] + t.buf_offset;
    }
    return nullptr;
}

void llama_flex_release_layer(llama_flex_context & ctx, int layer_id) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    L.released = true;
    L.last_use = now_us();
    ctx.stats.releases++;
    flex_trace_locked(ctx, "release", layer_id, L.slot, 0, 0, 0);
}

const llama_flex_stats & llama_flex_get_stats(const llama_flex_context & ctx) {
    return ctx.stats;
}
```

### runtime ring resize：llama_flex_resize_ring()

来源：`src/llama-flex.cpp:1388-1483`

```cpp
llama_flex_resize_result llama_flex_resize_ring(
        llama_flex_context & ctx,
        int                 target_slots) {
    llama_flex_resize_result result;
    if (!llama_flex_enabled(&ctx)) {
        result.reason = "disabled";
        return result;
    }

    std::lock_guard<std::mutex> lock(ctx.mutex);
    result.attempted = true;
    result.old_slots = (int) ctx.slots.size();
    result.new_slots = result.old_slots;
    result.old_bytes = (uint64_t) ctx.stats.ring_bytes;
    result.new_bytes = result.old_bytes;

    if (ctx.slot_bytes == 0) {
        result.reason = "zero_slot";
        return result;
    }

    target_slots = std::max(1, std::min(target_slots, ctx.n_layers));
    if (target_slots == result.old_slots) {
        result.reason = "unchanged";
        return result;
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (target_slots > result.old_slots) {
        const int add = target_slots - result.old_slots;
        std::vector<void *> new_slots;
        new_slots.reserve(add);
        for (int i = 0; i < add; ++i) {
            void * p = nullptr;
            if (posix_memalign(&p, page, ctx.slot_bytes) != 0 || p == nullptr) {
                for (void * q : new_slots) {
                    free(q);
                }
                result.reason = "alloc_failed";
                return result;
            }
            new_slots.push_back(p);
        }
        for (void * p : new_slots) {
            ctx.slots.push_back(p);
            ctx.slot_layer.push_back(-1);
        }
        ctx.params.ring_layers = target_slots;
        ctx.stats.ring_bytes = ctx.slot_bytes * (size_t) target_slots;
        flex_note_ahead_locked(ctx);
        result.changed = true;
        result.new_slots = target_slots;
        result.new_bytes = (uint64_t) ctx.stats.ring_bytes;
        result.reason = "grown";
        if (ctx.params.debug_log) {
            std::fprintf(stderr,
                    "llama_flex: resize ring %d -> %d (slot=%.2f MiB total=%.2f MiB)\n",
                    result.old_slots, result.new_slots,
                    ctx.slot_bytes / 1048576.0,
                    ctx.stats.ring_bytes / 1048576.0);
        }
        ctx.cv_work.notify_all();
        return result;
    }

    int removed = 0;
    for (int i = (int) ctx.slots.size() - 1; i >= 0 && (int) ctx.slots.size() > target_slots; --i) {
        if (ctx.slot_layer[i] >= 0) {
            continue;
        }
        free(ctx.slots[i]);
        ctx.slots.erase(ctx.slots.begin() + i);
        ctx.slot_layer.erase(ctx.slot_layer.begin() + i);
        for (auto & L : ctx.layers) {
            if (L.slot > i) {
                L.slot--;
            }
        }
        removed++;
    }
    ctx.params.ring_layers = (int) ctx.slots.size();
    ctx.stats.ring_bytes = ctx.slot_bytes * ctx.slots.size();
    flex_note_ahead_locked(ctx);
    result.changed = removed > 0;
    result.new_slots = (int) ctx.slots.size();
    result.new_bytes = (uint64_t) ctx.stats.ring_bytes;
    result.reason = result.new_slots == target_slots ? "shrunk" : "busy";
    if (ctx.params.debug_log && result.changed) {
        std::fprintf(stderr,
                "llama_flex: resize ring %d -> %d (slot=%.2f MiB total=%.2f MiB)\n",
                result.old_slots, result.new_slots,
                ctx.slot_bytes / 1048576.0,
                ctx.stats.ring_bytes / 1048576.0);
    }
    return result;
}
```

### async delta pin：llama_flex_delta_pin_async()

来源：`src/llama-flex.cpp:1627-1753`

```cpp
llama_flex_delta_pin_result llama_flex_delta_pin_async(
        llama_flex_context & ctx,
        uint64_t             budget_bytes,
        double               min_roi) {
    const uint64_t fn_t0 = now_us();
    llama_flex_delta_pin_result result;
    result.requested_bytes = budget_bytes;
    if (!llama_flex_enabled(&ctx)) {
        result.reason = "disabled";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }
    if (budget_bytes == 0) {
        result.reason = "zero_budget";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    std::vector<flex_delta_pin_candidate> candidates;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        result.attempted = true;
        ctx.stats.delta_pin_attempts++;
        if (!ctx.delta_pin_queue.empty()) {
            ctx.stats.delta_pin_async_rejected++;
            result.reason = "async_pending";
            result.elapsed_us = now_us() - fn_t0;
            return result;
        }
        for (int il = 0; il < ctx.n_layers; ++il) {
            const auto & L = ctx.layers[il];
            if (L.always_resident) {
                continue;
            }
            for (size_t it = 0; it < L.tensors.size(); ++it) {
                const auto & t = L.tensors[it];
                if (t.locked || t.delta_locked || t.size == 0 || t.size > budget_bytes) {
                    continue;
                }
                const size_t value = flex_pin_value_bytes(ctx, t);
                const double roi = (double) value / (double) t.size;
                if (roi < min_roi) {
                    continue;
                }
                candidates.push_back({ il, it, t, value, roi, 0 });
            }
        }
    }
    result.candidates = candidates.size();
    if (candidates.empty()) {
        result.reason = "no_candidate";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    std::sort(candidates.begin(), candidates.end(),
            [](const flex_delta_pin_candidate & a, const flex_delta_pin_candidate & b) {
                if (a.roi != b.roi) {
                    return a.roi > b.roi;
                }
                if (a.value != b.value) {
                    return a.value > b.value;
                }
                if (a.tensor.size != b.tensor.size) {
                    return a.tensor.size < b.tensor.size;
                }
                if (a.layer != b.layer) {
                    return a.layer < b.layer;
                }
                return a.tensor.name < b.tensor.name;
            });

    flex_delta_pin_job job;
    job.requested_bytes = budget_bytes;
    job.submitted_us = now_us();
    for (auto & c : candidates) {
        if (job.used + c.tensor.size > budget_bytes) {
            continue;
        }
        c.dst_offset = job.used;
        job.used += c.tensor.size;
        job.value_sum += (double) c.value;
        job.selected.push_back(c);
    }
    if (job.selected.empty() || job.used == 0) {
        result.reason = "below_budget";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    job.alloc_size = ((job.used + page - 1) / page) * page;
    if (posix_memalign(&job.delta, page, job.alloc_size) != 0 || job.delta == nullptr) {
        result.reason = "alloc_failed";
        result.elapsed_us = now_us() - fn_t0;
        std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.stats.delta_pin_failures++;
        return result;
    }

    const uint64_t enqueued_bytes = job.used;
    const uint64_t enqueued_tensors = job.selected.size();
    const double enqueued_value_sum = job.value_sum;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if (!ctx.delta_pin_queue.empty()) {
            free(job.delta);
            ctx.stats.delta_pin_async_rejected++;
            result.reason = "async_pending";
            result.elapsed_us = now_us() - fn_t0;
            return result;
        }
        ctx.delta_pin_queue.push_back(std::move(job));
        ctx.stats.delta_pin_async_submitted++;
        ctx.stats.delta_pin_async_pending = ctx.delta_pin_queue.size();
    }
    ctx.cv_delta_pin.notify_one();

    result.changed = false;
    result.pinned_bytes = enqueued_bytes;
    result.saved_per_token_bytes = enqueued_bytes;
    result.pinned_tensors = enqueued_tensors;
    result.roi = enqueued_bytes > 0 ? enqueued_value_sum / (double) enqueued_bytes : 0.0;
    result.reason = "async_enqueued";
    result.elapsed_us = now_us() - fn_t0;
    return result;
}
```

### sync delta pin：llama_flex_delta_pin()

来源：`src/llama-flex.cpp:1627-1753`

```cpp
llama_flex_delta_pin_result llama_flex_delta_pin_async(
        llama_flex_context & ctx,
        uint64_t             budget_bytes,
        double               min_roi) {
    const uint64_t fn_t0 = now_us();
    llama_flex_delta_pin_result result;
    result.requested_bytes = budget_bytes;
    if (!llama_flex_enabled(&ctx)) {
        result.reason = "disabled";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }
    if (budget_bytes == 0) {
        result.reason = "zero_budget";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    std::vector<flex_delta_pin_candidate> candidates;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        result.attempted = true;
        ctx.stats.delta_pin_attempts++;
        if (!ctx.delta_pin_queue.empty()) {
            ctx.stats.delta_pin_async_rejected++;
            result.reason = "async_pending";
            result.elapsed_us = now_us() - fn_t0;
            return result;
        }
        for (int il = 0; il < ctx.n_layers; ++il) {
            const auto & L = ctx.layers[il];
            if (L.always_resident) {
                continue;
            }
            for (size_t it = 0; it < L.tensors.size(); ++it) {
                const auto & t = L.tensors[it];
                if (t.locked || t.delta_locked || t.size == 0 || t.size > budget_bytes) {
                    continue;
                }
                const size_t value = flex_pin_value_bytes(ctx, t);
                const double roi = (double) value / (double) t.size;
                if (roi < min_roi) {
                    continue;
                }
                candidates.push_back({ il, it, t, value, roi, 0 });
            }
        }
    }
    result.candidates = candidates.size();
    if (candidates.empty()) {
        result.reason = "no_candidate";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    std::sort(candidates.begin(), candidates.end(),
            [](const flex_delta_pin_candidate & a, const flex_delta_pin_candidate & b) {
                if (a.roi != b.roi) {
                    return a.roi > b.roi;
                }
                if (a.value != b.value) {
                    return a.value > b.value;
                }
                if (a.tensor.size != b.tensor.size) {
                    return a.tensor.size < b.tensor.size;
                }
                if (a.layer != b.layer) {
                    return a.layer < b.layer;
                }
                return a.tensor.name < b.tensor.name;
            });

    flex_delta_pin_job job;
    job.requested_bytes = budget_bytes;
    job.submitted_us = now_us();
    for (auto & c : candidates) {
        if (job.used + c.tensor.size > budget_bytes) {
            continue;
        }
        c.dst_offset = job.used;
        job.used += c.tensor.size;
        job.value_sum += (double) c.value;
        job.selected.push_back(c);
    }
    if (job.selected.empty() || job.used == 0) {
        result.reason = "below_budget";
        result.elapsed_us = now_us() - fn_t0;
        return result;
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    job.alloc_size = ((job.used + page - 1) / page) * page;
    if (posix_memalign(&job.delta, page, job.alloc_size) != 0 || job.delta == nullptr) {
        result.reason = "alloc_failed";
        result.elapsed_us = now_us() - fn_t0;
        std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.stats.delta_pin_failures++;
        return result;
    }

    const uint64_t enqueued_bytes = job.used;
    const uint64_t enqueued_tensors = job.selected.size();
    const double enqueued_value_sum = job.value_sum;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if (!ctx.delta_pin_queue.empty()) {
            free(job.delta);
            ctx.stats.delta_pin_async_rejected++;
            result.reason = "async_pending";
            result.elapsed_us = now_us() - fn_t0;
            return result;
        }
        ctx.delta_pin_queue.push_back(std::move(job));
        ctx.stats.delta_pin_async_submitted++;
        ctx.stats.delta_pin_async_pending = ctx.delta_pin_queue.size();
    }
    ctx.cv_delta_pin.notify_one();

    result.changed = false;
    result.pinned_bytes = enqueued_bytes;
    result.saved_per_token_bytes = enqueued_bytes;
    result.pinned_tensors = enqueued_tensors;
    result.roi = enqueued_bytes > 0 ? enqueued_value_sum / (double) enqueued_bytes : 0.0;
    result.reason = "async_enqueued";
    result.elapsed_us = now_us() - fn_t0;
    return result;
}
```

### released slot clean reclaim：llama_flex_reclaim_released()

来源：`src/llama-flex.cpp:1963-2010`

```cpp
llama_flex_reclaim_result llama_flex_reclaim_released(
        llama_flex_context & ctx,
        uint64_t             target_bytes,
        uint32_t             max_layers) {
    llama_flex_reclaim_result result;
    if (!llama_flex_enabled(&ctx) || target_bytes == 0 || max_layers == 0) {
        return result;
    }

    std::lock_guard<std::mutex> lock(ctx.mutex);
    while (result.released_bytes < target_bytes && result.released_layers < max_layers) {
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int l = 0; l < ctx.n_layers; ++l) {
            const auto & L = ctx.layers[l];
            if (l == ctx.cur_compute_layer || L.always_resident ||
                    L.slot < 0 || !L.released || L.state != layer_state::resident) {
                continue;
            }
            if (L.last_use < oldest) {
                oldest = L.last_use;
                victim = l;
            }
        }
        if (victim < 0) {
            break;
        }

        auto & L = ctx.layers[victim];
        const int slot = L.slot;
        if (slot >= 0 && slot < (int) ctx.slots.size() && ctx.slots[slot] != nullptr) {
#if defined(MADV_DONTNEED)
            madvise(ctx.slots[slot], ctx.slot_bytes, MADV_DONTNEED);
#endif
            ctx.slot_layer[slot] = -1;
        }
        L.slot = -1;
        L.state = layer_state::not_resident;
        L.released = true;
        L.last_use = now_us();
        ctx.stats.evictions++;
        result.released_bytes += (uint64_t) ctx.slot_bytes;
        result.released_layers++;
        flex_trace_locked(ctx, "governor_reclaim", victim, slot, ctx.slot_bytes, 0, 0);
    }
    result.target_satisfied = result.released_bytes >= target_bytes;
    return result;
}
```

### graph begin：llama_flex_graph_begin()

来源：`src/llama-flex.cpp:2012-2032`

```cpp
void llama_flex_graph_begin(llama_flex_context & ctx) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    if (ctx.cur_compute_layer >= 0) {
        llama_flex_release_layer(ctx, ctx.cur_compute_layer);
    }
    ctx.cur_compute_layer = -1;
    ctx.last_layer_enter_us = 0;
    ctx.graph_id++;
    ctx.stats.graphs++;
    const int ahead = flex_effective_ahead(ctx);
    ctx.stats.effective_ahead = ahead;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        flex_trace_locked(ctx, "graph_begin", -1, -1, 0, 0, 0);
    }
    for (int l = 0; l <= ahead; ++l) {
        flex_request_layer(ctx, l % ctx.n_layers, true);
    }
}
```

### compute callback：llama_flex_stream_callback()

来源：`src/llama-flex.cpp:2034-2082`

```cpp
bool llama_flex_stream_callback(struct ggml_tensor * op, int ith, void * user_data) {
    auto * ctx = static_cast<llama_flex_context *>(user_data);
    if (ctx == nullptr || op == nullptr) {
        return false;
    }

    int op_layer = -1;
    uint64_t wait_us_total = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor * s = op->src[i];
        if (s == nullptr || s->name[0] == '\0') {
            continue;
        }
        auto it = ctx->name_layer.find(s->name);
        if (it == ctx->name_layer.end()) {
            continue;
        }
        op_layer = it->second;
        if (ith == 0) {
            const uint64_t wait_us = flex_wait_layer_us(*ctx, op_layer);
            wait_us_total += wait_us;
            void * p = llama_flex_get_tensor(*ctx, op_layer, s->name);
            if (p != nullptr) {
                s->data = p;
            }
        }
    }

    if (op_layer < 0) {
        return false;
    }

    // Drive prefetch/release when compute advances to a new layer (ith==0 only).
    if (ith == 0 && op_layer != ctx->cur_compute_layer) {
        const int prev = ctx->cur_compute_layer;
        flex_adapt_after_layer(*ctx, wait_us_total);
        ctx->cur_compute_layer = op_layer;
        const int ahead = flex_effective_ahead(*ctx);
        ctx->stats.effective_ahead = ahead;
        for (int a = 1; a <= ahead; ++a) {
            flex_request_layer(*ctx, (op_layer + a) % ctx->n_layers, true);
        }
        if (prev >= 0) {
            llama_flex_release_layer(*ctx, prev);
        }
    }

    return true;
}
```

## 16.2 MoE Buffer 完整关键代码

### MoE Buffer 对外参数/统计/API 声明

来源：`src/llama-moe-buffer.h:1-216`

```cpp
#pragma once

// MoE expert streaming via explicit anonymous buffers (per-expert-slice repoint).
//
// Unlike the mmap+madvise expert window (whose pages the kernel keeps re-caching,
// so the real physical footprint stays pinned at the memory cap), this path never
// keeps the expert weights mmap-backed. For every `ffn_*_exps` weight tensor it
// allocates a full-size ANONYMOUS buffer and repoints tensor->data to it. Only the
// expert slices actually selected by the router are pread() into their id-indexed
// slot; cold experts are never read, so their slots stay zero-fill (unbacked) and
// occupy no physical memory. Resident slots are bounded by an LRU byte budget and
// evicted with madvise(MADV_DONTNEED) (anonymous => pages are truly freed).
//
// Because mul_mat_id reads `tensor->data + expert_id * stride`, the full-size
// buffer means no id remapping and no kernel change: the GEMM just reads the slot
// we streamed. Residency is guaranteed before the kernel runs by the CPU
// weight-stream callback (ith==0 streams the op's selected experts, then a barrier
// publishes them to all worker threads).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;
struct llama_moe_buffer_context;

struct llama_moe_buffer_params {
    bool   enabled      = false;
    bool   debug_log    = false;
    bool   direct_io    = false;     // O_DIRECT streaming reads (needs alignment)
    bool   hebf_schedule = true;     // priority scheduling for async expert prefetch
    bool   dynamic_bits  = true;     // policy-only bit-width selection (no data-format change)
    bool   dynamic_bits_real = true; // require exact low-bit/MWQ data; fallback to full reads if unavailable
    bool   strict_sidecar = false; // abort if the selected sidecar bit width is unavailable
    bool   native_hot = false;       // allow target bits >= base_bits to use original GGUF quant slices in the op override
    bool   fuse_gate_up = true;      // defer ffn_gate/up_exps MUL_MAT_ID until the sibling op is seen, then compute both together
    bool   fuse_swiglu = true;       // override SWIGLU when its inputs are the just-fused gate/up expert outputs
    bool   fuse_direct_swiglu = true; // compute dot(gate), dot(up), and silu*up inside the SWIGLU override without writing gate/up tensors
    bool   fuse_expert_ffn = false;  // skip full SWIGLU tensor and fuse gate/up SWIGLU directly into down projection
    bool   prefetch_down_with_swiglu = false; // when direct-SWIGLU ensures gate/up, also warm the same expert's down slice
    bool   avx512_q2 = true;         // use the AVX512 q2 hierarchical direct-SWIGLU hot path when available
    bool   avx512_q2_dot = true;     // use the AVX512 q2 hierarchical MUL_MAT_ID/down-projection dot path when available
    bool   vnni_q2 = false;          // quantize F32 activations to int8 and use AVX512-VNNI for q2 hierarchical expert dots
    bool   vnni_q2_down = true;      // allow the VNNI q2 path for MUL_MAT_ID/down-projection dots
    bool   vnni_q2_swiglu = true;    // allow the VNNI q2 path for fused gate/up SWIGLU
    int    vnni_block = 64;          // activation int8 quantization block; q2 VNNI currently uses 64-column chunks
    int    avx512_prefetch = 0;      // q2-hier kernel prefetch distance in MWQ blocks; 0 disables explicit prefetch
    size_t budget_bytes = 1024ull * 1024ull * 1024ull; // resident-expert byte budget when bounded
    bool   budget_unbounded = false; // explicit unbounded mode; separates unlimited from a zero-byte bounded budget
    size_t planner_safe_budget_bytes = 0; // load-time safe budget derived from memory.max
    size_t planner_floor_bytes = 0;       // non-reclaimable load-time floor
    int    n_workers    = 2;         // parallel prefetch workers (raise to lift effective
                                     // read bandwidth on NVMe: single-thread O_DIRECT
                                     // random reads under-utilise the device)
    int    base_bits    = 2;         // current on-disk expert precision for dyn-bit accounting
    int    hot_bits     = 2;         // target bits for rank-0/hot predicted experts
    int    warm_bits    = 2;         // target bits for mid-rank predicted experts
    int    cold_bits    = 2;         // target bits for low-rank predicted experts
    int    gate_bits    = 2;         // optional exact target bits for ffn_gate_exps; 0 = use hot/warm/cold policy
    int    up_bits      = 2;         // optional exact target bits for ffn_up_exps; 0 = use hot/warm/cold policy
    int    down_bits    = 2;         // optional exact target bits for ffn_down_exps; 0 = use hot/warm/cold policy
    int    fixed_bits   = 0;         // force every expert tensor/rank to this sidecar bit width; 0 = dynamic policy
    int    gate_min_bits = 2;        // default sensitivity floor: gate logits should not use q2 unless explicitly requested
    int    up_min_bits   = 0;        // up is the least sensitive expert projection; 0 = no floor
    int    down_min_bits = 2;        // down projection feeds the residual path, keep at least q3 by default
    int    sync_top_k   = 4;         // routed expert ids are assumed grouped by top-k rank
    float  hot_ratio    = 0.0f;      // relative hotness: an expert is pinned (never
                                     // LRU-evicted) when its activation count exceeds
                                     // hot_ratio * (tensor mean activation). 0 = pure
                                     // LRU. Bounded by construction: only experts
                                     // routed more than hot_ratio× the per-tensor
                                     // average stay pinned, so the pin set cannot grow
                                     // to "all experts" as the sequence lengthens.
    float  pinned_fraction = 0.35f;  // fraction of resident budget reserved for dynamic
                                     // (layer, expert) hot groups. Pinned groups are
                                     // excluded from eviction until demoted by the
                                     // periodic top-score refresh.
    float  pinned_layer_fraction = 0.18f; // max fraction of pinned budget one layer may use
    int    active_window = 4;        // future-use distance protected by Belady-style eviction
    int    group_cooldown_tokens = 0; // protect recently used (layer, expert) groups for N decode-token epochs
    bool   pressure_adaptive = true; // shrink soft expert-cache protection as resident/budget pressure rises
    bool   pressure_scale_pin = true; // let pressure shrink the global pinned fraction
    bool   pressure_scale_layer_pin = true; // let pressure shrink the per-layer pinned cap
    bool   pressure_scale_window = true; // let pressure shrink future-use active protection
    bool   pressure_scale_cooldown = true; // let pressure shrink recent-use cooldown protection
    bool   pressure_scale_spec_guard = true; // let pressure shrink speculative-unused keep penalty
    double pressure_soft_ratio = 0.88; // resident/budget ratio where pressure adaptation starts
    double pressure_hard_ratio = 0.97; // resident/budget ratio where soft protection reaches its floor
    float  pressure_pin_floor = 0.0f;  // minimum effective pinned fraction under hard pressure
    int    pin_refresh_interval = 128; // group touches between top-score pin refreshes
    double warm_coverage = 0.95;    // target per-layer expert probability mass for warm-start sizing
    std::string sidecar_path;        // optional exact low-bit/MWQ sidecar data source
};

struct llama_moe_buffer_stats {
    size_t   resident_bytes = 0;
    size_t   budget_bytes = 0;
    bool     budget_unbounded = false;
    size_t   planner_safe_budget_bytes = 0;
    size_t   planner_floor_bytes = 0;
    size_t   expert_bytes = 0;
    uint64_t streams = 0;
    uint64_t hits = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_late = 0;
    uint64_t prefetch_unused = 0;
    uint64_t prefetch_budget_dropped = 0;
    size_t   prefetch_budget_bytes = 0;
    size_t   prefetch_budget_available_bytes = 0;
    size_t   warm_working_set_bytes = 0;
    uint64_t warm_working_set_groups = 0;
    double   warm_working_set_coverage = 0.0;
};

struct llama_moe_buffer_reclaim_result {
    uint64_t released_bytes = 0;
    uint32_t released_groups = 0;
    bool target_satisfied = false;
};

std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params);

bool llama_moe_buffer_enabled(const llama_moe_buffer_context * ctx);

// Total bytes of all registered expert tensors. Used by the adaptive-budget
// decision (the streamable working set upper bound).
size_t llama_moe_buffer_expert_bytes(const llama_moe_buffer_context * ctx);

// Override the resident-expert byte budget after registration (for adaptive
// budgeting computed once the expert total and available memory are known).
void llama_moe_buffer_set_budget(llama_moe_buffer_context * ctx, size_t budget_bytes);

// Install the load-time memory.max plan. This makes the bounded/unbounded
// state explicit so a bounded zero-byte budget is not confused with unlimited.
void llama_moe_buffer_set_budget_plan(
        llama_moe_buffer_context * ctx,
        size_t                    budget_bytes,
        bool                      budget_unbounded,
        size_t                    planner_safe_budget_bytes,
        size_t                    planner_floor_bytes);

// Estimate the warm-start resident working set from the registered
// (layer, expert) groups. Uses observed per-request/historical group scores when
// present, and the structural top-k/uniform prior before any routing history
// exists.
size_t llama_moe_buffer_warm_working_set_bytes(
        llama_moe_buffer_context * ctx,
        double                    coverage);

// Register one `*_exps` weight tensor: allocate a full-size anonymous buffer,
// repoint exps->data to it, and record per-expert file metadata. `fd` is the
// model file descriptor (duplicated internally). Returns true on success; on
// failure the tensor is left untouched (still mmap-backed).
bool llama_moe_buffer_register(
        llama_moe_buffer_context & ctx,
        ggml_tensor *              exps,
        int                        fd,
        size_t                     file_offset,
        size_t                     expert_stride,
        int                        n_expert);

// ggml_cpu_weight_stream_callback. user_data must be a llama_moe_buffer_context*.
// For a managed GGML_OP_MUL_MAT_ID op, ith==0 ensures every expert in op->src[2]
// is resident in the buffer (pread on miss, LRU-evict to stay within budget).
// Returns true iff the op is managed (so the CPU backend issues a barrier).
bool llama_moe_buffer_stream_callback(ggml_tensor * op, int ith, void * user_data);

// CPU override for managed GGML_OP_MUL_MAT_ID ops whose selected experts are
// resident as MWQ sidecar slices. Returns true when it completed the op.
bool llama_moe_buffer_mul_mat_id_callback(ggml_tensor * op, int ith, int nth, void * user_data);

// Asynchronous prefetch hint, issued by the CLG predictor while layer L computes
// to give layer L+1's experts lead time. For every `*_exps` tensor of `layer`,
// the listed experts are enqueued to a background worker that streams their
// slices into the anonymous buffer (no-op for already-resident/in-flight slots).
// When the layer's mul_mat_id later runs, the weight-stream callback finds the
// slices already resident (hit) instead of taking a synchronous read. Safe to
// call concurrently with the weight-stream callback. No-op if ctx is null/disabled.
void llama_moe_buffer_prefetch(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        int                        n_experts);

// Ranked prefetch hint. `scores` may be null; when present, higher scores are
// scheduled first and used by the dynamic-bit policy for accounting. `experts`
// should be ordered by predicted utility (rank 0 = hottest/most likely).
void llama_moe_buffer_prefetch_ranked(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        const float *              scores,
        int                        n_experts);

llama_moe_buffer_stats llama_moe_buffer_get_stats(llama_moe_buffer_context & ctx);

// Reclaim cold resident expert groups using the buffer's existing victim
// selection. This releases clean anonymous expert pages; the model file/sidecar
// remains the authoritative source for future reloads.
llama_moe_buffer_reclaim_result llama_moe_buffer_reclaim_clean(
        llama_moe_buffer_context & ctx,
        uint64_t                   target_bytes,
        uint32_t                   max_groups);

// Set a per-tick speculative prefetch budget. A zero budget disables the gate.
// Demand loads are never gated by this API.
void llama_moe_buffer_set_prefetch_budget(
        llama_moe_buffer_context & ctx,
        uint64_t                  budget_bytes);

void llama_moe_buffer_print_stats(const llama_moe_buffer_context & ctx);
```

### MoE group 状态：moe_group_state

来源：`src/llama-moe-buffer.cpp:305-409`

```cpp
struct moe_group_state {
    int layer = -1;
    int expert = -1;
    bool group_lru_linked = false;
    std::list<uint64_t>::iterator group_lru_pos;
    size_t resident_bytes_cached = 0;
    uint64_t resident_bit_weight_cached = 0;
    uint8_t resident_slices_cached = 0;
    uint64_t evict_scan_generation = 0;
    uint64_t access = 0;
    uint64_t rank0_access = 0;
    uint64_t last_used_epoch = 0;
    uint64_t last_used_token_epoch = 0;
    uint64_t next_use_epoch = UINT64_MAX;
    uint64_t pin_epoch = 0;
    double hot_score = 0.0;
    bool pinned = false;

    // Sequence-level Expert Activation Matrix (EAM). The first step used these
    // fields only for observability; the activation-aware cache policy now also
    // uses them to score pinned groups and eviction victims.
    uint64_t seq_access = 0;
    uint64_t seq_rank0_access = 0;
    uint64_t seq_first_token_epoch = 0;
    uint64_t seq_last_token_epoch = 0;
    uint64_t seq_future_hints = 0;
    uint64_t seq_future_rank0_hints = 0;
    uint64_t seq_prefetch_queued = 0;
    uint64_t seq_prefetch_hits = 0;
    uint64_t seq_prefetch_late = 0;
    uint64_t seq_prefetch_unused = 0;
    uint64_t seq_cache_hits = 0;
    uint64_t seq_cache_misses = 0;
    uint64_t seq_predicted = 0;
    uint64_t seq_pred_enqueued = 0;
    double   seq_future_score_sum = 0.0;
    double   seq_predict_score_sum = 0.0;
    std::array<uint64_t, 4> seq_tensor_access = {0, 0, 0, 0}; // other, gate, up, down

    // Online short-term reuse estimator. This is learned from the current
    // request as it runs; trace files are only used offline to validate it.
    double   reuse_ema = 0.0;
    double   inter_token_gap_ema = 16.0;
    uint64_t reuse_observed = 0;
    uint64_t reuse_within_1 = 0;
    uint64_t reuse_within_4 = 0;
    uint64_t reuse_within_16 = 0;
    uint64_t last_evicted_token_epoch = 0;
    bool     last_evicted_valid = false;
    uint8_t  last_evicted_reason = MOE_EVICT_REASON_UNKNOWN;
    uint64_t layer_window_bad_reload = 0;
    uint64_t layer_window_last_bad_reload_token = 0;
    double   bad_reload_score = 0.0;
    uint64_t bad_reload_last_token = 0;
    uint64_t bad_reload_1 = 0;
    uint64_t bad_reload_4 = 0;
    uint64_t bad_reload_16 = 0;
    uint64_t bad_reload_protect_hits = 0;
    uint64_t bad_reload_soft_hits = 0;
    double   ghost_reload_risk_ema = 0.0;
    uint64_t ghost_generation = 0;
    uint64_t ghost_outcomes = 0;
    uint64_t ghost_bad_1 = 0;
    uint64_t ghost_bad_4 = 0;
    uint64_t ghost_bad_16 = 0;
    uint64_t ghost_success = 0;
    uint64_t ghost_last_outcome_token = 0;
    bool     ghost_pending = false;
    uint64_t demand_async_generation = 0;
    bool     demand_async_pending = false;
    uint64_t demand_admission_token = UINT64_MAX;
    bool     demand_admission_decided = false;
    bool     demand_admission_bypass = false;
    uint64_t admission_outcome_generation = 0;
    uint64_t admission_outcome_token = 0;
    uint64_t admission_outcome_deadline = 0;
    uint8_t  admission_outcome_action = 0; // 1=admit, 2=bypass
    bool     admission_outcome_pending = false;
    size_t   admission_staging_reserved_bytes = 0;
    double   admission_outcome_candidate = 0.0;
    double   admission_outcome_victim = 0.0;
    int      admission_outcome_victim_layer = -1;
    int      admission_outcome_victim_expert = -1;
    int      admission_outcome_victim_groups = 0;
    size_t   admission_outcome_victim_bytes = 0;
    double   admission_feedback_ema = 0.0;
    uint64_t admission_feedback_samples = 0;
    double   admission_regret_ema = 0.0;
    uint64_t admission_regret_samples = 0;
    double   eam_replace_pred = 0.0;
    double   eam_replace_mass = 0.0;
    double   eam_replace_keep = 0.0;
    uint8_t  cct_conf = 0;
    uint64_t cct_protect_hits = 0;

    // Same-layer next-token resident predictor. This is intentionally a
    // retention-only signal: it protects resident groups from layer-done
    // eviction, but it does not enqueue speculative reads.
    uint8_t  next_token_conf = 0;
    uint64_t next_token_protect_until = 0;
    uint64_t next_token_predicted = 0;
    uint64_t next_token_hit = 0;
    uint64_t next_token_miss = 0;
    uint64_t next_token_protect_hits = 0;
};
```

### MoE Buffer 内部状态：llama_moe_buffer_context

来源：`src/llama-moe-buffer.cpp:539-623`

```cpp
struct llama_moe_buffer_context;
static void moe_print_stats_impl(const llama_moe_buffer_context & ctx, const char * prefix);
static void moe_eamc_snapshot_current_locked(llama_moe_buffer_context & ctx, bool force);
static void moe_eam_trace_append_locked(const llama_moe_buffer_context & ctx);
static void moe_eamc_load_sidecar(llama_moe_buffer_context & ctx);
static void moe_admission_outcome_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_admission_regret_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_olecar_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_cct_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        src_layer,
        int                        src_expert,
        int                        target_layer,
        int                        target_expert,
        int                        conf_before,
        int                        conf_after,
        const char *               reason);
static bool moe_next_token_group_protected_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g);

struct llama_moe_buffer_context {
    llama_moe_buffer_params params;

    std::unordered_map<std::string, moe_managed>      by_name;
    std::unordered_map<int, std::vector<moe_managed*>> by_layer;
    std::unordered_map<uint64_t, moe_group_state> groups;
    std::vector<int> fds;
    int sidecar_fd = -1;
    uint32_t sidecar_bits_mask = 0;
    int cache_fd = -1;
    uint64_t cache_next = 0;
    std::unordered_map<std::string, moe_sidecar_entry> sidecar;
    std::unordered_map<std::string, char> cache_state; // 0 missing, 1 building, 2 ready
    std::mutex cache_mtx;
    std::condition_variable cache_cv;
    std::vector<mwq_transient_slice> mwq_transients;
    ggml_tensor * pending_up_op = nullptr;
    ggml_tensor * pending_up_src1 = nullptr;
    ggml_tensor * pending_up_ids = nullptr;
    int pending_up_layer = -1;
    ggml_tensor * pending_gate_op = nullptr;
    ggml_tensor * pending_gate_src1 = nullptr;
    ggml_tensor * pending_gate_ids = nullptr;
    int pending_gate_layer = -1;
    ggml_tensor * fused_up_op = nullptr;
    ggml_tensor * fused_gate_op = nullptr;
    int fused_layer = -1;
    ggml_tensor * last_fused_up_op = nullptr;
    ggml_tensor * last_fused_gate_op = nullptr;

    // Residency state (resident[], lru, resident_bytes, buf allocation).
    std::mutex              mtx;
    std::condition_variable cv_done;   // signalled when an in-flight slice completes
    std::condition_variable cv_fuse;   // signalled when a deferred gate/up pair is matched
    std::list<std::pair<moe_managed *, int>> lru;  // front = MRU
    std::list<uint64_t> group_lru;  // front = MRU, one node per resident ExpertGroup
    size_t resident_bytes = 0;
    size_t demand_admission_staging_bytes = 0; // guarded by mtx; included in resident_bytes
    size_t demand_admission_staging_reserved_bytes = 0; // guarded by mtx
    size_t expert_total   = 0;   // sum of all registered exps tensor bytes
    size_t warm_working_set_bytes = 0;
    uint64_t warm_working_set_groups = 0;
    double warm_working_set_coverage = 0.0;
    size_t align          = 4096;

    // Single-flight coordination for llama_moe_buffer_swiglu_direct_compute's
    // residency-ensure phase: that function is invoked as a ggml op_override
    // callback (no barrier separates the nth worker threads, unlike the
    // weight-stream+barrier path which only fires for GGML_OP_MUL_MAT_ID), so
    // without this every thread would redundantly redo the ensure-residency
    // loop. Keyed by the GLU op tensor pointer, which is stable across the nth
    // per-thread invocations of one "wave" but distinct across graph nodes.
    struct moe_direct_ensure_entry {
        bool ok        = false;   // residency-ensure result, valid once done == true
        bool done      = false;
        int  remaining = 0;       // threads of this wave still to consume the result
    };
```

### pressure-adaptive effective 参数：moe_pressure_* / moe_effective_*

来源：`src/llama-moe-buffer.cpp:1019-1071`

```cpp
static double moe_pressure_ratio_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.params.pressure_adaptive || moe_budget_unbounded(ctx) || ctx.params.budget_bytes == 0) {
        return 0.0;
    }
    return (double) ctx.resident_bytes / (double) ctx.params.budget_bytes;
}

static double moe_pressure_scale_locked(const llama_moe_buffer_context & ctx) {
    const double ratio = moe_pressure_ratio_locked(ctx);
    const double soft = std::max(0.0, ctx.params.pressure_soft_ratio);
    const double hard = std::max(soft + 1.0e-6, ctx.params.pressure_hard_ratio);
    if (ratio <= soft) {
        return 0.0;
    }
    if (ratio >= hard) {
        return 1.0;
    }
    return (ratio - soft) / (hard - soft);
}

static float moe_effective_pinned_fraction_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.params.pressure_adaptive || !ctx.params.pressure_scale_pin) {
        return ctx.params.pinned_fraction;
    }
    const double scale = moe_pressure_scale_locked(ctx);
    const float floor = std::max(0.0f, std::min(ctx.params.pinned_fraction, ctx.params.pressure_pin_floor));
    const float effective = (float) ((double) ctx.params.pinned_fraction * (1.0 - scale));
    return std::max(floor, effective);
}

static float moe_effective_pinned_layer_fraction_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.params.pressure_adaptive || !ctx.params.pressure_scale_layer_pin) {
        return ctx.params.pinned_layer_fraction;
    }
    const double scale = moe_pressure_scale_locked(ctx);
    return std::max(0.0f, (float) ((double) ctx.params.pinned_layer_fraction * (1.0 - scale)));
}

static int moe_effective_active_window_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.params.pressure_adaptive || !ctx.params.pressure_scale_window) {
        return ctx.params.active_window;
    }
    const double scale = moe_pressure_scale_locked(ctx);
    return std::max(0, (int) std::floor((double) ctx.params.active_window * (1.0 - scale)));
}

static int moe_effective_group_cooldown_tokens_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.params.pressure_adaptive || !ctx.params.pressure_scale_cooldown) {
        return ctx.params.group_cooldown_tokens;
    }
    const double scale = moe_pressure_scale_locked(ctx);
    return std::max(0, (int) std::floor((double) ctx.params.group_cooldown_tokens * (1.0 - scale)));
}
```

### 后台 worker loop：moe_worker_loop()

来源：`src/llama-moe-buffer.cpp:1520-1577`

```cpp
static void moe_worker_loop(llama_moe_buffer_context * ctx) {
    for (;;) {
        moe_prefetch_task task;
        {
            std::unique_lock<std::mutex> lk(ctx->qmtx);
            ctx->cv_q.wait(lk, [ctx] { return ctx->stop || !ctx->queue.empty(); });
            if (ctx->stop && ctx->queue.empty()) {
                return;
            }
            task = ctx->queue.top();
            ctx->queue.pop();
            if (!task.is_group && task.m != nullptr && task.e >= 0 && task.e < (int) task.m->queued.size()) {
                if (task.e < (int) task.m->queued_bits.size() && task.m->queued_bits[task.e] > task.target_bits) {
                    ctx->queue_dups.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                task.m->queued[task.e] = false;
                if (task.e < (int) task.m->queued_bits.size()) {
                    task.m->queued_bits[task.e] = 0;
                }
            }
        }
        if (task.is_group) {
            moe_prefetch_group_worker(*ctx, task.layer, task.e, task.rank, task.score, task.skip_kind,
                    task.is_demand, task.primary_kind);
            if (task.is_demand) {
                std::lock_guard<std::mutex> lk(ctx->mtx);
                auto git = ctx->groups.find(moe_group_key(task.layer, task.e));
                if (git != ctx->groups.end() &&
                        git->second.demand_async_generation == task.generation) {
                    ctx->demand_admission_staging_reserved_bytes -= std::min(
                            ctx->demand_admission_staging_reserved_bytes,
                            git->second.admission_staging_reserved_bytes);
                    git->second.admission_staging_reserved_bytes = 0;
                    git->second.demand_async_pending = false;
                }
                ctx->cv_done.notify_all();
            }
            ctx->worker_streams.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (ctx->params.dynamic_bits && task.m != nullptr) {
            const uint64_t full = task.m->stride;
            const uint64_t eff = moe_effective_stream_bytes_for_target(
                    *ctx, *task.m, task.e, task.target_bits);
            ctx->dyn_effective_bytes.fetch_add(eff, std::memory_order_relaxed);
            ctx->dyn_saved_bytes.fetch_add(full > eff ? full - eff : 0, std::memory_order_relaxed);
        }
        if (task.m != nullptr) {
            if (ctx->params.dynamic_bits_real && moe_mwq_entry(*ctx, *task.m, task.e, task.target_bits) != nullptr) {
                moe_stream_mwq_slice(*ctx, *task.m, task.e, false, task.target_bits, task.rank);
            } else {
                moe_stream_slice(*ctx, *task.m, task.e, false, task.target_bits, task.rank);
            }
        }
        ctx->worker_streams.fetch_add(1, std::memory_order_relaxed);
    }
}
```

### 创建上下文：llama_moe_buffer_create()

来源：`src/llama-moe-buffer.cpp:1579-1762`

```cpp
std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params) {
    auto ctx = std::make_shared<llama_moe_buffer_context>();
    ctx->params = params;
    ctx->profile = std::getenv("LLAMA_MOE_PROFILE") != nullptr &&
            std::atoi(std::getenv("LLAMA_MOE_PROFILE")) > 0;
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_CACHE_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->cache_trace = std::fopen(trace_path, "wb");
            if (ctx->cache_trace != nullptr) {
                std::fprintf(ctx->cache_trace,
                        "token\texec\tlayer\texpert\tkind\trank\ttarget_bits\tarrival_state\tarrival_bits\tarrival_mwq\tarrival_queued\tarrival_queued_bits\tarrival_prefetched\taction\twait_ns\tresident_mib\tbudget_mib\tlru_entries\tresident_groups_total\tresident_groups_current_layer\tresident_groups_other_layer\tresident_full_groups_total\tresident_full_groups_current_layer\texact_group_resident_slices\texact_group_resident_bytes\texact_group_resident_full\tresident_current_layer_mib\tresident_other_layer_mib\tstreams\tevictions\tcache_hit_total\tcache_miss_total\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_last_token\tevicted_age_tokens\tgroup_score\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open cache trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_PREFETCH_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->prefetch_trace = std::fopen(trace_path, "wb");
            if (ctx->prefetch_trace != nullptr) {
                std::fprintf(ctx->prefetch_trace,
                        "token\texec\tevent\treason\tlayer\texpert\tkind\trank\ttarget_bits\tclg_score\tadmit_score\tcandidate_value\tvictim_value\tvictim_layer\tvictim_expert\tspeculative_mib\tresident_mib\tbudget_mib\tstate\tresident_bits\tresident_mwq\tqueued\tqueued_bits\tprefetched\tresident_groups_total\tresident_groups_current_layer\tresident_groups_other_layer\texact_group_resident_slices\texact_group_resident_bytes\texact_group_full\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_score\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open prefetch trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_RESIDENT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->resident_trace = std::fopen(trace_path, "wb");
            if (ctx->resident_trace != nullptr) {
                std::fprintf(ctx->resident_trace,
                        "token\texec\tcurrent_layer\tresident_layer\texpert\tslices\tbytes\tfull\tbits_min\tbits_max\tmwq_slices\tprefetched_slices\ttouched_slices\tqueued_slices\tqueued_bits_max\tresident_mib\tbudget_mib\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_last_token\tgroup_score\treuse_keep\treuse_ema\treuse_gap_ema\treuse_observed\treuse_within_1\treuse_within_4\treuse_within_16\tnext_use_dist\tlayer_dist\treuse_predicted_soon\th0_current_layer\th1_next_token_keep\th4_short_reuse\tnext_token_conf\tnext_token_protect_until\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open resident trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_EVICT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->evict_trace = std::fopen(trace_path, "wb");
            if (ctx->evict_trace != nullptr) {
                std::fprintf(ctx->evict_trace,
                        "event\ttoken\texec\tlayer\texpert\treason\treload_gap\treleased_bytes\tresident_mib\tbudget_mib\tscore\treuse_keep\treuse_ema\treuse_gap_ema\treuse_observed\treuse_within_1\treuse_within_4\treuse_within_16\tseq_access\tseq_rank0\tseq_rate\tcache_hit\tcache_miss\tprefetch_hit\tprefetch_unused\thot_score\teamc_prior\team_replace_pred\team_replace_mass\team_replace_keep\tcct_conf\tcct_protect\tbad_reload_score\tbad_reload_effective\tbad_reload_age\tbad_reload_1\tbad_reload_4\tbad_reload_16\tbad_reload_protect\tnext_use_dist\tlayer_dist\tpinned\tactive\trecent\thigh_sequence\tearly_high\tspeculative_unused\tlayer_window\tpredicted_soon\tlast_evict_reason\tlayer_bad_reload\tlayer_last_bad_reload_token\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open evict trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_CCT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->cct_trace = std::fopen(trace_path, "wb");
            if (ctx->cct_trace != nullptr) {
                std::fprintf(ctx->cct_trace,
                        "event\ttoken\texec\tsrc_layer\tsrc_expert\ttarget_layer\ttarget_expert\tconf_before\tconf_after\treason\tresident_mib\tbudget_mib\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open CCT trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_NEXT_TOKEN_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->next_token_trace = std::fopen(trace_path, "wb");
            if (ctx->next_token_trace != nullptr) {
                std::fprintf(ctx->next_token_trace,
                        "event\ttoken\texec\tlayer\texpert\tconf_before\tconf_after\tprotect_until\tresident_mib\tbudget_mib\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open next-token trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_RUN_QUEUE_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->run_queue_trace = std::fopen(trace_path, "wb");
            if (ctx->run_queue_trace != nullptr) {
                std::fprintf(ctx->run_queue_trace,
                        "token\texec\tlayer\texpert\tkind\ttarget_bits\trank\tcount\tstate\torder\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open run queue trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_GROUP_PREFETCH_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->group_prefetch_trace = std::fopen(trace_path, "wb");
            if (ctx->group_prefetch_trace != nullptr) {
                std::fprintf(ctx->group_prefetch_trace,
                        "token\texec\tevent\tlayer\texpert\tkind\ttarget_bits\tstate\tresident_mib\tbudget_mib\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open group prefetch trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_ADMISSION_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->admission_trace = std::fopen(trace_path, "wb");
            if (ctx->admission_trace != nullptr) {
                std::fprintf(ctx->admission_trace,
                        "event\ttoken\texec\tlayer\texpert\tgeneration\taction\tgap\tcandidate_value\tvictim_value\tvictim_layer\tvictim_expert\tvictim_groups\tvictim_bytes\tresident_mib\tbudget_mib\tfeedback_ema\tfeedback_samples\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open admission trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_ADMISSION_REGRET_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->admission_regret_trace = std::fopen(trace_path, "wb");
            if (ctx->admission_regret_trace != nullptr) {
                std::fprintf(ctx->admission_regret_trace,
                        "event\ttoken\texec\tpair_id\taction\tincoming_layer\tincoming_expert\tvictim_layer\tvictim_expert\tincoming_gap\tvictim_gap\tregret_target\tregret_ema\tregret_samples\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open admission regret trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_LRB_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->lrb_trace = std::fopen(trace_path, "wb");
            if (ctx->lrb_trace != nullptr) {
                std::fprintf(ctx->lrb_trace,
                        "event\ttoken\texec\tpair_id\taction\tdecision_token\tdeadline\tincoming_gap\tvictim_gap\tregret_target\tpreference\tcorrect\tresident_mib\tbudget_mib\tstaging_mib\tresident_groups");
                const char * cols[] = {
                    "layer", "expert", "last_touch_gap", "last_evict_gap",
                    "next_use_dist", "layer_dist", "resident_mib", "cache_score",
                    "reuse_ema", "inter_token_gap_ema", "reuse_observed",
                    "reuse_within_1", "reuse_within_4", "reuse_within_16",
                    "bad_reload_score", "bad_reload_effective", "bad_reload_age",
                    "bad_reload_1", "bad_reload_4", "bad_reload_16",
                    "ghost_reload_risk_ema", "ghost_outcomes",
                    "admission_feedback_ema", "admission_feedback_samples",
                    "admission_regret_ema", "admission_regret_samples",
                    "seq_access", "seq_rank0_access", "seq_cache_hits",
                    "seq_cache_misses", "seq_prefetch_hits", "seq_prefetch_late",
                    "seq_prefetch_unused", "seq_future_hints",
                    "seq_future_rank0_hints", "seq_predicted", "seq_pred_enqueued",
                    "eamc_prior", "cct_conf", "next_token_conf", "pinned",
                    "active", "demand_pending",
                };
                for (const char * prefix : {"incoming", "victim"}) {
                    for (const char * col : cols) {
                        std::fprintf(ctx->lrb_trace, "\t%s_%s", prefix, col);
                    }
                }
                std::fprintf(ctx->lrb_trace, "\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open LRB trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_OLECAR_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->olecar_trace = std::fopen(trace_path, "wb");
            if (ctx->olecar_trace != nullptr) {
                std::fprintf(ctx->olecar_trace,
                        "event\tkind\ttoken\texec\tid\tdecision_id\tbucket_id\tpolicy_id\tpolicy\tlayer\texpert\tselected\tgap\tcost\testimated_cost\tdecision_token\tdeadline\tresident_mib\tbudget_mib\tweight_before\tweight_after\tpolicy_prob\tsupport_prob\tupdate_count\tfinal_score\tlru_score\trecency_score\tcache_score\tbad_reload_score\tnext_use_score\tlayer_score\treuse_score\tcct_score\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open OLECAR trace %s\n",
                        trace_path);
            }
        }
    }
#if defined(_SC_PAGESIZE)
    g_page = (size_t) sysconf(_SC_PAGESIZE);
#endif
    ctx->align = g_page;
    moe_sidecar_index(*ctx);
    moe_cache_create(*ctx);
    moe_eamc_load_sidecar(*ctx);
    if (params.enabled) {
        const int n = std::max(1, params.n_workers);
        ctx->workers.reserve(n);
        for (int i = 0; i < n; ++i) {
            ctx->workers.emplace_back(moe_worker_loop, ctx.get());
        }
    }
    return ctx;
}
```

### 注册 expert tensor：llama_moe_buffer_register()

来源：`src/llama-moe-buffer.cpp:1768-1839`

```cpp
bool llama_moe_buffer_register(
        llama_moe_buffer_context & ctx,
        ggml_tensor *              exps,
        int                        fd,
        size_t                     file_offset,
        size_t                     expert_stride,
        int                        n_expert) {
    if (!ctx.params.enabled || exps == nullptr || n_expert <= 0 || expert_stride == 0) {
        return false;
    }
    // Reopen the fd, preferring O_DIRECT so streamed expert pages bypass the page
    // cache (the exps region is still mmap-mapped, which would otherwise pin any
    // buffered-read cache pages and defeat the footprint reduction).
    int  dfd    = -1;
    bool direct = false;
#if defined(__linux__)
    char proc[64];
    std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
#if defined(O_DIRECT)
    dfd = open(proc, O_RDONLY | O_DIRECT);
    if (dfd >= 0) direct = true;
#endif
    if (dfd < 0) dfd = open(proc, O_RDONLY);
#endif
    if (dfd < 0) dfd = dup(fd);
    if (dfd < 0) {
        return false;
    }
    ctx.fds.push_back(dfd);

    struct stat st;
    const size_t fsize = (fstat(dfd, &st) == 0) ? (size_t) st.st_size : SIZE_MAX;

    moe_managed m;
    m.name        = ggml_get_name(exps);
    m.fd          = dfd;
    m.direct      = direct;
    m.fsize       = fsize;
    m.file_offset = file_offset;
    m.stride      = expert_stride;
    m.n_expert    = n_expert;
    m.nbytes      = ggml_nbytes(exps);
    m.type        = exps->type;
    m.elems       = ggml_nelements(exps) / n_expert;
    m.resident.assign(n_expert, ST_COLD);
    m.resident_mwq.assign(n_expert, false);
    m.resident_bits.assign(n_expert, 0);
    m.resident_size.assign(n_expert, 0);
    m.resident_touched.assign(n_expert, false);
    m.prefetched.assign(n_expert, false);
    m.demand_async_loaded.assign(n_expert, false);
    m.last_evicted_token.assign(n_expert, 0);
    m.queued.assign(n_expert, false);
    m.queued_bits.assign(n_expert, 0);
    m.activation.assign(n_expert, 0);
    m.lru_pos.resize(n_expert);
    std::sscanf(m.name.c_str(), "blk.%d.", &m.layer);

    auto res = ctx.by_name.emplace(m.name, std::move(m));
    if (res.second) {
        ctx.expert_total += res.first->second.nbytes;
        if (res.first->second.layer >= 0) {
            ctx.by_layer[res.first->second.layer].push_back(&res.first->second);
            for (int e = 0; e < n_expert; ++e) {
                moe_group_state & g = moe_group_get(ctx, res.first->second.layer, e);
                g.layer = res.first->second.layer;
                g.expert = e;
            }
        }
    }
    return true;
}
```

### expert 总量：llama_moe_buffer_expert_bytes()

来源：`src/llama-moe-buffer.cpp:1841-1843`

```cpp
size_t llama_moe_buffer_expert_bytes(const llama_moe_buffer_context * ctx) {
    return ctx != nullptr ? ctx->expert_total : 0;
}
```

### warm working set：llama_moe_buffer_warm_working_set_bytes()

来源：`src/llama-moe-buffer.cpp:1845-1960`

```cpp
size_t llama_moe_buffer_warm_working_set_bytes(
        llama_moe_buffer_context * ctx,
        double                    coverage) {
    if (ctx == nullptr || coverage <= 0.0) {
        return 0;
    }
    coverage = std::min(1.0, coverage);

    struct warm_candidate {
        int layer = -1;
        int expert = -1;
        size_t bytes = 0;
        double prob = 0.0;
        double gain = 0.0;
    };

    size_t total_bytes = 0;
    uint64_t total_groups = 0;
    double selected_coverage_sum = 0.0;
    uint64_t layers_with_groups = 0;

    std::lock_guard<std::mutex> lk(ctx->mtx);
    for (const auto & layer_it : ctx->by_layer) {
        const int layer = layer_it.first;
        const auto & tensors = layer_it.second;
        if (tensors.empty() || tensors[0] == nullptr || tensors[0]->n_expert <= 0) {
            continue;
        }

        const int n_expert = tensors[0]->n_expert;
        std::vector<warm_candidate> candidates;
        candidates.reserve((size_t) n_expert);
        double prob_sum = 0.0;
        bool has_observed_prior = false;

        for (int e = 0; e < n_expert; ++e) {
            size_t group_bytes = 0;
            for (const moe_managed * m : tensors) {
                if (m != nullptr && e < m->n_expert) {
                    group_bytes += m->stride;
                }
            }
            if (group_bytes == 0) {
                continue;
            }

            const auto git = ctx->groups.find(moe_group_key(layer, e));
            const moe_group_state * g = git == ctx->groups.end() ? nullptr : &git->second;
            const double observed =
                g == nullptr ? 0.0 :
                (double) g->seq_access +
                (double) g->access +
                std::max(0.0, g->hot_score);
            if (observed > 0.0) {
                has_observed_prior = true;
            }
            const double rd =
                g != nullptr && g->inter_token_gap_ema > 0.0
                    ? g->inter_token_gap_ema
                    : (double) std::max(1, n_expert);
            const double prob = observed > 0.0 ? observed : 1.0;
            candidates.push_back({ layer, e, group_bytes, prob,
                    prob * (double) group_bytes / std::max(1.0, rd) });
            prob_sum += prob;
        }

        if (candidates.empty() || prob_sum <= 0.0) {
            continue;
        }
        if (!has_observed_prior) {
            prob_sum = (double) candidates.size();
            for (auto & c : candidates) {
                c.prob = 1.0 / prob_sum;
                c.gain = c.prob * (double) c.bytes;
            }
        } else {
            for (auto & c : candidates) {
                c.prob /= prob_sum;
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                [](const warm_candidate & a, const warm_candidate & b) {
                    if (a.gain != b.gain) {
                        return a.gain > b.gain;
                    }
                    if (a.prob != b.prob) {
                        return a.prob > b.prob;
                    }
                    if (a.bytes != b.bytes) {
                        return a.bytes < b.bytes;
                    }
                    return a.expert < b.expert;
                });

        double layer_coverage = 0.0;
        for (const warm_candidate & c : candidates) {
            if (layer_coverage >= coverage) {
                break;
            }
            total_bytes += c.bytes;
            total_groups++;
            layer_coverage += c.prob;
        }
        selected_coverage_sum += std::min(layer_coverage, 1.0);
        layers_with_groups++;
    }

    ctx->warm_working_set_bytes = total_bytes;
    ctx->warm_working_set_groups = total_groups;
    ctx->warm_working_set_coverage =
        layers_with_groups > 0
            ? selected_coverage_sum / (double) layers_with_groups
            : 0.0;
    return total_bytes;
}
```

### budget 设置：llama_moe_buffer_set_budget()

来源：`src/llama-moe-buffer.cpp:1962-1969`

```cpp
void llama_moe_buffer_set_budget(llama_moe_buffer_context * ctx, size_t budget_bytes) {
    if (ctx == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx->mtx);
    ctx->params.budget_bytes = budget_bytes;
    ctx->params.budget_unbounded = false;
}
```

### budget plan：llama_moe_buffer_set_budget_plan()

来源：`src/llama-moe-buffer.cpp:1971-1985`

```cpp
void llama_moe_buffer_set_budget_plan(
        llama_moe_buffer_context * ctx,
        size_t                    budget_bytes,
        bool                      budget_unbounded,
        size_t                    planner_safe_budget_bytes,
        size_t                    planner_floor_bytes) {
    if (ctx == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx->mtx);
    ctx->params.budget_bytes = budget_unbounded ? 0 : budget_bytes;
    ctx->params.budget_unbounded = budget_unbounded;
    ctx->params.planner_safe_budget_bytes = planner_safe_budget_bytes;
    ctx->params.planner_floor_bytes = planner_floor_bytes;
}
```

### EAM transition 更新：moe_eam_update_transition_locked()

来源：`src/llama-moe-buffer.cpp:6795-6894`

```cpp
static void moe_eam_update_transition_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const std::vector<int> &   actual_experts,
        int                        n_expert) {
    if (layer < 0 || actual_experts.empty()) {
        return;
    }
    const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
    if (moe_cct_enabled() && (int) ctx.cct_recent_layer_experts.size() < n_layer) {
        ctx.cct_recent_layer_experts.resize((size_t) n_layer);
    }
    const bool adjacent =
        ctx.eam_last_actual_layer >= 0 &&
        ctx.eam_last_actual_layer != layer &&
        ((ctx.eam_last_actual_layer + 1) % n_layer) == layer;
    if (adjacent && !ctx.eam_last_layer_experts.empty()) {
        std::vector<uint8_t> predicted_before((size_t) n_expert, 0);
        if (moe_cct_enabled()) {
            for (int expert : actual_experts) {
                if (expert < 0 || expert >= n_expert) {
                    continue;
                }
                for (int prev_expert : ctx.eam_last_layer_experts) {
                    if (moe_cct_conf_from_source_locked(ctx, ctx.eam_last_actual_layer,
                                prev_expert, expert) > 0) {
                        predicted_before[(size_t) expert] = 1;
                        break;
                    }
                }
            }
        }
        for (int prev_expert : ctx.eam_last_layer_experts) {
            if (prev_expert < 0) {
                continue;
            }
            std::vector<uint32_t> & row =
                ctx.eam_layer_transition[moe_group_key(ctx.eam_last_actual_layer, prev_expert)];
            if ((int) row.size() < n_expert) {
                row.resize((size_t) n_expert, 0);
            }
            std::vector<moe_cct_entry> * cct_row = nullptr;
            if (moe_cct_enabled()) {
                cct_row = &ctx.cct_transition[moe_group_key(ctx.eam_last_actual_layer, prev_expert)];
                if ((int) cct_row->size() < n_expert) {
                    cct_row->resize((size_t) n_expert);
                }
            }
            std::vector<uint8_t> actual_mask((size_t) n_expert, 0);
            for (int expert : actual_experts) {
                if (expert >= 0 && expert < n_expert && row[(size_t) expert] != UINT32_MAX) {
                    row[(size_t) expert]++;
                    actual_mask[(size_t) expert] = 1;
                }
            }
            if (cct_row != nullptr) {
                for (int expert = 0; expert < n_expert; ++expert) {
                    moe_cct_entry & ent = (*cct_row)[(size_t) expert];
                    const uint8_t before = ent.conf;
                    if (actual_mask[(size_t) expert]) {
                        ent.conf = before == 0 ? 2 : (uint8_t) std::min<int>(3, before + 1);
                        if (before == 0) {
                            ent.replace++;
                            ctx.cct_replaced.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            ent.hit++;
                            ctx.cct_hits.fetch_add(1, std::memory_order_relaxed);
                        }
                        moe_cct_trace_write_locked(ctx, "update", ctx.eam_last_actual_layer,
                                prev_expert, layer, expert, before, ent.conf,
                                before == 0 ? "replace_actual" : "actual_hit");
                    } else if (before > 0) {
                        ent.conf = (uint8_t) (before - 1);
                        ent.unused++;
                        ctx.cct_unused.fetch_add(1, std::memory_order_relaxed);
                        moe_cct_trace_write_locked(ctx, "update", ctx.eam_last_actual_layer,
                                prev_expert, layer, expert, before, ent.conf, "not_used");
                    }
                    ent.last_update_token = ctx.profile_token_epoch;
                }
            }
        }
        if (moe_cct_enabled()) {
            for (int expert : actual_experts) {
                if (expert >= 0 && expert < n_expert && !predicted_before[(size_t) expert]) {
                    ctx.cct_misses.fetch_add(1, std::memory_order_relaxed);
                    moe_cct_trace_write_locked(ctx, "miss", ctx.eam_last_actual_layer,
                            -1, layer, expert, 0, 2, "actual_not_predicted");
                }
            }
            ctx.cct_updates.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ctx.eam_last_actual_layer = layer;
    ctx.eam_last_actual_token_epoch = ctx.profile_token_epoch;
    ctx.eam_last_layer_experts = actual_experts;
    if (moe_cct_enabled() && layer >= 0 && layer < (int) ctx.cct_recent_layer_experts.size()) {
        ctx.cct_recent_layer_experts[(size_t) layer] = actual_experts;
    }
}
```

### EAM 预测与 prefetch：moe_eam_predict_after_route()

来源：`src/llama-moe-buffer.cpp:6896-7018`

```cpp
static void moe_eam_predict_after_route(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const std::vector<int> &   actual_experts,
        int                        n_expert) {
    if (!moe_eam_predict_enabled() || layer < 0 || actual_experts.empty()) {
        return;
    }

    struct predict_candidate {
        int layer = -1;
        int expert = -1;
        int rank = 0;
        double score = 0.0;
    };

    std::vector<predict_candidate> candidates;
    const int depth = moe_eam_predict_depth(ctx);
    const int per_layer = moe_eam_predict_max_tasks_per_layer();
    const int per_token = moe_eam_predict_max_tasks_per_token();
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        moe_eam_update_transition_locked(ctx, layer, actual_experts, n_expert);
        const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
        for (int d = 1; d <= depth; ++d) {
            const int target_layer = (layer + d) % n_layer;
            auto lit = ctx.by_layer.find(target_layer);
            if (lit == ctx.by_layer.end() || lit->second.empty() || lit->second[0] == nullptr) {
                continue;
            }
            const int target_n_expert = lit->second[0]->n_expert;
            std::vector<predict_candidate> layer_candidates;
            layer_candidates.reserve((size_t) target_n_expert);
            for (int expert = 0; expert < target_n_expert; ++expert) {
                bool needs_prefetch = false;
                for (const moe_managed * m : lit->second) {
                    if (m == nullptr) {
                        continue;
                    }
                    const int target_bits = moe_target_bits_for_rank(ctx, *m, expert, 0);
                    if (moe_prefetch_needed_unlocked(ctx, *m, expert, target_bits)) {
                        needs_prefetch = true;
                        break;
                    }
                }
                if (!needs_prefetch) {
                    continue;
                }
                moe_group_state & g = moe_group_get(ctx, target_layer, expert);
                const double score = moe_eam_predict_score_locked(
                        ctx, g, layer, actual_experts, target_layer, expert);
                if (score < moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_MIN_SCORE", 75.0)) {
                    continue;
                }
                layer_candidates.push_back({target_layer, expert, 0, score});
            }
            std::sort(layer_candidates.begin(), layer_candidates.end(),
                    [](const predict_candidate & a, const predict_candidate & b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.expert < b.expert;
            });
            const int n_take = std::min<int>(per_layer, (int) layer_candidates.size());
            for (int i = 0; i < n_take && (int) candidates.size() < per_token; ++i) {
                layer_candidates[(size_t) i].rank = i;
                candidates.push_back(layer_candidates[(size_t) i]);
                moe_group_state & g = moe_group_get(ctx, layer_candidates[(size_t) i].layer,
                        layer_candidates[(size_t) i].expert);
                g.seq_predicted++;
                g.seq_predict_score_sum += layer_candidates[(size_t) i].score;
            }
        }
    }

    if (candidates.empty()) {
        return;
    }

    ctx.eam_predict_runs.fetch_add(1, std::memory_order_relaxed);
    ctx.eam_predict_candidates.fetch_add((uint64_t) candidates.size(), std::memory_order_relaxed);

    int enqueued = 0;
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        for (const predict_candidate & c : candidates) {
            auto lit = ctx.by_layer.find(c.layer);
            if (lit == ctx.by_layer.end()) {
                continue;
            }
            bool any = false;
            for (moe_managed * m : lit->second) {
                if (m == nullptr) {
                    continue;
                }
                const int target_bits = moe_target_bits_for_rank(ctx, *m, c.expert, c.rank);
                if (!moe_prefetch_needed_unlocked(ctx, *m, c.expert, target_bits)) {
                    continue;
                }
                any = moe_enqueue_prefetch(ctx, *m, c.expert, c.rank, (float) c.score, true) || any;
            }
            if (any) {
                ++enqueued;
                std::lock_guard<std::mutex> mlk(ctx.mtx);
                moe_group_state & g = moe_group_get(ctx, c.layer, c.expert);
                g.seq_pred_enqueued++;
            }
        }
    }

    if (enqueued > 1) {
        ctx.cv_q.notify_all();
    } else if (enqueued == 1) {
        ctx.cv_q.notify_one();
    }
    ctx.eam_predict_enqueued.fetch_add((uint64_t) enqueued, std::memory_order_relaxed);
    ctx.eam_predict_dropped.fetch_add((uint64_t) (candidates.size() - (size_t) enqueued), std::memory_order_relaxed);
    uint64_t score_sum = 0;
    for (const predict_candidate & c : candidates) {
        score_sum += (uint64_t) std::max(0.0, c.score * 1000.0);
    }
    ctx.eam_predict_score_x1000.fetch_add(score_sum, std::memory_order_relaxed);
}
```

### LRU/Belady-like eviction：moe_evict_lru()

来源：`src/llama-moe-buffer.cpp:7023-8230`

```cpp
static bool moe_evict_lru(
        llama_moe_buffer_context &              ctx,
        size_t                                  target_release_bytes = 0,
        const moe_demand_admission_request *    admission_request = nullptr,
        moe_demand_admission_result *           admission_result = nullptr,
        const std::vector<moe_demand_admission_request> * admission_batch = nullptr,
        std::vector<moe_demand_admission_result> * batch_results = nullptr,
        size_t                                  batch_free_bytes = 0,
        size_t                                  batch_staging_bytes = 0) {
    if (ctx.lru.empty()) {
        return false;
    }
    const uint64_t prof_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    const uint64_t prof_scan_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    uint64_t prof_sort_us = 0;
    uint64_t prof_online_us = 0;
    uint64_t prof_admission_us = 0;
    uint64_t prof_trace_us = 0;
    uint64_t prof_release_us = 0;
    uint64_t prof_candidate_us = 0;
    uint64_t prof_resident_queries = 0;

    moe_ghost_expire_locked(ctx);
    moe_olecar_expire_locked(ctx);
    moe_refresh_pins_locked(ctx);

    struct victim_candidate {
        int layer = -1;
        int expert = -1;
        double score = -1.0e300;
        double hot_score = 0.0;
        double lru_score = 0.0;
        double recency_score = 0.0;
        double cache_policy_score = 0.0;
        double bad_reload_policy_score = 0.0;
        double next_use_score = 0.0;
        double layer_score = 0.0;
        double reuse_score = 0.0;
        double cct_score = 0.0;
        uint64_t age = 0;
        bool belady = false;
        enum reason_t {
            SPEC_UNUSED,
            LOW_SCORE,
            FAR_FUTURE,
            LAYER_WINDOW,
            REUSE_LOW,
            EAM_REPLACE,
            RELAXED,
        } reason = LOW_SCORE;
    };

    struct olecar_policy_best {
        const char * name = "";
        int id = -1;
        bool has = false;
        double score = -1.0e300;
        victim_candidate candidate;
    };

    victim_candidate best;
    victim_candidate temporary_best;
    std::vector<olecar_policy_best> olecar_best = {
        {"final", 0, false, -1.0e300, {}},
        {"lru", 1, false, -1.0e300, {}},
        {"recency", 2, false, -1.0e300, {}},
        {"cache", 3, false, -1.0e300, {}},
        {"bad_reload", 4, false, -1.0e300, {}},
        {"next_use", 5, false, -1.0e300, {}},
        {"layer", 6, false, -1.0e300, {}},
        {"reuse", 7, false, -1.0e300, {}},
        {"cct", 8, false, -1.0e300, {}},
    };
    auto olecar_consider = [&](victim_candidate candidate) {
        if (candidate.layer < 0 || candidate.expert < 0) {
            return;
        }
        const double scores[] = {
            candidate.score,
            candidate.lru_score,
            candidate.recency_score,
            candidate.cache_policy_score,
            candidate.bad_reload_policy_score,
            candidate.next_use_score,
            candidate.layer_score,
            candidate.reuse_score,
            candidate.cct_score,
        };
        for (size_t i = 0; i < olecar_best.size(); ++i) {
            const double s = scores[i];
            olecar_policy_best & best_policy = olecar_best[i];
            const bool better =
                !best_policy.has || s > best_policy.score ||
                (s == best_policy.score &&
                    (candidate.age > best_policy.candidate.age ||
                        (candidate.age == best_policy.candidate.age &&
                            (candidate.hot_score < best_policy.candidate.hot_score ||
                                (candidate.hot_score == best_policy.candidate.hot_score &&
                                    (candidate.layer < best_policy.candidate.layer ||
                                        (candidate.layer == best_policy.candidate.layer &&
                                            candidate.expert < best_policy.candidate.expert)))))));
            if (better) {
                best_policy.has = true;
                best_policy.score = s;
                best_policy.candidate = candidate;
            }
        }
    };
    const bool olecar_online = moe_env_flag("LLAMA_LAZY_MOE_OLECAR_ONLINE", 0);
    const char * force_policy_env = std::getenv("LLAMA_LAZY_MOE_OLECAR_FORCE_POLICY");
    std::string force_policy = force_policy_env == nullptr ? "" : std::string(force_policy_env);
    const bool explicit_force_policy = !force_policy.empty();
    const double budget_reference = (double) std::max<size_t>(
            1,
            ctx.warm_working_set_bytes > 0
                ? ctx.warm_working_set_bytes
                : ctx.expert_total);
    if (force_policy.empty() && moe_budget_bounded(ctx) &&
            (double) ctx.params.budget_bytes < budget_reference) {
        force_policy = "cache";
    }
    auto forced_policy_score = [&](const victim_candidate & candidate, double * out_score) {
        if (force_policy.empty() || force_policy == "final") {
            return false;
        }
        if (force_policy == "lru") {
            *out_score = candidate.lru_score;
        } else if (force_policy == "recency") {
            *out_score = candidate.recency_score;
        } else if (force_policy == "cache") {
            *out_score = candidate.cache_policy_score;
        } else if (force_policy == "bad_reload") {
            *out_score = candidate.bad_reload_policy_score;
        } else if (force_policy == "next_use") {
            *out_score = candidate.next_use_score;
        } else if (force_policy == "layer") {
            *out_score = candidate.layer_score;
        } else if (force_policy == "reuse") {
            *out_score = candidate.reuse_score;
        } else if (force_policy == "cct") {
            *out_score = candidate.cct_score;
        } else {
            return false;
        }
        return true;
    };
    auto better_candidate = [](const victim_candidate & a, const victim_candidate & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        if (a.age != b.age) {
            return a.age > b.age;
        }
        if (a.hot_score != b.hot_score) {
            return a.hot_score < b.hot_score;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    };
    std::vector<victim_candidate> normal_candidates;
    std::vector<victim_candidate> temporary_candidates;
    if (target_release_bytes > 0) {
        normal_candidates.reserve(ctx.groups.size());
        temporary_candidates.reserve(ctx.groups.size());
    }
    double temporary_keep_score = 1.0e300;
    const int recent_tokens = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_RECENT_TOKENS", 3);
    const uint64_t bad_reload_guard_tokens = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_BAD_RELOAD_GUARD_TOKENS", 16));
    const double bad_reload_protect_score =
        moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_PROTECT_SCORE", 6.0);
    const bool cct_evict_active = moe_cct_evict_enabled() && moe_cct_lowmem_active(ctx);
    const bool need_cct_score = cct_evict_active || olecar_online ||
        force_policy == "cct" || ctx.olecar_trace != nullptr;
    const int cct_protect_conf = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_CONF", 3);
    const int cct_protect_distance = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_DISTANCE", 1);
    const bool reuse_evict_active = moe_reuse_evict_enabled();
    const int aggressive_lazy_mode = std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EVICT_AGGRESSIVE_LAZY", 2));
    const double reuse_protect_score = moe_env_f64("LLAMA_LAZY_MOE_REUSE_PROTECT_SCORE", 260.0);
    const int reuse_protect_tokens = moe_env_i32("LLAMA_LAZY_MOE_REUSE_PROTECT_TOKENS", 4);
    const double high_keep_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_SCORE", 125.0);
    const double high_keep_rate = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_RATE", 0.0080);
    const int early_high_layers = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_LAYERS", 4);
    const double early_high_keep_score =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_KEEP_SCORE", 95.0);
    const double relaxed_next_use_weight = moe_env_f64("LLAMA_LAZY_MOE_RELAXED_NEXT_USE_WEIGHT", 0.0);
    const double relaxed_layer_distance_weight =
        moe_env_f64("LLAMA_LAZY_MOE_RELAXED_LAYER_DISTANCE_WEIGHT", 2.5e8);
    const uint64_t eam_replace_next_guard = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EAM_REPLACE_NEXT_USE_GUARD",
                std::max(4, ctx.params.active_window)));
    const double layer_reload_protect_penalty =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_PENALTY", 4.0e8);
    const double spec_unused_bonus_env =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_SPEC_UNUSED_BONUS", 3.0e8);
    const double low_score_ceil = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_LOW_SCORE_CEIL", 140.0);
    const double reuse_evict_ceil = moe_env_f64("LLAMA_LAZY_MOE_REUSE_EVICT_CEIL", 170.0);
    const double reuse_evict_weight = moe_env_f64("LLAMA_LAZY_MOE_REUSE_EVICT_WEIGHT", 65536.0);
    const double reuse_keep_weight = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_WEIGHT", 0.0);
    const double reuse_protect_penalty_env =
        moe_env_f64("LLAMA_LAZY_MOE_REUSE_PROTECT_PENALTY", 3.0e8);
    const double layer_reserve_evict_bonus =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_EVICT_BONUS", 2.5e8);
    const double layer_reserve_protect_penalty =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_PROTECT_PENALTY", 3.0e8);
    const double layer_reserve_persistent_score =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_PERSISTENT_SCORE", 120.0);
    const double bad_reload_evict_weight =
        moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_EVICT_WEIGHT", 2.0e8);
    const double cct_evict_keep_weight =
        moe_env_f64("LLAMA_LAZY_MOE_CCT_EVICT_KEEP_WEIGHT", 2.5e8);
    const double eam_replace_score_scale =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_SCORE_SCALE", 1.0e12);
    const bool eam_replace = moe_eam_replace_enabled();
    bool eam_replace_stats_ready = false;
    moe_eam_replace_stats eam_replace_stats;
    uint64_t scan_entries = 0;
    uint64_t scan_unique = 0;
    uint64_t scan_duplicates = 0;
    uint64_t scan_absolute = 0;
    uint64_t scan_temporary = 0;
    uint64_t scan_normal = 0;
    size_t scan_candidate_bytes = 0;
    if (++ctx.evict_scan_generation == 0) {
        for (auto & kv : ctx.groups) {
            kv.second.evict_scan_generation = 0;
        }
        ctx.evict_scan_generation = 1;
    }
    const uint64_t scan_generation = ctx.evict_scan_generation;
    size_t target_window_bytes = 0;
    bool layer_reserve_needed = false;
    const bool layer_reload_protect_enabled = moe_env_flag("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT", 0);
    const bool layer_reload_hard_protect = moe_env_flag("LLAMA_LAZY_MOE_LAYER_RELOAD_HARD_PROTECT", 1);
    const int layer_feedback_window = moe_env_i32("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_WINDOW", 16);
    const uint64_t layer_feedback_min = (uint64_t) std::max(1,
            moe_env_i32("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_MIN", 2));
    auto layer_feedback_protected = [&](const moe_group_state & g, bool layer_window) {
        if (!layer_reload_protect_enabled || !layer_reserve_needed || layer_window ||
                g.layer_window_bad_reload < layer_feedback_min) {
            return false;
        }
        const uint64_t age =
            g.layer_window_last_bad_reload_token == 0 || ctx.profile_token_epoch < g.layer_window_last_bad_reload_token ?
            UINT64_MAX : ctx.profile_token_epoch - g.layer_window_last_bad_reload_token;
        return layer_feedback_window <= 0 ||
            (age != UINT64_MAX && age <= (uint64_t) layer_feedback_window);
    };
    if (ctx.evict_target_layer >= 0 && ctx.evict_layer_reserve_bytes > 0) {
        const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
        for (int d = 0; d <= std::max(0, ctx.evict_target_ahead); ++d) {
            target_window_bytes += moe_layer_resident_bytes_locked(ctx, (ctx.evict_target_layer + d) % n_layer);
        }
        layer_reserve_needed = target_window_bytes < ctx.evict_layer_reserve_bytes;
    }
    std::vector<std::pair<moe_managed *, int>> evict_scan_items;
    const bool group_lru_scan = moe_env_flag("LLAMA_LAZY_MOE_GROUP_LRU_SCAN", 1);
    const int sample_k_env = moe_env_i32("LLAMA_LAZY_MOE_EVICT_SAMPLE_K", 0);
    const size_t sample_k = sample_k_env <= 0 ? 0 : (size_t) sample_k_env;
    const bool sample_fallback_bytes =
        moe_env_flag("LLAMA_LAZY_MOE_EVICT_SAMPLE_FALLBACK_BYTES", 0);
    const int cold_window_env = moe_env_i32("LLAMA_LAZY_MOE_EVICT_COLD_WINDOW", 0);
    const size_t cold_window = cold_window_env <= 0 ? 0 : (size_t) cold_window_env;
    std::vector<std::pair<moe_managed *, int>> all_scan_items;
    std::vector<std::pair<moe_managed *, int>> candidate_pool;
    bool sampled_scan = false;
    bool cold_window_scan = false;
    if (group_lru_scan) {
        if (ctx.group_lru.empty() && !ctx.lru.empty()) {
            moe_group_lru_rebuild_locked(ctx);
        }
        all_scan_items.reserve(ctx.group_lru.size());
        for (auto it = ctx.group_lru.rbegin(); it != ctx.group_lru.rend(); ++it) {
            const int layer = (int) (uint32_t) (*it >> 32);
            const int e = (int) (uint32_t) *it;
            auto lit = ctx.by_layer.find(layer);
            if (lit == ctx.by_layer.end() || lit->second.empty() || lit->second.front() == nullptr) {
                continue;
            }
            all_scan_items.push_back({lit->second.front(), e});
        }
        if (cold_window > 0 && cold_window < all_scan_items.size()) {
            cold_window_scan = true;
            candidate_pool.assign(all_scan_items.begin(), all_scan_items.begin() + cold_window);
            ctx.evict_cold_window_calls.fetch_add(1, std::memory_order_relaxed);
            ctx.evict_cold_window_items.fetch_add(candidate_pool.size(), std::memory_order_relaxed);
            ctx.evict_cold_window_source.fetch_add(all_scan_items.size(), std::memory_order_relaxed);
        } else {
            candidate_pool = all_scan_items;
        }
        size_t effective_sample_k = sample_k;
        if (effective_sample_k > 0 && target_release_bytes > 0 && ctx.resident_bytes > 0 &&
                !ctx.group_lru.empty()) {
            const size_t avg_group_bytes =
                std::max<size_t>(1, ctx.resident_bytes / ctx.group_lru.size());
            const size_t release_groups =
                (target_release_bytes + avg_group_bytes - 1) / avg_group_bytes;
            effective_sample_k = std::max(effective_sample_k, release_groups + sample_k);
        }
        if (effective_sample_k > 0 && effective_sample_k < candidate_pool.size()) {
            sampled_scan = true;
            evict_scan_items.reserve(effective_sample_k);
            std::unordered_set<size_t> selected_indexes;
            uint64_t x = scan_generation * 0x9e3779b97f4a7c15ull;
            x ^= ctx.profile_token_epoch + 0xbf58476d1ce4e5b9ull + (x << 6) + (x >> 2);
            x ^= ctx.exec_epoch + 0x94d049bb133111ebull + (x << 6) + (x >> 2);
            size_t attempts = 0;
            const size_t max_attempts = candidate_pool.size() * 4;
            while (evict_scan_items.size() < effective_sample_k && attempts++ < max_attempts) {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                const size_t idx =
                    (size_t) ((x * 0x2545f4914f6cdd1dull) % candidate_pool.size());
                if (!selected_indexes.insert(idx).second) {
                    continue;
                }
                evict_scan_items.push_back(candidate_pool[idx]);
            }
            for (size_t i = 0; evict_scan_items.size() < effective_sample_k && i < candidate_pool.size(); ++i) {
                if (selected_indexes.insert(i).second) {
                    evict_scan_items.push_back(candidate_pool[i]);
                }
            }
            ctx.evict_sample_calls.fetch_add(1, std::memory_order_relaxed);
            ctx.evict_sample_k.fetch_add(evict_scan_items.size(), std::memory_order_relaxed);
            ctx.evict_sample_source.fetch_add(candidate_pool.size(), std::memory_order_relaxed);
        } else {
            evict_scan_items = candidate_pool;
        }
    } else {
        evict_scan_items.reserve(ctx.lru.size());
        for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it) {
            evict_scan_items.push_back(*it);
        }
        all_scan_items = evict_scan_items;
    }
    auto scan_pass = [&](const std::vector<std::pair<moe_managed *, int>> & scan_items) {
    for (const auto & scan_item : scan_items) {
        ++scan_entries;
        moe_managed * m = scan_item.first;
        const int e = scan_item.second;
        if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
            continue;
        }

        const uint64_t key = moe_group_key(m->layer, e);
        auto git = ctx.groups.find(key);
        if (git == ctx.groups.end()) {
            moe_group_get(ctx, m->layer, e);
            git = ctx.groups.find(key);
        }
        moe_group_state & g = git->second;
        if (g.evict_scan_generation == scan_generation) {
            ++scan_duplicates;
            continue;
        }
        g.evict_scan_generation = scan_generation;

        ++prof_resident_queries;
        const uint64_t prof_candidate_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const size_t bytes = moe_group_resident_bytes(ctx, m->layer, e);
        if (bytes == 0) {
            continue;
        }
        ++scan_unique;
        const uint64_t group_age =
            g.last_used_epoch == 0 || ctx.exec_epoch < g.last_used_epoch ?
            UINT64_MAX : ctx.exec_epoch - g.last_used_epoch;
        const uint64_t token_age =
            g.last_used_token_epoch == 0 || ctx.profile_token_epoch < g.last_used_token_epoch ?
            UINT64_MAX : ctx.profile_token_epoch - g.last_used_token_epoch;

        g.hot_score = moe_group_cache_score(ctx, g, bytes);
        const bool current_layer = ctx.profile_last_layer >= 0 && m->layer == ctx.profile_last_layer;
        const bool layer_window = moe_layer_in_evict_window(ctx, m->layer);
        const bool speculative_unused = moe_group_has_speculative_unused(ctx, m->layer, e);
        const bool layer_feedback_guard = !speculative_unused &&
            layer_feedback_protected(g, layer_window);
        if (layer_feedback_guard) {
            ctx.layer_reload_protect.fetch_add(1, std::memory_order_relaxed);
        }

        moe_evict_protect_options opt;
        opt.speculative_unused = speculative_unused;
        opt.current_layer = current_layer;
        opt.layer_window = layer_window;
        opt.protect_layer_window = layer_reserve_needed;
        opt.protect_recent = true;
        opt.aggressive_lazy_mode = aggressive_lazy_mode;
        opt.recent_tokens = recent_tokens;
        opt.bad_reload_guard_tokens = bad_reload_guard_tokens;
        opt.bad_reload_score = bad_reload_protect_score;
        opt.cct_evict_active = cct_evict_active;
        opt.need_cct_score = need_cct_score;
        opt.cct_protect_conf = cct_protect_conf;
        opt.cct_protect_distance = cct_protect_distance;
        opt.reuse_evict_active = reuse_evict_active;
        opt.reuse_protect_score = reuse_protect_score;
        opt.reuse_protect_tokens = reuse_protect_tokens;
        opt.high_keep_score = high_keep_score;
        opt.high_keep_rate = high_keep_rate;
        opt.early_high_layers = early_high_layers;
        opt.early_high_keep_score = early_high_keep_score;
        opt.relaxed_next_use_weight = relaxed_next_use_weight;
        opt.relaxed_layer_distance_weight = relaxed_layer_distance_weight;
        const moe_evict_protect_result protect =
            moe_group_evict_protect_locked(ctx, g, m->layer, e, g.hot_score, opt);
        if (!speculative_unused) {
            if (protect.recent) {
                ctx.eam_evict_protect_recent.fetch_add(1, std::memory_order_relaxed);
            }
            if (protect.high_sequence || moe_group_is_hot(ctx, m->layer, e)) {
                ctx.eam_evict_protect_high.fetch_add(1, std::memory_order_relaxed);
            }
            if (protect.early_high) {
                ctx.eam_evict_protect_early.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (protect.reason != moe_evict_protect_reason::NONE) {
            switch (protect.reason) {
                case moe_evict_protect_reason::PINNED:
                    ctx.eam_evict_protect_pinned.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::ACTIVE:
                case moe_evict_protect_reason::CURRENT_LAYER:
                    ctx.eam_evict_protect_active.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::BAD_RELOAD:
                    g.bad_reload_protect_hits++;
                    ctx.bad_reload_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::NEXT_TOKEN:
                    g.next_token_protect_hits++;
                    ctx.next_token_evict_protect.fetch_add(1, std::memory_order_relaxed);
                    moe_next_token_trace_write_locked(ctx, "evict_keep", m->layer, e,
                            g.next_token_conf, g.next_token_conf, g.next_token_protect_until,
                            "global_victim_protect");
                    break;
                case moe_evict_protect_reason::CCT:
                    g.cct_protect_hits++;
                    ctx.cct_evict_protect.fetch_add(1, std::memory_order_relaxed);
                    moe_cct_trace_write_locked(ctx, "evict_protect", ctx.eam_last_actual_layer, -1,
                            m->layer, e, protect.cct_conf, protect.cct_conf, "near_high_conf");
                    break;
                case moe_evict_protect_reason::PREDICTED_SOON:
                    ctx.reuse_predict_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::REUSE:
                    ctx.reuse_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                default:
                    break;
            }
        }

        const uint64_t dist = protect.next_use_dist;
        const int layer_dist = protect.layer_dist;
        const bool recent = protect.recent;
        const bool high_sequence = protect.high_sequence;
        const bool early_high = protect.early_high;
        const double reuse_keep = protect.reuse_keep;
        const double bad_reload_effective = protect.bad_reload_effective;
        const uint8_t cct_conf = protect.cct_conf;
        const bool cct_lowmem = cct_evict_active;
        const bool eam_replace_next_protected =
            eam_replace && !speculative_unused && eam_replace_next_guard > 0 &&
            dist != UINT64_MAX && dist <= eam_replace_next_guard;
        if (eam_replace_next_protected) {
            ctx.reuse_predict_protect.fetch_add(1, std::memory_order_relaxed);
        }
        if (protect.protect_class == moe_evict_protect_class::ABSOLUTE) {
            ++scan_absolute;
            continue;
        }
        const bool external_temporary =
            (layer_feedback_guard && layer_reload_hard_protect) || eam_replace_next_protected;
        if (protect.protect_class == moe_evict_protect_class::TEMPORARY || external_temporary) {
            ++scan_temporary;
            double keep_score = protect.temporary_keep_score;
            if (layer_feedback_guard && layer_reload_hard_protect) {
                keep_score += layer_reload_protect_penalty *
                    (double) std::min<uint64_t>(4, g.layer_window_bad_reload);
            }
            if (eam_replace_next_protected) {
                keep_score += (double) (eam_replace_next_guard + 1 - dist) * 1.0e8;
            }
            victim_candidate candidate;
            candidate.layer = m->layer;
            candidate.expert = e;
            candidate.score = -keep_score;
            candidate.hot_score = g.hot_score;
            candidate.lru_score = group_age == UINT64_MAX ? 1.0e9 : (double) group_age;
            candidate.recency_score = token_age == UINT64_MAX ? 1.0e9 : (double) token_age;
            candidate.cache_policy_score = -g.hot_score;
            candidate.bad_reload_policy_score = -bad_reload_effective;
            candidate.next_use_score = dist == UINT64_MAX ? 1.0e9 : (double) dist;
            candidate.layer_score = (double) layer_dist;
            candidate.reuse_score = -reuse_keep;
            candidate.cct_score = -(double) cct_conf;
            candidate.age = group_age;
            candidate.belady = protect.future_ranked;
            candidate.reason = victim_candidate::RELAXED;
            double forced_score = 0.0;
            const bool forced = forced_policy_score(candidate, &forced_score);
            if (forced) {
                candidate.score = forced_score;
            }
            olecar_consider(candidate);
            if (forced) {
                if (target_release_bytes > 0) {
                    normal_candidates.push_back(candidate);
                }
                if (better_candidate(candidate, best)) {
                    best = candidate;
                }
                continue;
            }
            if (target_release_bytes > 0) {
                temporary_candidates.push_back(candidate);
                scan_candidate_bytes += bytes;
            }
            const bool better_temporary =
                keep_score < temporary_keep_score ||
                (keep_score == temporary_keep_score &&
                    (group_age > temporary_best.age ||
                        (group_age == temporary_best.age &&
                            (g.hot_score < temporary_best.hot_score ||
                                (g.hot_score == temporary_best.hot_score &&
                                    (m->layer < temporary_best.layer ||
                                        (m->layer == temporary_best.layer && e < temporary_best.expert)))))));
            if (better_temporary) {
                temporary_keep_score = keep_score;
                temporary_best = candidate;
            }
            continue;
        }
        ++scan_normal;
        const bool predicted_soon = protect.predicted_soon;
        const bool reuse_protected = protect.reuse_protected;
        const bool spec_evictable = speculative_unused &&
            moe_group_speculative_unused_evictable(ctx, g, dist, layer_dist);
        const double speculative_bonus = spec_evictable ?
            spec_unused_bonus_env : 0.0;
        const double low_score_bonus = std::max(0.0,
                low_score_ceil - g.hot_score) * 16384.0;
        const double dist_score = dist == UINT64_MAX ? 1.0e9 : (double) dist * 8192.0;
        const double recent_penalty = recent ? 1.0e8 : 0.0;
        const double pressure_scale = ctx.params.pressure_scale_spec_guard ? moe_pressure_scale_locked(ctx) : 0.0;
        const double spec_guard_penalty = speculative_unused && !spec_evictable ? 4.0e8 * (1.0 - pressure_scale) : 0.0;
        const double size_bonus = (double) bytes / 1048576.0;
        const double bit_bonus = moe_group_resident_bit_score(ctx, m->layer, e) * 8.0;
        const double high_penalty = high_sequence || moe_group_is_hot(ctx, m->layer, e) ? 5.0e8 : 0.0;
        const double early_penalty = early_high ? 2.0e8 : 0.0;
        const bool reuse_low_candidate = reuse_evict_active && g.reuse_observed > 0 &&
            !predicted_soon && !layer_window &&
            reuse_keep < reuse_evict_ceil;
        if (reuse_low_candidate) {
            ctx.reuse_low_candidates.fetch_add(1, std::memory_order_relaxed);
        }
        const double reuse_low_bonus = reuse_low_candidate ?
            std::max(0.0, reuse_evict_ceil - reuse_keep) *
                reuse_evict_weight : 0.0;
        const double reuse_keep_penalty = reuse_evict_active ?
            reuse_keep * reuse_keep_weight : 0.0;
        const double reuse_protect_penalty = reuse_protected ?
            reuse_protect_penalty_env : 0.0;
        const double layer_window_bonus = layer_reserve_needed && !layer_window ?
            layer_reserve_evict_bonus : 0.0;
        const double layer_window_penalty = layer_reserve_needed && layer_window ?
            layer_reserve_protect_penalty : 0.0;
        const double layer_feedback_penalty = layer_feedback_guard ?
            layer_reload_protect_penalty *
                (double) std::min<uint64_t>(4, g.layer_window_bad_reload) : 0.0;
        const double persistent_penalty = !layer_window && ctx.evict_persistent_bytes > 0 &&
            ctx.resident_bytes <= ctx.evict_persistent_bytes && g.hot_score >=
                layer_reserve_persistent_score ? 2.0e8 : 0.0;
        const double bad_reload_penalty = !speculative_unused ?
            bad_reload_effective * bad_reload_evict_weight : 0.0;
        if (bad_reload_penalty > 0.0) {
            g.bad_reload_soft_hits++;
            ctx.bad_reload_soft_keep.fetch_add(1, std::memory_order_relaxed);
        }
        const double cct_keep_penalty = cct_lowmem ?
            (double) cct_conf * cct_evict_keep_weight : 0.0;
        if (cct_keep_penalty > 0.0) {
            ctx.cct_evict_keep.fetch_add(1, std::memory_order_relaxed);
        }
        if (eam_replace && !eam_replace_stats_ready) {
            eam_replace_stats = moe_eam_replace_build_stats_locked(ctx);
            eam_replace_stats_ready = true;
        }
        const double layer_replace_mass = eam_replace && m->layer >= 0 && m->layer < eam_replace_stats.n_layers ?
            eam_replace_stats.layer_mass[(size_t) m->layer] : 0.0;
        const double eam_replace_pred = eam_replace ?
            moe_group_eam_replace_pred_from_stats_locked(ctx, eam_replace_stats, g) : 0.0;
        const double eam_replace_keep = eam_replace ?
            moe_group_eam_replace_keep_from_stats_locked(ctx, eam_replace_stats, g) : 0.0;
        if (eam_replace) {
            g.eam_replace_pred = eam_replace_pred;
            g.eam_replace_mass = layer_replace_mass;
            g.eam_replace_keep = eam_replace_keep;
        }
        const double score = eam_replace ?
            (-eam_replace_keep * eam_replace_score_scale +
                size_bonus - cct_keep_penalty - bad_reload_penalty) :
            (speculative_bonus + low_score_bonus + reuse_low_bonus + dist_score +
                size_bonus + bit_bonus + layer_window_bonus -
                recent_penalty - spec_guard_penalty - high_penalty - early_penalty -
                reuse_keep_penalty - reuse_protect_penalty -
                layer_window_penalty - layer_feedback_penalty - persistent_penalty -
                cct_keep_penalty - bad_reload_penalty);
        victim_candidate candidate;
        candidate.layer = m->layer;
        candidate.expert = e;
        candidate.score = score;
        candidate.hot_score = g.hot_score;
        candidate.lru_score = group_age == UINT64_MAX ? 1.0e9 : (double) group_age;
        candidate.recency_score = token_age == UINT64_MAX ? 1.0e9 : (double) token_age;
        candidate.cache_policy_score = -g.hot_score;
        candidate.bad_reload_policy_score = -bad_reload_effective;
        candidate.next_use_score = dist == UINT64_MAX ? 1.0e9 : (double) dist;
        candidate.layer_score = (double) layer_dist;
        candidate.reuse_score = -reuse_keep;
        candidate.cct_score = -(double) cct_conf;
        candidate.age = group_age;
        candidate.belady = dist != UINT64_MAX;
        candidate.reason = eam_replace ? victim_candidate::EAM_REPLACE :
            layer_reserve_needed && !layer_window ? victim_candidate::LAYER_WINDOW :
            spec_evictable ? victim_candidate::SPEC_UNUSED :
            (reuse_low_bonus > 0.0 && reuse_keep < reuse_evict_ceil ?
                victim_candidate::REUSE_LOW :
            (dist == UINT64_MAX || dist > (uint64_t) std::max(1, ctx.params.active_window) ?
                victim_candidate::FAR_FUTURE : victim_candidate::LOW_SCORE));
        double forced_score = 0.0;
        if (forced_policy_score(candidate, &forced_score)) {
            candidate.score = forced_score;
        }
        olecar_consider(candidate);
        if (target_release_bytes > 0) {
            normal_candidates.push_back(candidate);
            scan_candidate_bytes += bytes;
        }
        if (better_candidate(candidate, best)) {
            best = candidate;
        }
        if (ctx.profile) {
            prof_candidate_us += (moe_profile_now_ns() - prof_candidate_t0) / 1000;
        }
    }
    };
    scan_pass(evict_scan_items);
    const bool limited_no_candidate =
        (sampled_scan || cold_window_scan) &&
        ((target_release_bytes > 0 && normal_candidates.empty() && temporary_candidates.empty()) ||
         (sample_fallback_bytes && target_release_bytes > 0 &&
             scan_candidate_bytes < target_release_bytes) ||
         (target_release_bytes == 0 && best.layer < 0 && temporary_best.layer < 0));
    if (limited_no_candidate) {
        if (sampled_scan) {
            ctx.evict_sample_fallback.fetch_add(1, std::memory_order_relaxed);
        }
        if (cold_window_scan) {
            ctx.evict_cold_window_fallback.fetch_add(1, std::memory_order_relaxed);
        }
        scan_pass(all_scan_items);
    }
    if (ctx.profile) {
        ctx.prof_evict_scan_us.fetch_add((moe_profile_now_ns() - prof_scan_t0) / 1000,
                std::memory_order_relaxed);
        ctx.prof_evict_candidate_us.fetch_add(prof_candidate_us, std::memory_order_relaxed);
    }

    ctx.evict_scan_calls.fetch_add(1, std::memory_order_relaxed);
    ctx.evict_scan_entries.fetch_add(scan_entries, std::memory_order_relaxed);
    ctx.evict_scan_unique.fetch_add(scan_unique, std::memory_order_relaxed);
    ctx.evict_scan_duplicates.fetch_add(scan_duplicates, std::memory_order_relaxed);
    ctx.evict_scan_absolute.fetch_add(scan_absolute, std::memory_order_relaxed);
    ctx.evict_scan_temporary.fetch_add(scan_temporary, std::memory_order_relaxed);
    ctx.evict_scan_normal.fetch_add(scan_normal, std::memory_order_relaxed);

    auto release_candidate = [&](const victim_candidate & candidate) {
        const uint64_t release_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const size_t released = moe_release_group_locked(ctx, candidate.layer, candidate.expert);
        if (released == 0) {
            return (size_t) 0;
        }
        if (ctx.admission_floor_layer == candidate.layer &&
                ctx.admission_floor_expert == candidate.expert) {
            ctx.admission_floor_valid = false;
        }
        if (candidate.belady) {
            ctx.belady_evictions.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx.lru_fallback_evictions.fetch_add(1, std::memory_order_relaxed);
        }
        const char * evict_reason = "unknown";
        uint8_t evict_reason_code = MOE_EVICT_REASON_UNKNOWN;
        switch (candidate.reason) {
            case victim_candidate::SPEC_UNUSED:
                evict_reason = "spec_unused";
                evict_reason_code = MOE_EVICT_REASON_SPEC_UNUSED;
                ctx.eam_evict_spec_unused.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::LOW_SCORE:
                evict_reason = "low_score";
                evict_reason_code = MOE_EVICT_REASON_LOW_SCORE;
                ctx.eam_evict_low_score.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::FAR_FUTURE:
                evict_reason = "far_future";
                evict_reason_code = MOE_EVICT_REASON_FAR_FUTURE;
                ctx.eam_evict_far_future.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::LAYER_WINDOW:
                evict_reason = "layer_window";
                evict_reason_code = MOE_EVICT_REASON_LAYER_WINDOW;
                ctx.layer_reserve_evictions.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::REUSE_LOW:
                evict_reason = "reuse_low";
                evict_reason_code = MOE_EVICT_REASON_REUSE_LOW;
                ctx.reuse_evict_low.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::EAM_REPLACE:
                evict_reason = "eam_replace";
                evict_reason_code = MOE_EVICT_REASON_EAM_REPLACE;
                ctx.eam_evict_low_score.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::RELAXED:
                evict_reason = "relaxed";
                evict_reason_code = MOE_EVICT_REASON_RELAXED;
                if (layer_reserve_needed) {
                    ctx.layer_reserve_relaxed.fetch_add(1, std::memory_order_relaxed);
                }
                ctx.eam_evict_lru_relaxed.fetch_add(1, std::memory_order_relaxed);
                break;
        }
        moe_group_get(ctx, candidate.layer, candidate.expert).last_evicted_reason = evict_reason_code;
        moe_evict_trace_write_locked(ctx, "evict", candidate.layer, candidate.expert,
                evict_reason, 0, released, candidate.score);
        if (ctx.profile) {
            prof_release_us += (moe_profile_now_ns() - release_t0) / 1000;
        }
        return released;
    };

    auto write_olecar_recommendations = [&](const std::vector<victim_candidate> & selected) {
        if (ctx.olecar_trace == nullptr) {
            return;
        }
        const uint64_t trace_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const uint64_t decision_id = ctx.next_olecar_decision_id++;
        const int bucket = moe_olecar_budget_bucket(ctx);
        const double weight_sum = moe_olecar_weight_sum_locked(ctx, bucket);
        std::unordered_map<uint64_t, std::array<double, MOE_OLECAR_FAMILY_COUNT>> family_support;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.id < 0 || policy.id >= MOE_OLECAR_POLICY_COUNT) {
                continue;
            }
            const int family = moe_olecar_policy_family(policy.id);
            if (family < 0 || family >= MOE_OLECAR_FAMILY_COUNT) {
                continue;
            }
            const uint64_t key = moe_group_key(policy.candidate.layer, policy.candidate.expert);
            const double p = weight_sum > 0.0 ?
                ctx.olecar_weights[(size_t) bucket][(size_t) policy.id] / weight_sum : 0.0;
            auto & per_family = family_support[key];
            per_family[(size_t) family] = std::max(per_family[(size_t) family], p);
        }
        std::unordered_map<uint64_t, double> support_prob;
        for (const auto & kv : family_support) {
            double s = 0.0;
            for (double p : kv.second) {
                s += p;
            }
            support_prob[kv.first] = s;
        }
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has) {
                continue;
            }
            const victim_candidate & c = policy.candidate;
            bool selected_by_current = false;
            for (const victim_candidate & s : selected) {
                if (s.layer == c.layer && s.expert == c.expert) {
                    selected_by_current = true;
                    break;
                }
            }
            moe_olecar_record record;
            record.decision_id = decision_id;
            record.policy_id = policy.id;
            record.bucket_id = bucket;
            record.policy = policy.name;
            record.layer = c.layer;
            record.expert = c.expert;
            record.selected = selected_by_current;
            if (policy.id >= 0 && policy.id < MOE_OLECAR_POLICY_COUNT &&
                    bucket >= 0 && bucket < MOE_OLECAR_BUCKET_COUNT) {
                record.weight_before =
                    ctx.olecar_weights[(size_t) bucket][(size_t) policy.id];
                record.weight_after = record.weight_before;
                record.policy_prob = weight_sum > 0.0 ? record.weight_before / weight_sum : 0.0;
            }
            record.support_prob = support_prob[moe_group_key(c.layer, c.expert)];
            record.update_count = ctx.olecar_updates[(size_t) bucket];
            record.final_score = c.score;
            record.lru_score = c.lru_score;
            record.recency_score = c.recency_score;
            record.cache_score = c.cache_policy_score;
            record.bad_reload_score = c.bad_reload_policy_score;
            record.next_use_score = c.next_use_score;
            record.layer_score = c.layer_score;
            record.reuse_score = c.reuse_score;
            record.cct_score = c.cct_score;
            moe_olecar_begin_locked(ctx, record);
            if (selected_by_current) {
                record.id = 0;
                record.kind = "exp4";
                record.exp4_update = true;
                moe_olecar_begin_locked(ctx, record);
            }
        }
        if (ctx.profile) {
            prof_trace_us += (moe_profile_now_ns() - trace_t0) / 1000;
        }
    };

    auto finish_profile = [&]() {
        if (!ctx.profile) {
            return;
        }
        ctx.prof_evict_sort_us.fetch_add(prof_sort_us, std::memory_order_relaxed);
        ctx.prof_evict_online_us.fetch_add(prof_online_us, std::memory_order_relaxed);
        ctx.prof_evict_admission_us.fetch_add(prof_admission_us, std::memory_order_relaxed);
        ctx.prof_evict_trace_us.fetch_add(prof_trace_us, std::memory_order_relaxed);
        ctx.prof_evict_release_us.fetch_add(prof_release_us, std::memory_order_relaxed);
        ctx.prof_evict_resident_queries.fetch_add(prof_resident_queries, std::memory_order_relaxed);
        ctx.prof_victim_select_us.fetch_add(
                (moe_profile_now_ns() - prof_t0) / 1000,
                std::memory_order_relaxed);
    };

    auto olecar_online_explore = [&]() {
        const double epsilon = std::max(0.0, std::min(1.0,
                moe_env_f64("LLAMA_LAZY_MOE_OLECAR_EPSILON", 0.0)));
        if (epsilon <= 0.0) {
            return false;
        }
        uint64_t x = ctx.next_olecar_decision_id * 0x9e3779b97f4a7c15ull;
        x ^= ctx.profile_token_epoch + 0xbf58476d1ce4e5b9ull + (x << 6) + (x >> 2);
        x ^= ctx.exec_epoch + 0x94d049bb133111ebull + (x << 6) + (x >> 2);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        const double u = (double) (x >> 11) * (1.0 / 9007199254740992.0);
        return u < epsilon;
    };

    auto olecar_online_support = [&]() {
        std::unordered_map<uint64_t, double> support;
        if (!olecar_online || explicit_force_policy || olecar_online_explore()) {
            return support;
        }
        const int bucket = moe_olecar_budget_bucket(ctx);
        const double weight_sum = moe_olecar_weight_sum_locked(ctx, bucket);
        if (weight_sum <= 0.0) {
            return support;
        }
        std::unordered_map<uint64_t, std::array<double, MOE_OLECAR_FAMILY_COUNT>> family_support;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.id < 0 || policy.id >= MOE_OLECAR_POLICY_COUNT ||
                    policy.candidate.layer < 0 || policy.candidate.expert < 0) {
                continue;
            }
            const int family = moe_olecar_policy_family(policy.id);
            if (family < 0 || family >= MOE_OLECAR_FAMILY_COUNT) {
                continue;
            }
            const double p =
                ctx.olecar_weights[(size_t) bucket][(size_t) policy.id] / weight_sum;
            auto & per_family =
                family_support[moe_group_key(policy.candidate.layer, policy.candidate.expert)];
            per_family[(size_t) family] = std::max(per_family[(size_t) family], p);
        }
        for (const auto & kv : family_support) {
            double s = 0.0;
            for (double p : kv.second) {
                s += p;
            }
            support[kv.first] = s;
        }
        return support;
    };

    auto reorder_olecar_online = [&](std::vector<victim_candidate> & candidates) {
        if (candidates.size() < 2) {
            return;
        }
        const std::unordered_map<uint64_t, double> support = olecar_online_support();
        if (support.empty()) {
            return;
        }
        auto score_of = [&](const victim_candidate & c) {
            auto it = support.find(moe_group_key(c.layer, c.expert));
            return it == support.end() ? 0.0 : it->second;
        };
        const int topk_env = moe_env_i32("LLAMA_LAZY_MOE_OLECAR_ONLINE_TOPK", 32);
        const size_t topk = topk_env <= 0 ? candidates.size() :
            std::min(candidates.size(), (size_t) topk_env);
        std::stable_sort(candidates.begin(), candidates.begin() + topk,
                [&](const victim_candidate & a, const victim_candidate & b) {
                    const double sa = score_of(a);
                    const double sb = score_of(b);
                    if (sa != sb) {
                        return sa > sb;
                    }
                    return better_candidate(a, b);
                });
    };

    auto choose_olecar_online_best = [&](const victim_candidate & fallback) {
        const std::unordered_map<uint64_t, double> support = olecar_online_support();
        if (support.empty()) {
            return fallback;
        }
        victim_candidate best_online = fallback;
        double best_support = -1.0;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.candidate.layer < 0 || policy.candidate.expert < 0) {
                continue;
            }
            const victim_candidate & c = policy.candidate;
            if (moe_group_resident_bytes(ctx, c.layer, c.expert) == 0) {
                continue;
            }
            auto it = support.find(moe_group_key(c.layer, c.expert));
            const double s = it == support.end() ? 0.0 : it->second;
            if (s <= 0.0) {
                continue;
            }
            const double max_score_drop =
                moe_env_f64("LLAMA_LAZY_MOE_OLECAR_ONLINE_MAX_SCORE_DROP", 1.0e300);
            if (c.score < fallback.score - max_score_drop) {
                continue;
            }
            if (s > best_support ||
                    (s == best_support && better_candidate(c, best_online))) {
                best_support = s;
                best_online = c;
            }
        }
        return best_support > 0.0 ? best_online : fallback;
    };

    if (target_release_bytes > 0) {
        uint64_t phase_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        std::sort(normal_candidates.begin(), normal_candidates.end(), better_candidate);
        std::sort(temporary_candidates.begin(), temporary_candidates.end(), better_candidate);
        if (ctx.profile) {
            prof_sort_us += (moe_profile_now_ns() - phase_t0) / 1000;
            phase_t0 = moe_profile_now_ns();
        }
        reorder_olecar_online(normal_candidates);
        reorder_olecar_online(temporary_candidates);
        if (ctx.profile) {
            prof_online_us += (moe_profile_now_ns() - phase_t0) / 1000;
        }

        if (admission_batch != nullptr && batch_results != nullptr) {
            const uint64_t admission_t0 = ctx.profile ? moe_profile_now_ns() : 0;
            batch_results->assign(admission_batch->size(), {});
            std::vector<size_t> order(admission_batch->size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                const moe_demand_admission_request & ra = (*admission_batch)[a];
                const moe_demand_admission_request & rb = (*admission_batch)[b];
                if (ra.incoming_value != rb.incoming_value) {
                    return ra.incoming_value > rb.incoming_value;
                }
                if (ra.layer != rb.layer) {
                    return ra.layer < rb.layer;
                }
                return ra.expert < rb.expert;
            });

            std::vector<victim_candidate> victim_pool;
            victim_pool.reserve(normal_candidates.size() + temporary_candidates.size());
            victim_pool.insert(victim_pool.end(), normal_candidates.begin(), normal_candidates.end());
            victim_pool.insert(victim_pool.end(), temporary_candidates.begin(), temporary_candidates.end());

            std::vector<victim_candidate> selected;
            selected.reserve(victim_pool.size());
            size_t victim_cursor = 0;
            size_t cache_credit = batch_free_bytes;
            size_t staging_credit = batch_staging_bytes;

            for (size_t request_index : order) {
                const moe_demand_admission_request & request =
                    (*admission_batch)[request_index];
                moe_demand_admission_result & result =
                    (*batch_results)[request_index];
                if (request.target_bytes == 0) {
                    continue;
                }
                if (cache_credit >= request.target_bytes) {
                    cache_credit -= request.target_bytes;
                    continue;
                }

                const size_t required = request.target_bytes - cache_credit;
                size_t peek_cursor = victim_cursor;
                size_t valued_bytes = 0;
                size_t full_victim_bytes = 0;
                double weighted_value = 0.0;
                while (peek_cursor < victim_pool.size() && valued_bytes < required) {
                    const victim_candidate & candidate = victim_pool[peek_cursor++];
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0) {
                        continue;
                    }
                    moe_group_state & victim =
                        moe_group_get(ctx, candidate.layer, candidate.expert);
                    const double cache_score =
                        moe_group_cache_score(ctx, victim, resident);
                    const double value = moe_prefetch_victim_value_locked(
                            ctx, victim, resident, cache_score);
                    const size_t take = std::min(resident, required - valued_bytes);
                    weighted_value += value * (double) take / (double) required;
                    valued_bytes += take;
                    full_victim_bytes += resident;
                    if (!result.has_victim) {
                        result.has_victim = true;
                        result.victim_layer = candidate.layer;
                        result.victim_expert = candidate.expert;
                    }
                    result.victim_groups++;
                    result.victim_keys.push_back(
                            moe_group_key(candidate.layer, candidate.expert));
                }
                result.evaluated = true;
                result.victim_bytes = full_victim_bytes;
                result.victim_value = weighted_value;

                const bool staging_available =
                    staging_credit >= request.target_bytes;
                result.bypass = staging_available &&
                    (!result.has_victim ||
                     request.incoming_value <= result.victim_value + request.margin);
                if (result.bypass) {
                    staging_credit -= request.target_bytes;
                    continue;
                }

                cache_credit = 0;
                while (victim_cursor < peek_cursor) {
                    const victim_candidate & candidate = victim_pool[victim_cursor++];
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0) {
                        continue;
                    }
                    selected.push_back(candidate);
                    cache_credit += resident;
                }
                cache_credit = cache_credit > required ?
                    cache_credit - required : 0;
            }
            if (ctx.profile) {
                prof_admission_us += (moe_profile_now_ns() - admission_t0) / 1000;
            }

            size_t released_total = 0;
            write_olecar_recommendations(selected);
            for (const victim_candidate & candidate : selected) {
                const size_t released = release_candidate(candidate);
                if (released > 0) {
                    released_total += released;
                    if (candidate.reason == victim_candidate::RELAXED) {
                        ctx.evict_scan_selected_temporary.fetch_add(
                                1, std::memory_order_relaxed);
                    }
                }
            }
            finish_profile();
            return released_total > 0;
        }

        std::vector<victim_candidate> selected;
        const uint64_t admission_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        selected.reserve(normal_candidates.size() + temporary_candidates.size());
        size_t selected_bytes = 0;
        auto select_candidates = [&](const std::vector<victim_candidate> & candidates) {
            for (const victim_candidate & candidate : candidates) {
                ++prof_resident_queries;
                const size_t bytes =
                    moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                if (bytes == 0) {
                    continue;
                }
                selected.push_back(candidate);
                selected_bytes += bytes;
                if (selected_bytes >= target_release_bytes) {
                    break;
                }
            }
        };
        select_candidates(normal_candidates);
        if (selected_bytes < target_release_bytes) {
            select_candidates(temporary_candidates);
        }

        if (admission_request != nullptr && admission_result != nullptr) {
            admission_result->evaluated = true;
            admission_result->victim_groups = (int) selected.size();
            admission_result->victim_bytes = selected_bytes;
            for (const victim_candidate & candidate : selected) {
                admission_result->victim_keys.push_back(
                        moe_group_key(candidate.layer, candidate.expert));
            }
            if (!selected.empty()) {
                admission_result->has_victim = true;
                admission_result->victim_layer = selected.front().layer;
                admission_result->victim_expert = selected.front().expert;
                size_t valued_bytes = 0;
                double weighted_value = 0.0;
                for (const victim_candidate & candidate : selected) {
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0 || valued_bytes >= target_release_bytes) {
                        continue;
                    }
                    moe_group_state & victim =
                        moe_group_get(ctx, candidate.layer, candidate.expert);
                    const double cache_score =
                        moe_group_cache_score(ctx, victim, resident);
                    const double value = moe_prefetch_victim_value_locked(
                            ctx, victim, resident, cache_score);
                    const size_t take =
                        std::min(resident, target_release_bytes - valued_bytes);
                    weighted_value += value * (double) take /
                        (double) target_release_bytes;
                    valued_bytes += take;
                }
                admission_result->victim_value = weighted_value;
            }
            admission_result->bypass =
                !admission_result->has_victim ||
                admission_request->incoming_value <=
                    admission_result->victim_value + admission_request->margin;
            if (admission_result->bypass) {
                if (!admission_result->has_victim) {
                    ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
                }
                finish_profile();
                return false;
            }
        }
        if (ctx.profile) {
            prof_admission_us += (moe_profile_now_ns() - admission_t0) / 1000;
        }

        size_t released_total = 0;
        write_olecar_recommendations(selected);
        for (const victim_candidate & candidate : selected) {
            const size_t released = release_candidate(candidate);
            if (released > 0) {
                released_total += released;
                if (candidate.reason == victim_candidate::RELAXED) {
                    ctx.evict_scan_selected_temporary.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (released_total == 0) {
            ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
        }
        finish_profile();
        return released_total > 0;
    }

    if (best.layer < 0 && temporary_best.layer >= 0) {
        best = temporary_best;
        ctx.evict_scan_selected_temporary.fetch_add(1, std::memory_order_relaxed);
    }
    if (best.layer < 0) {
        ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
        finish_profile();
        return false;
    }

    best = choose_olecar_online_best(best);
    write_olecar_recommendations(std::vector<victim_candidate>{best});
    const bool released = release_candidate(best) > 0;
    finish_profile();
    return released;
}
```

### prefetch enqueue：moe_enqueue_group_prefetch_task()

来源：`src/llama-moe-buffer.cpp:9369-9406`

```cpp
static bool moe_enqueue_group_prefetch_task(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert,
        int                        rank,
        float                      score,
        int                        priority,
        uint64_t                   token_epoch,
        int                        skip_kind) {
    if (layer < 0 || expert < 0) {
        return false;
    }
    moe_prefetch_task task;
    task.is_group = true;
    task.layer = layer;
    task.skip_kind = skip_kind;
    task.e = expert;
    task.rank = rank;
    task.score = score;
    task.target_bits = ctx.params.base_bits;
    task.priority = priority;
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        const uint64_t key = moe_group_key(layer, expert);
        auto last_it = ctx.group_prefetch_last_enqueue_token.find(key);
        if (last_it != ctx.group_prefetch_last_enqueue_token.end() && last_it->second == token_epoch) {
            ctx.group_prefetch_dups.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        ctx.group_prefetch_last_enqueue_token[key] = token_epoch;
        task.seq = ctx.next_task_seq++;
        ctx.queue.push(task);
    }
    ctx.enqueued.fetch_add(1, std::memory_order_relaxed);
    ctx.group_prefetch_enqueued.fetch_add(1, std::memory_order_relaxed);
    ctx.cv_q.notify_one();
    return true;
}
```

### prefetch API：llama_moe_buffer_prefetch()

来源：`src/llama-moe-buffer.cpp:9697-9703`

```cpp
void llama_moe_buffer_prefetch(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        int                        n_experts) {
    llama_moe_buffer_prefetch_ranked(ctx, layer, experts, nullptr, n_experts);
}
```

### ranked prefetch API：llama_moe_buffer_prefetch_ranked()

来源：`src/llama-moe-buffer.cpp:9705-9746`

```cpp
void llama_moe_buffer_prefetch_ranked(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        const float *              scores,
        int                        n_experts) {
    if (ctx == nullptr || !ctx->params.enabled || experts == nullptr || n_experts <= 0) {
        return;
    }
    const uint64_t submit_t0 = ctx->profile ? moe_profile_now_ns() : 0;
    auto it = ctx->by_layer.find(layer);
    if (it == ctx->by_layer.end()) {
        return;
    }
    int n_enqueued = 0;
    {
        std::lock_guard<std::mutex> lk(ctx->qmtx);
        for (moe_managed * m : it->second) {
            for (int i = 0; i < n_experts; ++i) {
                const int e = experts[i];
                const float score = scores != nullptr ? scores[i] : 0.0f;
                if (ctx->prefetch_budget_enabled) {
                    const uint64_t charge = (uint64_t) m->stride;
                    if (charge > ctx->prefetch_budget_available_bytes) {
                        ctx->eam_prefetch_drop_budget.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    ctx->prefetch_budget_available_bytes -= charge;
                }
                n_enqueued += moe_enqueue_prefetch(*ctx, *m, e, i, score, true) ? 1 : 0;
            }
        }
    }
    if (n_enqueued > 1) {
        ctx->cv_q.notify_all();
    } else if (n_enqueued == 1) {
        ctx->cv_q.notify_one();
    }
    if (ctx->profile) {
        ctx->prof_sidecar_submit_us.fetch_add((moe_profile_now_ns() - submit_t0) / 1000, std::memory_order_relaxed);
    }
}
```

### prefetch budget API：llama_moe_buffer_set_prefetch_budget()

来源：`src/llama-moe-buffer.cpp:9748-9758`

```cpp
void llama_moe_buffer_set_prefetch_budget(
        llama_moe_buffer_context & ctx,
        uint64_t                  budget_bytes) {
    if (!llama_moe_buffer_enabled(&ctx)) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx.qmtx);
    ctx.prefetch_budget_enabled = budget_bytes > 0;
    ctx.prefetch_budget_bytes = budget_bytes;
    ctx.prefetch_budget_available_bytes = budget_bytes;
}
```

### compute stream callback：llama_moe_buffer_stream_callback()

来源：`src/llama-moe-buffer.cpp:9760-10077`

```cpp
bool llama_moe_buffer_stream_callback(ggml_tensor * op, int ith, void * user_data) {
    auto * ctx = static_cast<llama_moe_buffer_context *>(user_data);
    if (ctx == nullptr || op == nullptr || op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    ggml_tensor * exps = op->src[0];
    if (exps == nullptr) return false;
    auto it = ctx->by_name.find(ggml_get_name(exps));
    if (it == ctx->by_name.end()) {
        return false;  // not a managed expert GEMM
    }
    moe_managed & m = it->second;
    const uint64_t callback_t0 = (ith == 0 && ctx->profile) ? moe_profile_now_ns() : 0;

    if (ith == 0) {
        uint64_t token_epoch_snapshot = 0;
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (m.layer >= 0) {
                const int completed_layer = ctx->profile_last_layer;
                if (completed_layer >= 0 && completed_layer != m.layer) {
                    moe_release_demand_bypass_layer_locked(*ctx, completed_layer);
                    moe_evict_layer_done_locked(*ctx, completed_layer);
                }
                if (ctx->profile_last_layer >= 0 && m.layer < ctx->profile_last_layer) {
                    ++ctx->profile_token_epoch;
                    moe_admission_outcome_expire_locked(*ctx);
                    moe_admission_regret_expire_locked(*ctx);
                }
                ctx->profile_last_layer = m.layer;
            }
            ++ctx->exec_epoch;
            token_epoch_snapshot = ctx->profile_token_epoch;
        }
        moe_cleanup_mwq_transients(*ctx);

        ggml_tensor * ids = op->src[2];  // selected experts, I32
        const bool use_mwq_kernel = moe_op_all_mwq_available(*ctx, m, ids);
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (!use_mwq_kernel && !moe_ensure_buf(m)) {
                return false;
            }
            m.tensor = exps;
        }
        // Repoint only for the fallback original-layout kernel. The MWQ override
        // reads the managed MWQ buffers directly.
        if (!use_mwq_kernel && exps->data != m.buf) {
            exps->data = m.buf;
        }
        if (ids != nullptr && ids->data != nullptr && ids->type == GGML_TYPE_I32) {
            std::vector<int> op_target_bits;
            std::vector<int> op_counts;
            std::vector<int> op_min_rank;
            if (ctx->params.dynamic_bits_real) {
                const uint64_t route_t0 = ctx->profile ? moe_profile_now_ns() : 0;
                op_target_bits.assign(m.n_expert, 0);
                op_counts.assign(m.n_expert, 0);
                op_min_rank.assign(m.n_expert, 999999);
                for (int64_t token = 0; token < ids->ne[1]; ++token) {
                    for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                        if (expert < 0 || expert >= m.n_expert) {
                            continue;
                        }
                        const int64_t flat = token * ids->ne[0] + rank;
                        const int target_bits = moe_target_bits_for_id_index(*ctx, m, expert, flat);
                        op_target_bits[expert] = std::max(op_target_bits[expert], target_bits);
                        op_counts[expert]++;
                        op_min_rank[expert] = std::min(op_min_rank[expert], (int) rank);
                    }
                }
                if (ctx->profile) {
                    ctx->prof_route_us.fetch_add((moe_profile_now_ns() - route_t0) / 1000, std::memory_order_relaxed);
                }

                if (moe_env_flag("LLAMA_LAZY_MOE_GROUP_PREFETCH_SELFTEST", 0) &&
                        !ctx->group_prefetch_selftest_done && m.layer >= 0) {
                    for (int expert = 0; expert < m.n_expert; ++expert) {
                        if (op_counts[expert] > 0) {
                            ctx->group_prefetch_selftest_done = true;
                            const int rank = op_min_rank[expert] == 999999 ? 0 : op_min_rank[expert];
                            moe_enqueue_group_prefetch_task(*ctx, m.layer, expert, rank, 1.0f, INT_MAX / 2,
                                    token_epoch_snapshot, -1);
                            break;
                        }
                    }
                }

                struct route_run {
                    int expert = -1;
                    int target_bits = 0;
                    int best_rank = 0;
                    int count = 0;
                    int state = 2; // 0=ready, 1=loading/queued, 2=cold
                };
                auto classify_run_state = [&](int expert, int target_bits) {
                    std::lock_guard<std::mutex> lk(ctx->mtx);
                    if (ctx->params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(m, expert, target_bits) > 0) {
                        return 0;
                    }
                    if (moe_native_resident_compatible(*ctx, m, expert, target_bits)) {
                        return 0;
                    }
                    if (expert >= 0 && expert < m.n_expert &&
                            (m.resident[expert] == ST_INFLIGHT ||
                             (expert < (int) m.queued.size() && m.queued[expert]))) {
                        return 1;
                    }
                    return 2;
                };

                std::vector<route_run> runs;
                runs.reserve((size_t) m.n_expert);
                for (int expert = 0; expert < m.n_expert; ++expert) {
                    if (op_target_bits[expert] <= 0) {
                        continue;
                    }
                    route_run run;
                    run.expert = expert;
                    run.target_bits = op_target_bits[expert];
                    run.best_rank = op_min_rank[expert] == 999999 ? 0 : op_min_rank[expert];
                    run.count = op_counts[expert];
                    run.state = classify_run_state(expert, run.target_bits);
                    runs.push_back(run);
                }
                if (moe_env_flag("LLAMA_LAZY_MOE_RUN_QUEUE", 1)) {
                    std::stable_sort(runs.begin(), runs.end(), [](const route_run & a, const route_run & b) {
                        if (a.state != b.state) {
                            return a.state < b.state;
                        }
                        if (a.count != b.count) {
                            return a.count > b.count;
                        }
                        return a.expert < b.expert;
                    });
                }
                if (!runs.empty()) {
                    ctx->runq_ops.fetch_add(1, std::memory_order_relaxed);
                }
                if (moe_kind(m) == moe_tensor_kind::gate) {
                    moe_prepare_demand_groups(*ctx, m.layer, ids);
                }
                if (moe_kind(m) == moe_tensor_kind::gate && moe_group_prefetch_same_layer_enabled()) {
                    const int max_groups = moe_group_prefetch_same_layer_max_groups();
                    const int min_count = moe_group_prefetch_same_layer_min_count();
                    struct group_prefetch_candidate {
                        route_run run;
                        int missing_siblings = 0;
                    };
                    std::vector<group_prefetch_candidate> candidates;
                    candidates.reserve(runs.size());
                    {
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        for (const route_run & run : runs) {
                            if (run.count < min_count) {
                                continue;
                            }
                            const int missing = moe_group_missing_sibling_slices_locked(
                                    *ctx, m.layer, run.expert, run.best_rank, (int) moe_tensor_kind::gate);
                            if (missing <= 0) {
                                continue;
                            }
                            candidates.push_back({run, missing});
                        }
                    }
                    std::stable_sort(candidates.begin(), candidates.end(),
                            [](const group_prefetch_candidate & a, const group_prefetch_candidate & b) {
                        if (a.missing_siblings != b.missing_siblings) {
                            return a.missing_siblings > b.missing_siblings;
                        }
                        if (a.run.count != b.run.count) {
                            return a.run.count > b.run.count;
                        }
                        return a.run.expert < b.run.expert;
                    });
                    int submitted = 0;
                    for (const group_prefetch_candidate & cand : candidates) {
                        if (max_groups > 0 && submitted >= max_groups) {
                            break;
                        }
                        const route_run & run = cand.run;
                        const float score = (float) run.count;
                        const int priority = INT_MAX / 4 + cand.missing_siblings * 2048 + std::min(run.count, 1024);
                        if (moe_enqueue_group_prefetch_task(*ctx, m.layer, run.expert, run.best_rank, score, priority,
                                    token_epoch_snapshot, (int) moe_tensor_kind::gate)) {
                            ++submitted;
                        }
                    }
                    if (submitted > 0) {
                        ctx->group_prefetch_same_layer_ops.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                int run_order = 0;
                for (const route_run & run : runs) {
                    const int expert = run.expert;
                    const int target_bits = run.target_bits;
                    const int best_rank = run.best_rank;
                    if (run.state == 0) {
                        ctx->runq_ready.fetch_add(1, std::memory_order_relaxed);
                    } else if (run.state == 1) {
                        ctx->runq_loading.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        ctx->runq_cold.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (ctx->run_queue_trace != nullptr) {
                        const moe_tensor_kind kind = moe_kind(m);
                        const char * kind_name = kind == moe_tensor_kind::gate ? "gate" :
                            (kind == moe_tensor_kind::up ? "up" :
                             (kind == moe_tensor_kind::down ? "down" : "other"));
                        std::fprintf(ctx->run_queue_trace,
                                "%llu\t%llu\t%d\t%d\t%s\t%d\t%d\t%d\t%s\t%d\n",
                                (unsigned long long) ctx->profile_token_epoch,
                                (unsigned long long) ctx->exec_epoch,
                                m.layer,
                                expert,
                                kind_name,
                                target_bits,
                                best_rank,
                                run.count,
                                run.state == 0 ? "ready" : (run.state == 1 ? "loading" : "cold"),
                                run_order);
                    }
                    ++run_order;
                    if (use_mwq_kernel) {
                        moe_stream_mwq_slice(*ctx, m, expert, true, target_bits, best_rank);
                    } else {
                        moe_stream_slice(*ctx, m, expert, true, target_bits, best_rank);
                    }
                    const int extra = run.count - 1;
                    if (extra > 0) {
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        const uint64_t add = (uint64_t) extra;
                        const uint64_t room = UINT32_MAX - m.activation[expert];
                        const uint64_t inc = std::min(add, room);
                        m.activation[expert] += (uint32_t) inc;
                        m.total_act += inc;
                        moe_group_touch_locked(*ctx, m, expert, best_rank, inc);
                        ctx->hits.fetch_add(add, std::memory_order_relaxed);
                    }
                }
                if (moe_kind(m) == moe_tensor_kind::down) {
                    if (moe_next_token_keep_enabled()) {
                        std::vector<int> last_token_experts;
                        if (ids->ne[1] > 0) {
                            const int64_t token = ids->ne[1] - 1;
                            last_token_experts.reserve((size_t) ids->ne[0]);
                            for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                                const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                                if (expert >= 0 && expert < m.n_expert) {
                                    last_token_experts.push_back(expert);
                                }
                            }
                            std::sort(last_token_experts.begin(), last_token_experts.end());
                            last_token_experts.erase(std::unique(last_token_experts.begin(), last_token_experts.end()),
                                    last_token_experts.end());
                        }
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        moe_next_token_update_after_route_locked(*ctx, m.layer, last_token_experts, m.n_expert);
                    }
                    if (moe_eam_predict_enabled()) {
                        std::vector<int> actual_experts;
                        actual_experts.reserve((size_t) m.n_expert);
                        for (int expert = 0; expert < m.n_expert; ++expert) {
                            if (op_counts[expert] > 0) {
                                actual_experts.push_back(expert);
                            }
                        }
                        moe_eam_predict_after_route(*ctx, m.layer, actual_experts, m.n_expert);
                    }
                }
                if (ctx->profile) {
                    ctx->prof_moe_total_us.fetch_add((moe_profile_now_ns() - callback_t0) / 1000, std::memory_order_relaxed);
                }
                return true;
            }

            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                    const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                    const int64_t flat = token * ids->ne[0] + rank;
                    int target_bits = moe_target_bits_for_id_index(*ctx, m, expert, flat);
                    if (expert >= 0 && expert < m.n_expert && !op_target_bits.empty()) {
                        target_bits = std::max(target_bits, op_target_bits[expert]);
                    }
                    if (use_mwq_kernel) {
                        moe_stream_mwq_slice(*ctx, m, expert, true, target_bits, (int) rank);
                    } else {
                        moe_stream_slice(*ctx, m, expert, true, target_bits, (int) rank);  // resident or wait-for-inflight
                    }
                }
            }
            if (moe_kind(m) == moe_tensor_kind::down && moe_next_token_keep_enabled()) {
                std::vector<int> last_token_experts;
                if (ids->ne[1] > 0) {
                    const int64_t token = ids->ne[1] - 1;
                    last_token_experts.reserve((size_t) ids->ne[0]);
                    for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                        if (expert >= 0 && expert < m.n_expert) {
                            last_token_experts.push_back(expert);
                        }
                    }
                    std::sort(last_token_experts.begin(), last_token_experts.end());
                    last_token_experts.erase(std::unique(last_token_experts.begin(), last_token_experts.end()),
                            last_token_experts.end());
                }
                std::lock_guard<std::mutex> lk(ctx->mtx);
                moe_next_token_update_after_route_locked(*ctx, m.layer, last_token_experts, m.n_expert);
            }
        }
        if (ctx->profile) {
            ctx->prof_moe_total_us.fetch_add((moe_profile_now_ns() - callback_t0) / 1000, std::memory_order_relaxed);
        }
    }
    return true;  // managed → caller issues a barrier
}
```

### fused SWIGLU callback：llama_moe_buffer_swiglu_callback()

来源：`src/llama-moe-buffer.cpp:14670-14747`

```cpp
static bool llama_moe_buffer_swiglu_callback(llama_moe_buffer_context * ctx, ggml_tensor * op, int ith, int nth) {
    if (ctx == nullptr || op == nullptr || !ctx->params.fuse_swiglu || op->op != GGML_OP_GLU) {
        return false;
    }
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || op->op_params[1] != 0) {
        return false;
    }

    ggml_tensor * gate = op->src[0];
    ggml_tensor * up   = op->src[1];
    if (gate == nullptr || up == nullptr ||
            gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(gate, up) ||
            !ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up) || !ggml_is_contiguous_1(op)) {
        return false;
    }

    bool match = false;
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        match = gate == ctx->last_fused_gate_op && up == ctx->last_fused_up_op;
    }
    if (!match) {
        if (ith == 0 && moe_name_has(op, "ffn_moe_swiglu")) {
            ctx->fused_swiglu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    if (ctx->params.fuse_expert_ffn && ctx->params.fuse_direct_swiglu && moe_name_has(op, "ffn_moe_swiglu")) {
        return true;
    }

    if (ctx->params.fuse_direct_swiglu) {
        if (llama_moe_buffer_swiglu_direct_compute(ctx, op, gate, up, ith, nth)) {
            llama_moe_buffer_fused_ffn_oracle_after_swiglu(ctx, op, gate, up, ith, nth);
            if (ith == 0) {
                ctx->fused_swiglu_hits.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        if (ith == 0) {
            ctx->fused_direct_swiglu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        tls_mwq_keep_sum_cache = false;
        const bool ok_up = llama_moe_buffer_mul_mat_id_compute(up, ith, nth, ctx);
        tls_mwq_keep_sum_cache = true;
        const bool ok_gate = llama_moe_buffer_mul_mat_id_compute(gate, ith, nth, ctx);
        tls_mwq_keep_sum_cache = false;
        if (!ok_up || !ok_gate) {
            return false;
        }
    }

    const int64_t nc = gate->ne[0];
    const int64_t nr = ggml_nrows(gate);
    if (op->ne[0] != nc || ggml_nrows(op) != nr || nc > INT_MAX) {
        return false;
    }

    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = std::min<int64_t>(ir0 + dr, nr);
    for (int64_t row = ir0; row < ir1; ++row) {
        const float * gate_p = (const float *) ((const char *) gate->data + row * gate->nb[1]);
        const float * up_p   = (const float *) ((const char *) up->data   + row * up->nb[1]);
        float * dst_p        = (float *)       ((char *) op->data         + row * op->nb[1]);
        moe_vec_swiglu_f32((int) nc, dst_p, gate_p, up_p);
    }

    llama_moe_buffer_fused_ffn_oracle_after_swiglu(ctx, op, gate, up, ith, nth);

    if (ith == 0) {
        ctx->fused_swiglu_hits.fetch_add(1, std::memory_order_relaxed);
        ctx->fused_swiglu_rows.fetch_add((uint64_t) nr, std::memory_order_relaxed);
    }
    return true;
}
```

### op override callback：llama_moe_buffer_mul_mat_id_callback()

来源：`src/llama-moe-buffer.cpp:14749-14914`

```cpp
bool llama_moe_buffer_mul_mat_id_callback(ggml_tensor * op, int ith, int nth, void * user_data) {
    auto * ctx = static_cast<llama_moe_buffer_context *>(user_data);
    if (ctx == nullptr || op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_GLU) {
        return llama_moe_buffer_swiglu_callback(ctx, op, ith, nth);
    }
    if (op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr) {
        return false;
    }

    if (ctx->params.fuse_expert_ffn && moe_name_has(op->src[0], ".ffn_down_exps.weight")) {
        ggml_tensor * swiglu_op = op->src[1];
        ggml_tensor * gate_op = swiglu_op != nullptr ? swiglu_op->src[0] : nullptr;
        ggml_tensor * up_op   = swiglu_op != nullptr ? swiglu_op->src[1] : nullptr;
        if (llama_moe_buffer_fused_ffn_compute(ctx, op, swiglu_op, gate_op, up_op, ith, nth)) {
            return true;
        }
        if (ith == 0) {
            ctx->fused_ffn_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        if (swiglu_op != nullptr && swiglu_op->op == GGML_OP_GLU && gate_op != nullptr && up_op != nullptr &&
                !llama_moe_buffer_swiglu_materialize(ctx, swiglu_op, gate_op, up_op, ith, nth)) {
            return false;
        }
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    if (!ctx->params.fuse_gate_up) {
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    ggml_tensor * src0 = op->src[0];
    auto it = ctx->by_name.find(ggml_get_name(src0));
    if (it == ctx->by_name.end()) {
        return false;
    }
    moe_managed & m = it->second;

    const bool is_up = moe_name_has(src0, ".ffn_up_exps.weight");
    const bool is_gate = moe_name_has(src0, ".ffn_gate_exps.weight");
    if (!is_up && !is_gate) {
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    ggml_tensor * up_op = nullptr;
    ggml_tensor * gate_op = nullptr;
    const int layer = moe_layer_from_name(src0);

    if (ctx->params.fuse_direct_swiglu) {
        ggml_tensor * ids = op->src[2];
        const std::string sibling_name = moe_sibling_exps_name(ggml_get_name(src0), is_up);
        auto sibling_it = ctx->by_name.find(sibling_name);
        if (sibling_it == ctx->by_name.end() ||
                !moe_op_all_mwq_available(*ctx, m, ids) ||
                !moe_op_all_mwq_available(*ctx, sibling_it->second, ids)) {
            if (ith == 0) {
                ctx->fused_gate_up_misses.fetch_add(1, std::memory_order_relaxed);
            }
            return false;
        }
    }

    auto active_matches = [&]() {
        return ctx->fused_layer == layer &&
            (ctx->fused_up_op == op || ctx->fused_gate_op == op);
    };

    {
        std::unique_lock<std::mutex> lk(ctx->mtx);
        if (ith == 0) {
            if (is_up) {
                if (ctx->pending_gate_op != nullptr && ctx->pending_gate_layer == layer) {
                    ctx->fused_up_op = op;
                    ctx->fused_gate_op = ctx->pending_gate_op;
                    ctx->fused_layer = layer;
                    ctx->pending_gate_op = nullptr;
                    ctx->pending_gate_src1 = nullptr;
                    ctx->pending_gate_ids = nullptr;
                    ctx->pending_gate_layer = -1;
                    up_op = ctx->fused_up_op;
                    gate_op = ctx->fused_gate_op;
                    ctx->cv_fuse.notify_all();
                } else {
                    ctx->pending_up_op = op;
                    ctx->pending_up_src1 = op->src[1];
                    ctx->pending_up_ids = op->src[2];
                    ctx->pending_up_layer = layer;
                    ctx->fused_gate_up_deferred.fetch_add(1, std::memory_order_relaxed);
                    ctx->cv_fuse.notify_all();
                    return true;
                }
            } else {
                if (ctx->pending_up_op != nullptr && ctx->pending_up_layer == layer) {
                    ctx->fused_up_op = ctx->pending_up_op;
                    ctx->fused_gate_op = op;
                    ctx->fused_layer = layer;
                    ctx->pending_up_op = nullptr;
                    ctx->pending_up_src1 = nullptr;
                    ctx->pending_up_ids = nullptr;
                    ctx->pending_up_layer = -1;
                    up_op = ctx->fused_up_op;
                    gate_op = ctx->fused_gate_op;
                    ctx->cv_fuse.notify_all();
                } else {
                    ctx->pending_gate_op = op;
                    ctx->pending_gate_src1 = op->src[1];
                    ctx->pending_gate_ids = op->src[2];
                    ctx->pending_gate_layer = layer;
                    ctx->fused_gate_up_deferred.fetch_add(1, std::memory_order_relaxed);
                    ctx->cv_fuse.notify_all();
                    return true;
                }
            }
        } else {
            const bool second_half_wait =
                (is_up   && ctx->pending_gate_op != nullptr && ctx->pending_gate_layer == layer) ||
                (is_gate && ctx->pending_up_op   != nullptr && ctx->pending_up_layer   == layer);
            if (second_half_wait) {
                ctx->cv_fuse.wait_for(lk, std::chrono::milliseconds(100), active_matches);
            }
            if (active_matches()) {
                up_op = ctx->fused_up_op;
                gate_op = ctx->fused_gate_op;
            } else {
                return true;
            }
        }
    }

    if (ctx->params.fuse_direct_swiglu) {
        if (ith == 0) {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            ctx->last_fused_up_op = up_op;
            ctx->last_fused_gate_op = gate_op;
            ctx->fused_gate_up_hits.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    tls_mwq_keep_sum_cache = false;
    const bool ok_up = llama_moe_buffer_mul_mat_id_compute(up_op, ith, nth, user_data);
    tls_mwq_keep_sum_cache = true;
    const bool ok_gate = llama_moe_buffer_mul_mat_id_compute(gate_op, ith, nth, user_data);
    tls_mwq_keep_sum_cache = false;
    if (ith == 0 && ok_up && ok_gate) {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->last_fused_up_op = up_op;
        ctx->last_fused_gate_op = gate_op;
        ctx->fused_gate_up_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return ok_up && ok_gate;
}
```

### stats 导出：llama_moe_buffer_get_stats()

来源：`src/llama-moe-buffer.cpp:15549-15585`

```cpp
llama_moe_buffer_stats llama_moe_buffer_get_stats(llama_moe_buffer_context & ctx) {
    llama_moe_buffer_stats stats;
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        stats.resident_bytes = ctx.resident_bytes;
        stats.budget_bytes   = ctx.params.budget_bytes;
        stats.budget_unbounded = ctx.params.budget_unbounded;
        stats.planner_safe_budget_bytes = ctx.params.planner_safe_budget_bytes;
        stats.planner_floor_bytes = ctx.params.planner_floor_bytes;
        stats.expert_bytes   = ctx.expert_total;
    }
    stats.streams         = ctx.streams.load(std::memory_order_relaxed);
    stats.hits            = ctx.hits.load(std::memory_order_relaxed);
    stats.evictions       = ctx.evictions.load(std::memory_order_relaxed);
    stats.bytes_read      = ctx.bytes_read.load(std::memory_order_relaxed);
    stats.cache_hits      = ctx.cache_hits.load(std::memory_order_relaxed);
    stats.cache_misses    = ctx.cache_misses.load(std::memory_order_relaxed);
    stats.prefetch_hits   = ctx.eam_prefetch_hits.load(std::memory_order_relaxed);
    stats.prefetch_late   = ctx.eam_prefetch_late.load(std::memory_order_relaxed);
    stats.prefetch_unused = ctx.eam_prefetch_unused.load(std::memory_order_relaxed);
    stats.prefetch_budget_dropped =
        ctx.eam_prefetch_drop_budget.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        stats.prefetch_budget_bytes =
            (size_t) ctx.prefetch_budget_bytes;
        stats.prefetch_budget_available_bytes =
            (size_t) ctx.prefetch_budget_available_bytes;
    }
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        stats.warm_working_set_bytes = ctx.warm_working_set_bytes;
        stats.warm_working_set_groups = ctx.warm_working_set_groups;
        stats.warm_working_set_coverage = ctx.warm_working_set_coverage;
    }
    return stats;
}
```

### clean reclaim：llama_moe_buffer_reclaim_clean()

来源：`src/llama-moe-buffer.cpp:15587-15611`

```cpp
llama_moe_buffer_reclaim_result llama_moe_buffer_reclaim_clean(
        llama_moe_buffer_context & ctx,
        uint64_t                   target_bytes,
        uint32_t                   max_groups) {
    llama_moe_buffer_reclaim_result result;
    if (!llama_moe_buffer_enabled(&ctx) || target_bytes == 0 || max_groups == 0) {
        return result;
    }

    std::lock_guard<std::mutex> lk(ctx.mtx);
    while (result.released_bytes < target_bytes && result.released_groups < max_groups) {
        const size_t before = ctx.resident_bytes;
        if (before == 0 || !moe_evict_lru(ctx, target_bytes - result.released_bytes)) {
            break;
        }
        const size_t after = ctx.resident_bytes;
        if (before <= after) {
            break;
        }
        result.released_bytes += before - after;
        result.released_groups++;
    }
    result.target_satisfied = result.released_bytes >= target_bytes;
    return result;
}
```

## 16.3 Server Memory Governor / KV 协同完整关键代码

### async action kind 与队列状态

来源：`tools/server/server-context.cpp:807-858`

```cpp
    bool memory_governor_async_actions_enabled = true;
    uint32_t memory_governor_async_queue_max_depth = 0;
    enum class memory_governor_async_action_kind {
        dense_clean_reclaim,
        moe_clean_reclaim,
        kv_global_release,
        kv_sequence_offload,
        kv_sequence_slot_state_offload,
    };
    struct memory_governor_async_action {
        memory_governor_async_action_kind kind = memory_governor_async_action_kind::dense_clean_reclaim;
        int32_t seq_id = -1;
        int64_t score = 0;
        double roi = 0.0;
        uint64_t target_bytes = 0;
        uint64_t max_blocks = 0;
        uint32_t max_objects = 0;
        uint64_t sample_count = 0;
        uint64_t decision_id = 0;
        kv_pressure_state pressure_state = kv_pressure_state::NORMAL;
    };
    struct memory_governor_async_action_before {
        bool operator()(
                const memory_governor_async_action & lhs,
                const memory_governor_async_action & rhs) const {
            if (lhs.pressure_state != rhs.pressure_state) {
                return (int) lhs.pressure_state < (int) rhs.pressure_state;
            }
            if (lhs.roi != rhs.roi) return lhs.roi < rhs.roi;
            if (lhs.score != rhs.score) return lhs.score < rhs.score;
            if (lhs.target_bytes != rhs.target_bytes) return lhs.target_bytes < rhs.target_bytes;
            return lhs.decision_id > rhs.decision_id;
        }
    };
    std::mutex memory_governor_async_mtx;
    std::condition_variable memory_governor_async_cv;
    std::priority_queue<
        memory_governor_async_action,
        std::vector<memory_governor_async_action>,
        memory_governor_async_action_before> memory_governor_async_queue;
    std::thread memory_governor_async_worker;
    std::atomic<bool> memory_governor_async_stop { false };
    std::atomic<uint64_t> memory_governor_async_submitted { 0 };
    std::atomic<uint64_t> memory_governor_async_completed { 0 };
    std::atomic<uint64_t> memory_governor_async_rejected { 0 };
    std::atomic<uint64_t> memory_governor_async_dropped { 0 };
    std::atomic<uint64_t> memory_governor_async_relieved_pending_bytes { 0 };
    std::atomic<uint64_t> memory_governor_async_queue_depth { 0 };
    std::atomic<bool> memory_governor_async_dense_clean_inflight { false };
    std::atomic<bool> memory_governor_async_moe_clean_inflight { false };
    std::atomic<bool> memory_governor_async_kv_release_inflight { false };
    std::atomic<bool> memory_governor_async_kv_offload_inflight { false };
```

### slot-state offload fallback：memory_governor_slot_state_offload()

来源：`tools/server/server-context.cpp:974-1066`

```cpp
    memory_governor_slot_offload_result memory_governor_slot_state_offload(
            int32_t seq_id,
            uint64_t target_bytes) {
        memory_governor_slot_offload_result result;
        result.seq_id = seq_id;

        if (!ctx_tgt || !llama_get_memory(ctx_tgt)) {
            result.reason = "no_memory";
            return result;
        }
        if (!prompt_cache) {
            result.reason = "prompt_cache_unavailable";
            return result;
        }

        server_slot * slot = nullptr;
        for (server_slot & candidate : slots) {
            if (candidate.id == seq_id) {
                slot = &candidate;
                break;
            }
        }
        if (!slot) {
            result.reason = "slot_not_found";
            return result;
        }
        if (slot->is_processing() || slot->task != nullptr) {
            result.reason = "slot_active";
            return result;
        }
        if (slot->kv_resume_protected) {
            result.reason = "slot_protected";
            return result;
        }
        if (slot->prompt.n_tokens() == 0) {
            result.reason = "empty_prompt";
            return result;
        }

        result.bytes_before = (uint64_t) llama_state_seq_get_size_ext(
                ctx_tgt, slot->id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (result.bytes_before == 0) {
            result.reason = "empty_kv";
            return result;
        }

        bool prompt_cached = false;
        for (const auto & cached : prompt_cache->states) {
            const int cur_lcp_len = cached.tokens.get_common_prefix(slot->prompt.tokens);
            if (cur_lcp_len == (int) slot->prompt.tokens.size()) {
                prompt_cached = true;
                break;
            }
        }
        if (!prompt_cached) {
            const size_t cur_size_tgt = (size_t) result.bytes_before;
            const size_t cur_size_dft = ctx_dft
                ? llama_state_seq_get_size_ext(ctx_dft.get(), slot->id, LLAMA_STATE_SEQ_FLAGS_NONE)
                : 0;
            auto * cached = prompt_cache->alloc(slot->prompt, cur_size_tgt, cur_size_dft);
            if (cached == nullptr) {
                result.reason = "prompt_cache_alloc_failed";
                return result;
            }
            llama_state_seq_get_data_ext(
                    ctx_tgt,
                    cached->data.main.data(),
                    cur_size_tgt,
                    slot->id,
                    LLAMA_STATE_SEQ_FLAGS_NONE);
            if (ctx_dft) {
                llama_state_seq_get_data_ext(
                        ctx_dft.get(),
                        cached->data.drft.data(),
                        cur_size_dft,
                        slot->id,
                        LLAMA_STATE_SEQ_FLAGS_NONE);
            }
        }

        result.attempted = true;
        slot->prompt_clear(false);
        prompt_cache->update();
        result.bytes_after = (uint64_t) llama_state_seq_get_size_ext(
                ctx_tgt, slot->id, LLAMA_STATE_SEQ_FLAGS_NONE);
        result.relieved_bytes = result.bytes_before > result.bytes_after
            ? result.bytes_before - result.bytes_after
            : 0;
        result.state_changed = result.relieved_bytes > 0;
        result.target_satisfied = target_bytes == 0 || result.relieved_bytes >= target_bytes;
        result.reason = result.state_changed ? "slot_state_offload_submitted" : "no_relieved_bytes";
        return result;
    }
```

### 初始化 env gate：init_memory_governor_observer_from_env()

来源：`tools/server/server-context.cpp:1497-2015`

```cpp
    void init_memory_governor_observer_from_env() {
        memory_governor_observe_enabled = false;
        memory_governor_observe_last = server_kv_pressure_runtime::time_point {};
        memory_governor_observe_interval_ms = 1000;
        memory_governor_clean_reclaim_enabled = false;
        memory_governor_clean_reclaim_target_bytes = 0;
        memory_governor_clean_reclaim_max_objects = 1;
        memory_governor_clean_reclaim_ranked_enabled = false;
        memory_governor_clean_reclaim_max_passes = 2;
        memory_governor_kv_release_enabled = false;
        memory_governor_kv_release_target_bytes = 0;
        memory_governor_kv_release_max_blocks =
            server_kv_pressure_unified_action_config::DEFAULT_MAX_BLOCKS;
        memory_governor_kv_release_cooldown_samples = 1;
        memory_governor_kv_release_next_sample = 0;
        memory_governor_kv_offload_enabled = false;
        memory_governor_kv_offload_target_bytes = 0;
        memory_governor_kv_offload_max_blocks =
            server_kv_pressure_unified_action_config::DEFAULT_MAX_BLOCKS;
        memory_governor_kv_offload_cooldown_samples = 1;
        memory_governor_kv_offload_next_sample = 0;
        memory_governor_kv_soft_budget_enabled = false;
        memory_governor_kv_soft_target_bytes = 0;
        memory_governor_kv_soft_idle_bytes = 0;
        memory_governor_prefetch_budget_enabled = false;
        memory_governor_prefetch_budget_auto = false;
        memory_governor_prefetch_budget_runtime_auto = false;
        memory_governor_prefetch_budget_bytes_per_tick = 0;
        memory_governor_prefetch_budget_min_bytes_per_tick = 0;
        memory_governor_prefetch_budget_max_bytes_per_tick = 0;
        memory_governor_prefetch_budget_headroom_bytes = 0;
        memory_governor_prefetch_budget_normal_divisor = 0;
        memory_governor_prefetch_budget_pressure_divisor = 0;
        memory_governor_prefetch_budget_critical_divisor = 0;
        memory_governor_prefetch_budget_cgroup_max_bytes = 0;
        memory_governor_prefetch_budget_cgroup_current_bytes = 0;
        memory_governor_prefetch_budget_initial_headroom_bytes = 0;
        memory_governor_prefetch_budget_available_bytes = 0;
        memory_governor_prefetch_budget_kv_resume_used_bytes = 0;
        memory_governor_prefetch_budget_kv_resume_overruns = 0;
        memory_governor_async_actions_enabled = true;
        memory_governor_async_queue_max_depth = 0;
        memory_governor_async_submitted.store(0, std::memory_order_relaxed);
        memory_governor_async_completed.store(0, std::memory_order_relaxed);
        memory_governor_async_rejected.store(0, std::memory_order_relaxed);
        memory_governor_async_dropped.store(0, std::memory_order_relaxed);
        memory_governor_async_relieved_pending_bytes.store(0, std::memory_order_relaxed);
        memory_governor_async_queue_depth.store(0, std::memory_order_relaxed);
        memory_governor_async_reset_inflight();
        memory_governor_global_optimizer_enabled = false;
        memory_governor_global_min_catchup_enabled = true;
        memory_governor_global_hard_headroom_bytes = 0;
        memory_governor_global_roi_threshold = 0.0;
        memory_governor_global_moe_eviction_weight = 1.0;
        memory_governor_global_moe_read_weight = 1.0;
        memory_governor_global_moe_prefetch_weight = 1.0;
        memory_governor_global_kv_reclaim_weight = 1.0;
        memory_governor_global_kv_resident_weight = 1.0;
        memory_governor_reallocation_enabled = false;
        memory_governor_reallocation_apply_moe = true;
        memory_governor_reallocation_confirm_enabled = true;
        memory_governor_reallocation_credit_bytes = 0;
        memory_governor_reallocation_pending_bytes = 0;
        memory_governor_reallocation_last_current_bytes = 0;
        memory_governor_reallocation_last_current_valid = false;
        memory_governor_reallocation_credit_cap_bytes = 0;
        memory_governor_reallocation_max_grant_bytes = 0;
        memory_governor_reallocation_min_grant_bytes = 0;
        memory_governor_reallocation_hard_guard_bytes = 0;
        memory_governor_reallocation_confirm_slack_bytes = 0;
        memory_governor_reallocation_decay = 0.5;
        memory_governor_reallocation_normal_discount = 1.0;
        memory_governor_reallocation_pressure_discount = 0.75;
        memory_governor_moe_budget_dynamic_enabled = false;
        memory_governor_moe_budget_fast_start_enabled = true;
        memory_governor_moe_budget_min_bytes = 0;
        memory_governor_moe_budget_warm_bytes = 0;
        memory_governor_moe_budget_max_bytes = 0;
        memory_governor_moe_budget_grow_bytes = 0;
        memory_governor_moe_budget_headroom_bytes = 0;
        memory_governor_moe_budget_pressure_shrink_pct = 0;
        memory_governor_moe_budget_pressure_shrink_max_bytes = 0;
        memory_governor_moe_budget_grow_samples = 3;
        memory_governor_moe_budget_cooldown_samples = 3;
        memory_governor_moe_budget_next_sample = 0;
        memory_governor_moe_budget_normal_streak = 0;
        memory_governor_moe_budget_stats_valid = false;
        memory_governor_moe_budget_last_evictions = 0;
        memory_governor_moe_budget_last_bytes_read = 0;
        memory_governor_moe_budget_last_prefetch_late = 0;
        memory_governor_moe_budget_last_prefetch_dropped = 0;

        const auto init_cgroup = memory_governor_read_cgroup_memory();
        const bool governor_auto_by_cgroup = init_cgroup.valid && init_cgroup.max_bytes > 0;
        const bool governor_requested =
            server_env_flag("LLAMA_MEMORY_GOVERNOR") || governor_auto_by_cgroup;
        const bool governor_auto_backends =
                server_env_flag("LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS", governor_requested);
        memory_governor_auto_backends_enabled = governor_auto_backends;
        const char * observe = std::getenv("LLAMA_MEMORY_GOVERNOR_OBSERVE");
        const char * clean_reclaim = std::getenv("LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM");
        const char * clean_reclaim_ranked =
            std::getenv("LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED");
        const char * kv_release = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_RELEASE");
        const char * kv_offload = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD");
        const char * kv_soft_budget = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET");
        const char * prefetch_budget = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MB_PER_TICK");
        const char * prefetch_runtime_auto = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO");
        memory_governor_clean_reclaim_enabled =
            clean_reclaim != nullptr ? std::atoi(clean_reclaim) > 0 : governor_auto_backends;
        memory_governor_clean_reclaim_ranked_enabled =
            clean_reclaim_ranked != nullptr ? std::atoi(clean_reclaim_ranked) > 0 : false;
        memory_governor_kv_release_enabled =
            kv_release != nullptr ? std::atoi(kv_release) > 0 : governor_auto_backends;
        memory_governor_kv_offload_enabled =
            kv_offload != nullptr ? std::atoi(kv_offload) > 0 : governor_auto_backends;
        memory_governor_kv_soft_budget_enabled =
            kv_soft_budget != nullptr ? std::atoi(kv_soft_budget) > 0 : governor_auto_backends;
        if (prefetch_budget != nullptr) {
            const double mb = std::max(0.0, std::atof(prefetch_budget));
            memory_governor_prefetch_budget_bytes_per_tick =
                (uint64_t) (mb * 1024.0 * 1024.0);
            memory_governor_prefetch_budget_enabled =
                memory_governor_prefetch_budget_bytes_per_tick > 0;
            memory_governor_prefetch_budget_available_bytes =
                memory_governor_prefetch_budget_bytes_per_tick;
        } else if (governor_auto_backends) {
            const auto cgroup = init_cgroup;
            memory_governor_prefetch_budget_auto = true;
            memory_governor_prefetch_budget_cgroup_max_bytes = cgroup.max_bytes;
            memory_governor_prefetch_budget_cgroup_current_bytes = cgroup.current_bytes;
            memory_governor_prefetch_budget_initial_headroom_bytes = cgroup.headroom_bytes;
            memory_governor_prefetch_budget_bytes_per_tick =
                memory_governor_auto_prefetch_budget(cgroup);
            memory_governor_prefetch_budget_enabled = true;
            memory_governor_prefetch_budget_available_bytes =
                memory_governor_prefetch_budget_bytes_per_tick;
        }
        if (prefetch_runtime_auto != nullptr) {
            memory_governor_prefetch_budget_runtime_auto = std::atoi(prefetch_runtime_auto) > 0;
            if (memory_governor_prefetch_budget_runtime_auto &&
                    memory_governor_prefetch_budget_bytes_per_tick == 0) {
                memory_governor_prefetch_budget_bytes_per_tick =
                    memory_governor_prefetch_budget_max_bytes_per_tick;
            }
            memory_governor_prefetch_budget_enabled =
                memory_governor_prefetch_budget_enabled ||
                memory_governor_prefetch_budget_runtime_auto;
        } else if (governor_auto_backends) {
            memory_governor_prefetch_budget_runtime_auto = true;
            memory_governor_prefetch_budget_enabled = true;
        }
        if (memory_governor_clean_reclaim_enabled ||
                memory_governor_kv_release_enabled ||
                memory_governor_kv_offload_enabled ||
                memory_governor_kv_soft_budget_enabled ||
                memory_governor_prefetch_budget_enabled) {
            memory_governor_observe_enabled = true;
        }
        if ((observe == nullptr || std::atoi(observe) <= 0) &&
                !governor_requested &&
                !memory_governor_clean_reclaim_enabled &&
                !memory_governor_kv_release_enabled &&
                !memory_governor_kv_offload_enabled &&
                !memory_governor_kv_soft_budget_enabled &&
                !memory_governor_prefetch_budget_enabled) {
            return;
        }

        memory_governor_observe_enabled = true;
        if (const char * interval = std::getenv("LLAMA_MEMORY_GOVERNOR_OBSERVE_MS")) {
            memory_governor_observe_interval_ms =
                (uint32_t) std::max(100, std::atoi(interval));
        }
        if (const char * target_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_TARGET_MB")) {
            const double mb = std::max(0.0, std::atof(target_mb));
            memory_governor_clean_reclaim_target_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_objects = std::getenv("LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_MAX_OBJECTS")) {
            memory_governor_clean_reclaim_max_objects =
                (uint32_t) std::max(1, std::atoi(max_objects));
        }
        if (const char * max_passes = std::getenv("LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_MAX_PASSES")) {
            memory_governor_clean_reclaim_max_passes =
                (uint32_t) std::max(1, std::atoi(max_passes));
        }
        if (const char * target_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_RELEASE_TARGET_MB")) {
            const double mb = std::max(0.0, std::atof(target_mb));
            memory_governor_kv_release_target_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_blocks = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_RELEASE_MAX_BLOCKS")) {
            memory_governor_kv_release_max_blocks =
                (uint32_t) std::max(1, std::atoi(max_blocks));
        }
        if (const char * cooldown = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_RELEASE_COOLDOWN_SAMPLES")) {
            memory_governor_kv_release_cooldown_samples =
                (uint32_t) std::max(1, std::atoi(cooldown));
        }
        if (const char * target_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD_TARGET_MB")) {
            const double mb = std::max(0.0, std::atof(target_mb));
            memory_governor_kv_offload_target_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_blocks = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD_MAX_BLOCKS")) {
            memory_governor_kv_offload_max_blocks =
                (uint32_t) std::max(1, std::atoi(max_blocks));
        }
        if (const char * cooldown = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD_COOLDOWN_SAMPLES")) {
            memory_governor_kv_offload_cooldown_samples =
                (uint32_t) std::max(1, std::atoi(cooldown));
        }
        if (const char * target_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_SOFT_TARGET_MB")) {
            const double mb = std::max(0.0, std::atof(target_mb));
            memory_governor_kv_soft_target_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * idle_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_KV_SOFT_IDLE_MB")) {
            const double mb = std::max(0.0, std::atof(idle_mb));
            memory_governor_kv_soft_idle_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * min_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MIN_MB_PER_TICK")) {
            const double mb = std::max(0.0, std::atof(min_mb));
            memory_governor_prefetch_budget_min_bytes_per_tick =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MAX_MB_PER_TICK")) {
            const double mb = std::max(0.0, std::atof(max_mb));
            memory_governor_prefetch_budget_max_bytes_per_tick =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        memory_governor_async_actions_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS", true);
        if (const char * depth = std::getenv("LLAMA_MEMORY_GOVERNOR_ASYNC_QUEUE_DEPTH")) {
            memory_governor_async_queue_max_depth =
                (uint32_t) std::max(1, std::atoi(depth));
        }
        if (const char * headroom_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_HEADROOM_MB")) {
            const double mb = std::max(0.0, std::atof(headroom_mb));
            memory_governor_prefetch_budget_headroom_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * div = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_NORMAL_DIVISOR")) {
            memory_governor_prefetch_budget_normal_divisor =
                (uint32_t) std::max(1, std::atoi(div));
        }
        if (const char * div = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_PRESSURE_DIVISOR")) {
            memory_governor_prefetch_budget_pressure_divisor =
                (uint32_t) std::max(1, std::atoi(div));
        }
        if (const char * div = std::getenv("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_CRITICAL_DIVISOR")) {
            memory_governor_prefetch_budget_critical_divisor =
                (uint32_t) std::max(1, std::atoi(div));
        }
        if (memory_governor_prefetch_budget_runtime_auto && prefetch_budget == nullptr) {
            memory_governor_prefetch_budget_bytes_per_tick =
                memory_governor_prefetch_budget_max_bytes_per_tick;
            memory_governor_prefetch_budget_available_bytes =
                memory_governor_prefetch_budget_bytes_per_tick;
        }
        memory_governor_global_optimizer_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER", governor_auto_backends);
        memory_governor_global_min_catchup_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_GLOBAL_MIN_CATCHUP", true);
        if (const char * hard_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_HARD_HEADROOM_MB")) {
            const double mb = std::max(0.0, std::atof(hard_mb));
            memory_governor_global_hard_headroom_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * threshold = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_ROI_THRESHOLD")) {
            memory_governor_global_roi_threshold = std::max(0.0, std::atof(threshold));
        }
        if (const char * weight = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_MOE_EVICTION_WEIGHT")) {
            memory_governor_global_moe_eviction_weight = std::max(0.0, std::atof(weight));
        }
        if (const char * weight = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_MOE_READ_WEIGHT")) {
            memory_governor_global_moe_read_weight = std::max(0.0, std::atof(weight));
        }
        if (const char * weight = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_MOE_PREFETCH_WEIGHT")) {
            memory_governor_global_moe_prefetch_weight = std::max(0.0, std::atof(weight));
        }
        if (const char * weight = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_KV_RECLAIM_WEIGHT")) {
            memory_governor_global_kv_reclaim_weight = std::max(0.0, std::atof(weight));
        }
        if (const char * weight = std::getenv("LLAMA_MEMORY_GOVERNOR_GLOBAL_KV_RESIDENT_WEIGHT")) {
            memory_governor_global_kv_resident_weight = std::max(0.0, std::atof(weight));
        }
        memory_governor_reallocation_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_REALLOCATION", governor_auto_backends);
        memory_governor_reallocation_apply_moe =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_REALLOCATION_APPLY_MOE", true);
        memory_governor_reallocation_confirm_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM", true);
        if (const char * cap_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_CREDIT_CAP_MB")) {
            const double mb = std::max(0.0, std::atof(cap_mb));
            memory_governor_reallocation_credit_cap_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * grant_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_MAX_GRANT_MB")) {
            const double mb = std::max(0.0, std::atof(grant_mb));
            memory_governor_reallocation_max_grant_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * grant_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_MIN_GRANT_MB")) {
            const double mb = std::max(0.0, std::atof(grant_mb));
            memory_governor_reallocation_min_grant_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * guard_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_HARD_GUARD_MB")) {
            const double mb = std::max(0.0, std::atof(guard_mb));
            memory_governor_reallocation_hard_guard_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * slack_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM_SLACK_MB")) {
            const double mb = std::max(0.0, std::atof(slack_mb));
            memory_governor_reallocation_confirm_slack_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * decay = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_DECAY")) {
            memory_governor_reallocation_decay =
                std::max(0.0, std::min(1.0, std::atof(decay)));
        }
        if (const char * discount = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_NORMAL_DISCOUNT")) {
            memory_governor_reallocation_normal_discount =
                std::max(0.0, std::min(1.0, std::atof(discount)));
        }
        if (const char * discount = std::getenv("LLAMA_MEMORY_GOVERNOR_REALLOCATION_PRESSURE_DISCOUNT")) {
            memory_governor_reallocation_pressure_discount =
                std::max(0.0, std::min(1.0, std::atof(discount)));
        }
        memory_governor_dense_repin_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN", governor_auto_backends);
        memory_governor_dense_repin_async_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_ASYNC", true);
        memory_governor_dense_repin_idle_only =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_IDLE_ONLY", true);
        if (const char * cooldown = std::getenv("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_COOLDOWN_SAMPLES")) {
            memory_governor_dense_repin_cooldown_samples =
                (uint32_t) std::max(0, std::atoi(cooldown));
        }
        if (const char * step_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_STEP_MB")) {
            const double mb = std::max(0.0, std::atof(step_mb));
            memory_governor_dense_repin_step_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_MAX_MB")) {
            const double mb = std::max(0.0, std::atof(max_mb));
            memory_governor_dense_repin_max_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * headroom_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_HEADROOM_MB")) {
            const double mb = std::max(0.0, std::atof(headroom_mb));
            memory_governor_dense_repin_headroom_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * roi = std::getenv("LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_MIN_ROI")) {
            memory_governor_dense_repin_min_roi = std::max(0.0, std::atof(roi));
        }
        memory_governor_dense_runtime_ring_shrink_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_DENSE_RING_SHRINK", true);
        memory_governor_moe_budget_dynamic_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC", governor_auto_backends);
        memory_governor_moe_budget_fast_start_enabled =
            server_env_flag("LLAMA_MEMORY_GOVERNOR_MOE_FAST_START", true);
        if (const char * min_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_MIN_MB")) {
            const double mb = std::max(0.0, std::atof(min_mb));
            memory_governor_moe_budget_min_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * warm_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB")) {
            const double mb = std::max(0.0, std::atof(warm_mb));
            memory_governor_moe_budget_warm_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * max_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB")) {
            const double mb = std::max(0.0, std::atof(max_mb));
            memory_governor_moe_budget_max_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * grow_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_GROW_MB")) {
            const double mb = std::max(0.0, std::atof(grow_mb));
            memory_governor_moe_budget_grow_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * headroom_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_HEADROOM_MB")) {
            const double mb = std::max(0.0, std::atof(headroom_mb));
            memory_governor_moe_budget_headroom_bytes = (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * pct = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_PCT")) {
            memory_governor_moe_budget_pressure_shrink_pct =
                (uint32_t) std::max(0, std::min(100, std::atoi(pct)));
        }
        if (const char * max_mb = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_MAX_MB")) {
            const double mb = std::max(0.0, std::atof(max_mb));
            memory_governor_moe_budget_pressure_shrink_max_bytes =
                (uint64_t) (mb * 1024.0 * 1024.0);
        }
        if (const char * samples = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_GROW_SAMPLES")) {
            memory_governor_moe_budget_grow_samples =
                (uint32_t) std::max(1, std::atoi(samples));
        }
        if (const char * cooldown = std::getenv("LLAMA_MEMORY_GOVERNOR_MOE_COOLDOWN_SAMPLES")) {
            memory_governor_moe_budget_cooldown_samples =
                (uint32_t) std::max(1, std::atoi(cooldown));
        }
        SRV_INF("memory governor observe enabled: interval_ms=%" PRIu32
                " clean_reclaim=%d clean_target_bytes=%" PRIu64
                " clean_max_objects=%" PRIu32
                " clean_ranked=%d clean_max_passes=%" PRIu32
                " kv_release=%d kv_target_bytes=%" PRIu64
                " kv_max_blocks=%" PRIu32
                " kv_cooldown_samples=%" PRIu32
                " kv_offload=%d kv_offload_target_bytes=%" PRIu64
                " kv_offload_max_blocks=%" PRIu32
                " kv_offload_cooldown_samples=%" PRIu32
                " kv_soft_budget=%d kv_soft_target_bytes=%" PRIu64
                " kv_soft_idle_bytes=%" PRIu64
                " prefetch_budget_auto=%d"
                " prefetch_budget_runtime_auto=%d"
                " prefetch_budget_bytes_per_tick=%" PRIu64
                " prefetch_budget_min_bytes_per_tick=%" PRIu64
                " prefetch_budget_max_bytes_per_tick=%" PRIu64
                " prefetch_budget_runtime_headroom_bytes=%" PRIu64
                " prefetch_budget_normal_divisor=%" PRIu32
                " prefetch_budget_pressure_divisor=%" PRIu32
                " prefetch_budget_critical_divisor=%" PRIu32
                " prefetch_budget_cgroup_max_bytes=%" PRIu64
                " prefetch_budget_cgroup_current_bytes=%" PRIu64
                " prefetch_budget_headroom_bytes=%" PRIu64
                " async_actions=%d"
                " async_queue_depth=%" PRIu32
                " global_optimizer=%d"
                " global_min_catchup=%d"
                " global_hard_headroom_bytes=%" PRIu64
                " global_roi_threshold=%.6f"
                " global_moe_eviction_weight=%.6f"
                " global_moe_read_weight=%.6f"
                " global_moe_prefetch_weight=%.6f"
                " global_kv_reclaim_weight=%.6f"
                " global_kv_resident_weight=%.6f"
                " reallocation=%d"
                " reallocation_apply_moe=%d"
                " reallocation_confirm=%d"
                " reallocation_credit_cap_bytes=%" PRIu64
                " reallocation_max_grant_bytes=%" PRIu64
                " reallocation_min_grant_bytes=%" PRIu64
                " reallocation_hard_guard_bytes=%" PRIu64
                " reallocation_confirm_slack_bytes=%" PRIu64
                " reallocation_decay=%.6f"
                " reallocation_normal_discount=%.6f"
                " reallocation_pressure_discount=%.6f"
                " moe_budget_dynamic=%d"
                " moe_budget_fast_start=%d"
                " moe_budget_min_bytes=%" PRIu64
                " moe_budget_warm_bytes=%" PRIu64
                " moe_budget_max_bytes=%" PRIu64
                " moe_budget_grow_bytes=%" PRIu64
                " moe_budget_headroom_bytes=%" PRIu64
                " moe_budget_pressure_shrink_pct=%" PRIu32
                " moe_budget_pressure_shrink_max_bytes=%" PRIu64
                " moe_budget_grow_samples=%" PRIu32
                " moe_budget_cooldown_samples=%" PRIu32 "\n",
                memory_governor_observe_interval_ms,
                memory_governor_clean_reclaim_enabled ? 1 : 0,
                memory_governor_clean_reclaim_target_bytes,
                memory_governor_clean_reclaim_max_objects,
                memory_governor_clean_reclaim_ranked_enabled ? 1 : 0,
                memory_governor_clean_reclaim_max_passes,
                memory_governor_kv_release_enabled ? 1 : 0,
                memory_governor_kv_release_target_bytes,
                memory_governor_kv_release_max_blocks,
                memory_governor_kv_release_cooldown_samples,
                memory_governor_kv_offload_enabled ? 1 : 0,
                memory_governor_kv_offload_target_bytes,
                memory_governor_kv_offload_max_blocks,
                memory_governor_kv_offload_cooldown_samples,
                memory_governor_kv_soft_budget_enabled ? 1 : 0,
                memory_governor_kv_soft_target_bytes,
                memory_governor_kv_soft_idle_bytes,
                memory_governor_prefetch_budget_auto ? 1 : 0,
                memory_governor_prefetch_budget_runtime_auto ? 1 : 0,
                memory_governor_prefetch_budget_bytes_per_tick,
                memory_governor_prefetch_budget_min_bytes_per_tick,
                memory_governor_prefetch_budget_max_bytes_per_tick,
                memory_governor_prefetch_budget_headroom_bytes,
                memory_governor_prefetch_budget_normal_divisor,
                memory_governor_prefetch_budget_pressure_divisor,
                memory_governor_prefetch_budget_critical_divisor,
                memory_governor_prefetch_budget_cgroup_max_bytes,
                memory_governor_prefetch_budget_cgroup_current_bytes,
                memory_governor_prefetch_budget_initial_headroom_bytes,
                memory_governor_async_actions_enabled ? 1 : 0,
                memory_governor_async_queue_max_depth,
                memory_governor_global_optimizer_enabled ? 1 : 0,
                memory_governor_global_min_catchup_enabled ? 1 : 0,
                memory_governor_global_hard_headroom_bytes,
                memory_governor_global_roi_threshold,
                memory_governor_global_moe_eviction_weight,
                memory_governor_global_moe_read_weight,
                memory_governor_global_moe_prefetch_weight,
                memory_governor_global_kv_reclaim_weight,
                memory_governor_global_kv_resident_weight,
                memory_governor_reallocation_enabled ? 1 : 0,
                memory_governor_reallocation_apply_moe ? 1 : 0,
                memory_governor_reallocation_confirm_enabled ? 1 : 0,
                memory_governor_reallocation_credit_cap_bytes,
                memory_governor_reallocation_max_grant_bytes,
                memory_governor_reallocation_min_grant_bytes,
                memory_governor_reallocation_hard_guard_bytes,
                memory_governor_reallocation_confirm_slack_bytes,
                memory_governor_reallocation_decay,
                memory_governor_reallocation_normal_discount,
                memory_governor_reallocation_pressure_discount,
                memory_governor_moe_budget_dynamic_enabled ? 1 : 0,
                memory_governor_moe_budget_fast_start_enabled ? 1 : 0,
                memory_governor_moe_budget_min_bytes,
                memory_governor_moe_budget_warm_bytes,
                memory_governor_moe_budget_max_bytes,
                memory_governor_moe_budget_grow_bytes,
                memory_governor_moe_budget_headroom_bytes,
                memory_governor_moe_budget_pressure_shrink_pct,
                memory_governor_moe_budget_pressure_shrink_max_bytes,
                memory_governor_moe_budget_grow_samples,
                memory_governor_moe_budget_cooldown_samples);
    }
```

### prefetch budget reserve：memory_governor_prefetch_budget_reserve()

来源：`tools/server/server-context.cpp:2029-2040`

```cpp
    bool memory_governor_prefetch_budget_reserve(uint64_t bytes) {
        if (!memory_governor_prefetch_budget_enabled || bytes == 0) {
            return true;
        }
        if (bytes > memory_governor_prefetch_budget_available_bytes) {
            memory_governor_prefetch_budget_kv_resume_overruns++;
            return false;
        }
        memory_governor_prefetch_budget_available_bytes -= bytes;
        memory_governor_prefetch_budget_kv_resume_used_bytes += bytes;
        return true;
    }
```

### candidate 结构：memory_governor_would_candidate

来源：`tools/server/server-context.cpp:2042-2050`

```cpp
    struct memory_governor_would_candidate {
        const char * kind = "none";
        const char * action = "none";
        int32_t id = -1;
        int64_t score = std::numeric_limits<int64_t>::min();
        double roi = -std::numeric_limits<double>::infinity();
        uint64_t bytes = 0;
        const char * reason = "none";
    };
```

### auction select：memory_governor_auction_select()

来源：`tools/server/server-context.cpp:2289-2317`

```cpp
    static memory_governor_would_candidate memory_governor_auction_select(
            const std::vector<memory_governor_would_candidate> & candidates,
            kv_pressure_state pressure_state,
            bool (*predicate)(const memory_governor_would_candidate &),
            const char ** reason) {
        memory_governor_would_candidate best;
        bool has = false;
        bool saw_matching = false;
        bool saw_state_blocked = false;
        for (const auto & candidate : candidates) {
            if (!predicate(candidate)) {
                continue;
            }
            saw_matching = true;
            if (!memory_governor_candidate_allowed_in_state(candidate, pressure_state)) {
                saw_state_blocked = true;
                continue;
            }
            if (!has || memory_governor_candidate_before(candidate, best)) {
                best = candidate;
                has = true;
            }
        }
        if (reason) {
            *reason = has ? "selected" : saw_state_blocked ? "state_gate" :
                saw_matching ? "no_allowed_candidate" : "no_candidate";
        }
        return has ? best : memory_governor_would_candidate {};
    }
```

### KV release/offload result 结构

来源：`tools/server/server-context.cpp:2348-2385`

```cpp
    struct memory_governor_kv_release_result {
        bool attempted = false;
        int64_t score = 0;
        uint64_t target_bytes = 0;
        uint32_t max_blocks = 0;
        uint64_t relieved_bytes = 0;
        uint64_t bytes = 0;
        uint64_t blocks = 0;
        uint64_t shortfall_bytes = 0;
        bool evaluate_attempted = false;
        bool state_changed = false;
        bool io_failure = false;
        bool fail_stop = false;
        int io_errno = 0;
        const char * reason = "disabled";
        const char * outcome = "none";
        const char * action_reason = "none";
    };

    struct memory_governor_kv_offload_result {
        bool attempted = false;
        int32_t seq_id = -1;
        int64_t score = 0;
        uint64_t target_bytes = 0;
        uint32_t max_blocks = 0;
        uint64_t relieved_bytes = 0;
        uint64_t bytes = 0;
        uint64_t blocks = 0;
        uint64_t shortfall_bytes = 0;
        bool evaluate_attempted = false;
        bool state_changed = false;
        bool io_failure = false;
        bool fail_stop = false;
        int io_errno = 0;
        const char * backend = "none";
        const char * reason = "disabled";
        const char * outcome = "none";
        const char * action_reason = "none";
```

### async submit/pop/execute/start/stop 完整实现

来源：`tools/server/server-context.cpp:2421-2763`

```cpp
    void memory_governor_async_reset_inflight() {
        memory_governor_async_dense_clean_inflight.store(false, std::memory_order_relaxed);
        memory_governor_async_moe_clean_inflight.store(false, std::memory_order_relaxed);
        memory_governor_async_kv_release_inflight.store(false, std::memory_order_relaxed);
        memory_governor_async_kv_offload_inflight.store(false, std::memory_order_relaxed);
    }

    bool memory_governor_async_submit(memory_governor_async_action action) {
        if (!memory_governor_async_actions_enabled || action.target_bytes == 0) {
            return false;
        }
        auto & inflight = memory_governor_async_inflight_flag(action.kind);
        bool expected = false;
        if (!inflight.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            memory_governor_async_rejected.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        action.decision_id = next_kv_decision_id();
        {
            std::lock_guard<std::mutex> lock(memory_governor_async_mtx);
            if (memory_governor_async_stop.load(std::memory_order_relaxed)) {
                inflight.store(false, std::memory_order_release);
                memory_governor_async_rejected.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            const uint32_t effective_queue_max_depth =
                memory_governor_async_queue_max_depth > 0
                    ? memory_governor_async_queue_max_depth
                    : (uint32_t) std::max<size_t>(1, slots.size() + 1);
            if (memory_governor_async_queue.size() >= effective_queue_max_depth) {
                inflight.store(false, std::memory_order_release);
                memory_governor_async_dropped.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            memory_governor_async_queue.push(action);
            memory_governor_async_queue_depth.store(
                    memory_governor_async_queue.size(), std::memory_order_relaxed);
        }
        memory_governor_async_submitted.fetch_add(1, std::memory_order_relaxed);
        memory_governor_async_cv.notify_one();
        return true;
    }

    bool memory_governor_async_pop(memory_governor_async_action & action) {
        std::unique_lock<std::mutex> lock(memory_governor_async_mtx);
        memory_governor_async_cv.wait(lock, [&]() {
            return memory_governor_async_stop.load(std::memory_order_relaxed) ||
                !memory_governor_async_queue.empty();
        });
        if (memory_governor_async_stop.load(std::memory_order_relaxed) &&
                memory_governor_async_queue.empty()) {
            memory_governor_async_queue_depth.store(0, std::memory_order_relaxed);
            return false;
        }
        action = memory_governor_async_queue.top();
        memory_governor_async_queue.pop();
        memory_governor_async_queue_depth.store(
                memory_governor_async_queue.size(), std::memory_order_relaxed);
        return true;
    }

    bool memory_governor_async_slot_still_reclaimable(int32_t seq_id) const {
        for (const server_slot & slot : slots) {
            if (slot.id != seq_id) {
                continue;
            }
            const bool active = slot.is_processing() || slot.task != nullptr;
            const bool shared = slot.task && (slot.task->is_parent() || slot.task->is_child());
            return !active && !shared && !slot.kv_resume_protected && slot.prompt.n_tokens() > 0;
        }
        return false;
    }

    void memory_governor_async_execute(const memory_governor_async_action & action) {
        uint64_t relieved_bytes = 0;
        uint64_t bytes = 0;
        uint64_t blocks = 0;
        bool completed = false;
        bool state_changed = false;
        const char * reason = "not_executed";
        const char * outcome = "none";
        const char * action_reason = "none";
        const char * backend = "none";
        int io_errno = 0;

        if (memory_governor_async_stop.load(std::memory_order_relaxed)) {
            reason = "stopping";
        } else if (!ctx_tgt || !model_tgt) {
            reason = "context_unavailable";
        } else {
            switch (action.kind) {
            case memory_governor_async_action_kind::dense_clean_reclaim:
                backend = "flex";
                if (model_tgt->get_flex_context()) {
                    const auto result = llama_flex_reclaim_released(
                            *model_tgt->get_flex_context(),
                            action.target_bytes,
                            (uint32_t) std::max<uint32_t>(1, action.max_objects));
                    relieved_bytes = result.released_bytes;
                    completed = result.target_satisfied || result.released_bytes > 0;
                    state_changed = result.released_bytes > 0;
                    reason = "completed";
                    outcome = completed ? "completed" : "no_op";
                } else {
                    reason = "no_flex";
                }
                break;
            case memory_governor_async_action_kind::moe_clean_reclaim:
                backend = "moe";
                if (model_tgt->get_moe_buffer_context()) {
                    const auto result = llama_moe_buffer_reclaim_clean(
                            *model_tgt->get_moe_buffer_context(),
                            action.target_bytes,
                            (uint32_t) std::max<uint32_t>(1, action.max_objects));
                    relieved_bytes = result.released_bytes;
                    completed = result.target_satisfied || result.released_bytes > 0;
                    state_changed = result.released_bytes > 0;
                    reason = "completed";
                    outcome = completed ? "completed" : "no_op";
                } else {
                    reason = "no_moe";
                }
                break;
            case memory_governor_async_action_kind::kv_global_release:
                backend = "memory_backend";
                if (llama_get_memory(ctx_tgt)) {
                    auto * mem = llama_get_memory(ctx_tgt);
                    const auto evaluation = mem->execute_action({
                            llama_kv_action::evaluate,
                            action.decision_id,
                            -1,
                            0,
                            0,
                            false,
                            false,
                            llama_kv_memory_claimant::kv,
                            llama_kv_io_class::background_write,
                            0,
                            0,
                    });
                    if (evaluation.decision_id != action.decision_id) {
                        reason = "decision_mismatch";
                    } else if (!evaluation.capability.can_release) {
                        reason = "release_unsupported";
                    } else if (evaluation.outcome != llama_kv_action_outcome::completed ||
                            evaluation.reason != llama_kv_action_reason::none) {
                        reason = "evaluate_rejected";
                        outcome = memory_governor_action_outcome_name(evaluation.outcome);
                        action_reason = memory_governor_action_reason_name(evaluation.reason);
                    } else {
                        const auto release = mem->execute_action({
                                llama_kv_action::release,
                                action.decision_id,
                                -1,
                                action.target_bytes,
                                (uint32_t) action.max_blocks,
                                false,
                                false,
                                llama_kv_memory_claimant::kv,
                                llama_kv_io_class::background_write,
                                0,
                                action.target_bytes,
                        });
                        bytes = release.bytes;
                        blocks = release.blocks;
                        relieved_bytes = release.relieved_bytes;
                        state_changed = release.state_changed;
                        completed = release.outcome == llama_kv_action_outcome::completed;
                        io_errno = release.io_errno;
                        outcome = memory_governor_action_outcome_name(release.outcome);
                        action_reason = memory_governor_action_reason_name(release.reason);
                        reason = "completed";
                    }
                } else {
                    reason = "no_memory";
                }
                break;
            case memory_governor_async_action_kind::kv_sequence_offload:
                backend = "memory_backend";
                if (!memory_governor_async_slot_still_reclaimable(action.seq_id)) {
                    reason = "slot_not_reclaimable";
                } else if (llama_get_memory(ctx_tgt)) {
                    auto * mem = llama_get_memory(ctx_tgt);
                    const auto evaluation = mem->execute_action({
                            llama_kv_action::evaluate,
                            action.decision_id,
                            -1,
                            0,
                            0,
                            false,
                            false,
                            llama_kv_memory_claimant::kv,
                            llama_kv_io_class::capacity_write,
                            0,
                            0,
                    });
                    if (evaluation.decision_id != action.decision_id) {
                        reason = "decision_mismatch";
                    } else if (!evaluation.capability.can_offload) {
                        reason = "offload_unsupported";
                    } else if (evaluation.outcome != llama_kv_action_outcome::completed ||
                            evaluation.reason != llama_kv_action_reason::none) {
                        reason = "evaluate_rejected";
                        outcome = memory_governor_action_outcome_name(evaluation.outcome);
                        action_reason = memory_governor_action_reason_name(evaluation.reason);
                    } else {
                        const int64_t bounded_priority = std::max<int64_t>(
                                std::numeric_limits<int32_t>::min(),
                                std::min<int64_t>(
                                        std::numeric_limits<int32_t>::max(),
                                        action.score));
                        const auto offload = mem->execute_action({
                                llama_kv_action::offload,
                                action.decision_id,
                                action.seq_id,
                                action.target_bytes,
                                (uint32_t) action.max_blocks,
                                false,
                                false,
                                llama_kv_memory_claimant::kv,
                                llama_kv_io_class::capacity_write,
                                (int32_t) bounded_priority,
                                action.target_bytes,
                        });
                        bytes = offload.bytes;
                        blocks = offload.blocks;
                        relieved_bytes = offload.relieved_bytes;
                        state_changed = offload.state_changed;
                        completed = offload.outcome == llama_kv_action_outcome::completed;
                        io_errno = offload.io_errno;
                        outcome = memory_governor_action_outcome_name(offload.outcome);
                        action_reason = memory_governor_action_reason_name(offload.reason);
                        reason = "completed";
                    }
                } else {
                    reason = "no_memory";
                }
                break;
            case memory_governor_async_action_kind::kv_sequence_slot_state_offload:
                backend = "slot_state";
                if (!memory_governor_async_slot_still_reclaimable(action.seq_id)) {
                    reason = "slot_not_reclaimable";
                } else {
                    const auto offload =
                        memory_governor_slot_state_offload(action.seq_id, action.target_bytes);
                    bytes = offload.bytes_before;
                    relieved_bytes = offload.relieved_bytes;
                    state_changed = offload.state_changed;
                    completed = offload.state_changed;
                    reason = offload.reason;
                    outcome = completed ? "completed" : "deferred";
                    action_reason = completed ? "none" : offload.reason;
                }
                break;
            }
        }

        if (completed || state_changed || relieved_bytes > 0) {
            memory_governor_async_completed.fetch_add(1, std::memory_order_relaxed);
            if (relieved_bytes > 0) {
                memory_governor_async_relieved_pending_bytes.fetch_add(
                        relieved_bytes, std::memory_order_relaxed);
            }
        } else {
            memory_governor_async_rejected.fetch_add(1, std::memory_order_relaxed);
        }
        memory_governor_async_inflight_flag(action.kind).store(false, std::memory_order_release);

        SRV_INF("memory_governor_async_action"
                " kind=%s"
                " seq_id=%d"
                " decision_id=%" PRIu64
                " sample_count=%" PRIu64
                " score=%" PRId64
                " roi=%.6f"
                " target_bytes=%" PRIu64
                " max_blocks=%" PRIu64
                " backend=%s"
                " completed=%d"
                " state_changed=%d"
                " bytes=%" PRIu64
                " blocks=%" PRIu64
                " relieved_bytes=%" PRIu64
                " outcome=%s"
                " action_reason=%s"
                " reason=%s"
                " io_errno=%d\n",
                memory_governor_async_action_kind_name(action.kind),
                action.seq_id,
                action.decision_id,
                action.sample_count,
                action.score,
                action.roi,
                action.target_bytes,
                action.max_blocks,
                backend,
                completed ? 1 : 0,
                state_changed ? 1 : 0,
                bytes,
                blocks,
                relieved_bytes,
                outcome,
                action_reason,
                reason,
                io_errno);
    }

    void memory_governor_async_worker_loop() {
        while (true) {
            memory_governor_async_action action;
            if (!memory_governor_async_pop(action)) {
                break;
            }
            memory_governor_async_execute(action);
        }
    }

    void memory_governor_async_start() {
        if (!memory_governor_async_actions_enabled ||
                memory_governor_async_worker.joinable()) {
            return;
        }
        memory_governor_async_stop.store(false, std::memory_order_relaxed);
        memory_governor_async_worker =
            std::thread([this]() { memory_governor_async_worker_loop(); });
    }

    void memory_governor_async_stop_worker() {
        memory_governor_async_stop.store(true, std::memory_order_relaxed);
        memory_governor_async_cv.notify_all();
        if (memory_governor_async_worker.joinable()) {
            memory_governor_async_worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(memory_governor_async_mtx);
            while (!memory_governor_async_queue.empty()) {
                memory_governor_async_queue.pop();
                memory_governor_async_dropped.fetch_add(1, std::memory_order_relaxed);
            }
            memory_governor_async_queue_depth.store(0, std::memory_order_relaxed);
        }
        memory_governor_async_reset_inflight();
```

### 运行期主函数：publish_memory_governor_observation()

来源：`tools/server/server-context.cpp:2766-4988`

```cpp
    bool publish_memory_governor_observation(
            bool idle,
            uint64_t sample_count,
            const kv_pressure_telemetry * telemetry) {
        if (!memory_governor_observe_enabled) {
            return false;
        }
        const auto memory_governor_observe_t0 = std::chrono::steady_clock::now();

        llama_kv_release_budget_snapshot kv_budget;
        bool kv_memory_present = false;
        if (ctx_tgt && llama_get_memory(ctx_tgt)) {
            kv_memory_present = true;
            kv_budget = llama_get_memory(ctx_tgt)->sample_kv_release_budget();
        }

        uint64_t kv_slot_resident_bytes = 0;
        uint64_t kv_slot_reclaimable_resident_bytes = 0;
        uint32_t kv_slot_sequences = 0;
        uint32_t kv_slot_reclaimable_sequences = 0;
        bool kv_slot_budget_valid = false;
        if (ctx_tgt && llama_get_memory(ctx_tgt)) {
            for (const auto & slot : slots) {
                const bool active = slot.is_processing() || slot.task != nullptr;
                const bool shared = slot.task && (slot.task->is_parent() || slot.task->is_child());
                const uint64_t logical_tokens = slot.prompt.n_tokens() > 0
                    ? (uint64_t) slot.prompt.n_tokens()
                    : 0;
                if (logical_tokens == 0) {
                    continue;
                }
                const uint64_t seq_bytes = (uint64_t) llama_state_seq_get_size_ext(
                        ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (seq_bytes == 0) {
                    continue;
                }
                kv_slot_budget_valid = true;
                kv_slot_resident_bytes += seq_bytes;
                kv_slot_sequences++;
                if (!active && !shared && !slot.kv_resume_protected) {
                    kv_slot_reclaimable_resident_bytes += seq_bytes;
                    kv_slot_reclaimable_sequences++;
                }
            }
        }

        const bool kv_effective_budget_valid = kv_budget.valid || kv_slot_budget_valid;
        const uint64_t kv_effective_resident_bytes =
            kv_budget.valid ? kv_budget.resident_bytes : kv_slot_resident_bytes;
        const uint64_t kv_effective_reclaimable_resident_bytes =
            kv_budget.valid ? kv_budget.reclaimable_resident_bytes : kv_slot_reclaimable_resident_bytes;
        const char * kv_effective_budget_source =
            kv_budget.valid ? "paged_release" : (kv_slot_budget_valid ? "slot_state" : "none");

        llama_flex_stats flex_stats;
        bool flex_enabled = false;
        if (model_tgt && model_tgt->get_flex_context()) {
            auto * flex = model_tgt->get_flex_context();
            flex_enabled = llama_flex_enabled(flex);
            if (flex_enabled) {
                flex_stats = llama_flex_get_stats(*flex);
            }
        }
        const uint64_t derived_hard_headroom_bytes =
            flex_enabled && flex_stats.slot_bytes > 0
                ? (uint64_t) flex_stats.slot_bytes
                : (kv_slot_reclaimable_sequences > 0
                    ? kv_slot_reclaimable_resident_bytes / kv_slot_reclaimable_sequences
                    : (kv_slot_sequences > 0 ? kv_slot_resident_bytes / kv_slot_sequences : 0));
        const uint64_t effective_hard_headroom_bytes =
            memory_governor_global_hard_headroom_bytes > 0
                ? memory_governor_global_hard_headroom_bytes
                : derived_hard_headroom_bytes;
        const uint64_t effective_prefetch_headroom_bytes =
            memory_governor_prefetch_budget_headroom_bytes > 0
                ? memory_governor_prefetch_budget_headroom_bytes
                : effective_hard_headroom_bytes;
        const uint64_t effective_dense_repin_headroom_bytes =
            memory_governor_dense_repin_headroom_bytes > 0
                ? memory_governor_dense_repin_headroom_bytes
                : effective_hard_headroom_bytes;
        const uint64_t effective_dense_repin_step_bytes =
            memory_governor_dense_repin_step_bytes > 0
                ? memory_governor_dense_repin_step_bytes
                : (uint64_t) flex_stats.slot_bytes;
        const uint64_t effective_dense_repin_max_bytes =
            memory_governor_dense_repin_max_bytes > 0
                ? memory_governor_dense_repin_max_bytes
                : (flex_stats.stream_per_token > flex_stats.delta_locked_bytes
                    ? flex_stats.stream_per_token - flex_stats.delta_locked_bytes
                    : 0);

        const auto pressure_state = telemetry ? telemetry->state : kv_pressure_state::NORMAL;
        const auto pressure_source = telemetry ? telemetry->source : kv_pressure_source::NONE;
        const bool pressure_valid = telemetry ? telemetry->sample_valid : false;
        const bool pressure_stale = telemetry ? telemetry->stale : true;
        const uint64_t pressure_current_bytes = telemetry ? telemetry->pressure_current_bytes : 0;
        const uint64_t pressure_low_water_bytes = telemetry ? telemetry->pressure_low_water_bytes : 0;
        const uint64_t async_relieved_drained_bytes =
            memory_governor_async_relieved_pending_bytes.exchange(
                    0, std::memory_order_relaxed);
        const uint64_t pressure_raw_excess_bytes =
            pressure_current_bytes > pressure_low_water_bytes
                ? pressure_current_bytes - pressure_low_water_bytes
                : 0;
        const uint64_t pressure_excess_bytes = memory_governor_saturating_sub(
                pressure_raw_excess_bytes, async_relieved_drained_bytes);
        kv_pressure_state effective_pressure_state = pressure_state;
        const char * effective_pressure_reason = "telemetry";
        uint64_t effective_pressure_critical_excess_bytes =
            slots.empty() ? pressure_low_water_bytes : pressure_low_water_bytes / slots.size();
        if (pressure_valid && !pressure_stale && flex_enabled) {
            const auto cgroup = memory_governor_read_cgroup_memory();
            if (cgroup.valid && cgroup.max_bytes > 0) {
                const uint64_t pressure_headroom_bytes = effective_hard_headroom_bytes;
                const uint64_t critical_headroom_bytes = pressure_headroom_bytes / 2;
                const uint64_t headroom_bytes = cgroup.max_bytes > pressure_current_bytes
                    ? cgroup.max_bytes - pressure_current_bytes
                    : 0;

                if (cgroup.max_bytes > pressure_low_water_bytes + critical_headroom_bytes) {
                    effective_pressure_critical_excess_bytes =
                        cgroup.max_bytes - pressure_low_water_bytes - critical_headroom_bytes;
                }
                if (headroom_bytes <= critical_headroom_bytes) {
                    effective_pressure_state = kv_pressure_state::CRITICAL;
                    effective_pressure_reason = "dense_flex_hard_headroom";
                } else if (headroom_bytes <= pressure_headroom_bytes) {
                    effective_pressure_state = kv_pressure_state::PRESSURE;
                    effective_pressure_reason = "dense_flex_low_headroom";
                } else {
                    effective_pressure_state = kv_pressure_state::NORMAL;
                    effective_pressure_reason = "dense_flex_headroom";
                }
            }
        } else if (pressure_valid && !pressure_stale && pressure_excess_bytes > 0) {
            if (pressure_excess_bytes >= effective_pressure_critical_excess_bytes) {
                effective_pressure_state = kv_pressure_state::CRITICAL;
                effective_pressure_reason = "high_excess_critical";
            } else if (pressure_state == kv_pressure_state::NORMAL) {
                effective_pressure_state = kv_pressure_state::PRESSURE;
                effective_pressure_reason = "high_excess";
            }
        }
        const uint64_t kv_soft_protected_bytes =
            kv_effective_resident_bytes > kv_effective_reclaimable_resident_bytes
                ? kv_effective_resident_bytes - kv_effective_reclaimable_resident_bytes
                : 0;
        uint64_t kv_soft_target_bytes = 0;
        uint64_t kv_soft_excess_bytes = 0;
        uint64_t kv_soft_release_target_bytes = 0;
        uint64_t kv_soft_offload_target_bytes = 0;
        const char * kv_soft_reason = memory_governor_kv_soft_budget_enabled ? "no_budget" : "disabled";
        if (memory_governor_kv_soft_budget_enabled) {
            if (!kv_effective_budget_valid) {
                kv_soft_reason = "invalid_kv_budget";
            } else {
                kv_soft_target_bytes = memory_governor_kv_soft_target_bytes;
                if (kv_soft_target_bytes == 0) {
                    kv_soft_target_bytes =
                        kv_soft_protected_bytes + memory_governor_kv_soft_idle_bytes;
                    kv_soft_reason = "auto";
                } else {
                    kv_soft_reason = "explicit";
                }
                if (kv_effective_resident_bytes > kv_soft_target_bytes) {
                    kv_soft_excess_bytes = std::min<uint64_t>(
                            kv_effective_resident_bytes - kv_soft_target_bytes,
                            kv_effective_reclaimable_resident_bytes);
                }
            }
        }

        llama_window_stats window_stats;
        bool window_enabled = false;
        if (model_tgt && model_tgt->get_window_context()) {
            auto * window = model_tgt->get_window_context();
            window_enabled = llama_window_enabled(window);
            if (window_enabled) {
                window_stats = llama_window_get_stats(*window);
            }
        }

        llama_moe_buffer_stats moe_stats;
        bool moe_enabled = false;
        llama_moe_buffer_context * moe_context = nullptr;
        if (model_tgt && model_tgt->get_moe_buffer_context()) {
            moe_context = model_tgt->get_moe_buffer_context();
            moe_enabled = llama_moe_buffer_enabled(moe_context);
            if (moe_enabled) {
                moe_stats = llama_moe_buffer_get_stats(*moe_context);
            }
        }
        const uint64_t effective_moe_budget_min_bytes =
            memory_governor_moe_budget_min_bytes > 0
                ? memory_governor_moe_budget_min_bytes
                : moe_stats.expert_bytes;
        const uint64_t effective_moe_budget_grow_bytes =
            memory_governor_moe_budget_grow_bytes > 0
                ? memory_governor_moe_budget_grow_bytes
                : moe_stats.expert_bytes;
        const uint64_t effective_moe_budget_headroom_bytes =
            memory_governor_moe_budget_headroom_bytes > 0
                ? memory_governor_moe_budget_headroom_bytes
                : effective_hard_headroom_bytes;
        const uint64_t effective_moe_budget_warm_bytes =
            memory_governor_moe_budget_warm_bytes > 0
                ? memory_governor_moe_budget_warm_bytes
                : (moe_stats.warm_working_set_bytes > 0
                    ? (uint64_t) moe_stats.warm_working_set_bytes
                    : moe_stats.resident_bytes);
        const uint64_t effective_moe_budget_pressure_shrink_max_bytes =
            memory_governor_moe_budget_pressure_shrink_max_bytes > 0
                ? memory_governor_moe_budget_pressure_shrink_max_bytes
                : moe_stats.expert_bytes;

        const char * moe_budget_action = memory_governor_moe_budget_dynamic_enabled ? "keep" : "disabled";
        const char * moe_budget_reason = memory_governor_moe_budget_dynamic_enabled ? "no_moe" : "disabled";
        uint64_t moe_budget_old_bytes = moe_stats.budget_bytes;
        uint64_t moe_budget_new_bytes = moe_stats.budget_bytes;
        uint64_t moe_budget_headroom_bytes = pressure_low_water_bytes > pressure_current_bytes
            ? pressure_low_water_bytes - pressure_current_bytes
            : 0;
        uint64_t moe_budget_delta_evictions = 0;
        uint64_t moe_budget_delta_bytes_read = 0;
        uint64_t moe_budget_delta_prefetch_late = 0;
        uint64_t moe_budget_delta_prefetch_dropped = 0;
        uint64_t moe_budget_reclaim_target_bytes = 0;
        uint64_t moe_budget_reclaim_released_bytes = 0;
        uint32_t moe_budget_reclaim_released_groups = 0;
        bool moe_budget_reclaim_target_satisfied = false;
        double global_moe_utility = 0.0;
        double global_dense_utility = 0.0;
        double global_kv_utility = 0.0;
        double global_prefetch_utility = 0.0;
        double global_headroom_barrier = 0.0;
        double global_moe_roi = 0.0;
        double global_dense_roi = 0.0;
        double global_dense_headroom_risk = 0.0;
        double global_dense_pressure_tax = 0.0;
        double global_dense_io_bound_ratio = 0.0;
        uint64_t global_available_growth_bytes = 0;
        uint64_t global_moe_grant_bytes = 0;
        uint64_t global_dense_grant_bytes = 0;
        uint64_t global_dense_ring_flowback_bytes = 0;
        uint64_t global_dense_recommended_lock_bytes = 0;
        uint64_t global_dense_target_ring_slots = 0;
        const char * global_optimizer_decision =
            memory_governor_global_optimizer_enabled ? "keep" : "disabled";
        std::vector<memory_governor_would_candidate> grow_candidates;

        if (memory_governor_global_optimizer_enabled && flex_enabled) {
            static constexpr double mib = 1024.0 * 1024.0;
            static constexpr double eps = 1.0e-9;
            const bool dense_pressure_gate =
                effective_pressure_state != kv_pressure_state::NORMAL;
            const auto cgroup = memory_governor_read_cgroup_memory();
            const uint64_t hard_headroom = effective_hard_headroom_bytes;
            const uint64_t available =
                cgroup.valid && cgroup.max_bytes > pressure_current_bytes + hard_headroom
                    ? cgroup.max_bytes - pressure_current_bytes - hard_headroom
                    : 0;
            global_available_growth_bytes = std::max<uint64_t>(
                    global_available_growth_bytes,
                    available);

            const uint64_t dense_slot_bytes = flex_stats.slot_bytes > 0
                ? (uint64_t) flex_stats.slot_bytes
                : (uint64_t) flex_stats.stream_per_token;
            const uint64_t dense_current_slots =
                flex_stats.slot_bytes > 0
                    ? (uint64_t) (flex_stats.ring_bytes / flex_stats.slot_bytes)
                    : 0;
            const bool dense_io_bound =
                flex_stats.ewma_io_us > 0.0 &&
                flex_stats.ewma_compute_us > 0.0 &&
                flex_stats.ewma_io_us > flex_stats.ewma_compute_us;
            global_dense_io_bound_ratio =
                flex_stats.ewma_compute_us > 0.0
                    ? flex_stats.ewma_io_us / flex_stats.ewma_compute_us
                    : 0.0;
            if (dense_io_bound && dense_current_slots > 0) {
                const double useful_slots_f =
                    (double) dense_current_slots *
                    flex_stats.ewma_compute_us /
                    std::max(1.0, flex_stats.ewma_io_us);
                global_dense_target_ring_slots =
                    std::max<uint64_t>(1, (uint64_t) std::ceil(useful_slots_f));
                global_dense_target_ring_slots =
                    std::min<uint64_t>(global_dense_target_ring_slots, dense_current_slots);
            }
            if (dense_io_bound && dense_current_slots > global_dense_target_ring_slots &&
                    flex_stats.slot_bytes > 0) {
                global_dense_ring_flowback_bytes =
                    memory_governor_saturating_mul(
                            (uint64_t) flex_stats.slot_bytes,
                            dense_current_slots - global_dense_target_ring_slots);
                global_dense_recommended_lock_bytes =
                    (uint64_t) flex_stats.locked_bytes + global_dense_ring_flowback_bytes;
                const double dense_scale_mib =
                    std::max(1.0, (double) effective_hard_headroom_bytes / mib);
                global_dense_utility =
                    std::log1p((double) flex_stats.stream_per_token / (dense_scale_mib * mib)) +
                    std::log1p(global_dense_io_bound_ratio);
                global_dense_roi =
                    global_dense_utility /
                    std::max(1.0, (double) global_dense_ring_flowback_bytes / mib);
                global_prefetch_utility = std::max(global_prefetch_utility, global_dense_utility);
                global_optimizer_decision = "dense_lock_rebalance";
            } else if (dense_io_bound) {
                global_dense_recommended_lock_bytes = (uint64_t) flex_stats.locked_bytes;
                const double dense_scale_mib =
                    std::max(1.0, (double) effective_hard_headroom_bytes / mib);
                global_dense_utility =
                    std::log1p((double) flex_stats.stream_per_token / (dense_scale_mib * mib)) +
                    std::log1p(global_dense_io_bound_ratio);
                global_dense_roi = global_dense_utility;
                global_prefetch_utility = std::max(global_prefetch_utility, global_dense_utility);
                global_optimizer_decision = "dense_lock_first";
            }
            if (!dense_io_bound) {
                uint64_t dense_growth_slots = 0;
                if (dense_slot_bytes > 0 &&
                        flex_stats.ewma_wait_us > 0.0 &&
                        flex_stats.ewma_compute_us > 0.0) {
                    const uint64_t slots_by_headroom = available / dense_slot_bytes;
                    const uint64_t slots_by_stall =
                        (uint64_t) std::ceil(
                                flex_stats.ewma_wait_us /
                                std::max(1.0, flex_stats.ewma_compute_us));
                    dense_growth_slots = std::min<uint64_t>(slots_by_headroom, slots_by_stall);
                }
                const uint64_t dense_growth_bytes =
                    memory_governor_saturating_mul(dense_slot_bytes, dense_growth_slots);
                const double dense_growth_mib =
                    std::max(1.0, (double) dense_growth_bytes / mib);
                const double available_mib = std::max(0.0, (double) available / mib);
                const double pressure_margin_mib = std::max(0.0, available_mib);
                global_headroom_barrier = std::max(
                        global_headroom_barrier,
                        std::log1p(1.0 / std::max(eps, pressure_margin_mib)));
                const double dense_scale_mib =
                    std::max(1.0, (double) effective_hard_headroom_bytes / mib);
                const double stall_avoidance =
                    std::log1p((double) flex_stats.total_wait_us / 1000.0) +
                    std::log1p((double) flex_stats.demand_loads) +
                    std::log1p((double) flex_stats.prefetch_budget_dropped) +
                    std::log1p((double) flex_stats.stream_per_token / (dense_scale_mib * mib));
                global_dense_utility = stall_avoidance;
                global_prefetch_utility = std::max(
                        global_prefetch_utility,
                        global_dense_utility);
                global_dense_headroom_risk =
                    available_mib > 0.0
                        ? dense_growth_mib / std::max(1.0, available_mib)
                        : dense_growth_mib;
                global_dense_pressure_tax =
                    dense_pressure_gate ? std::log1p((double) pressure_excess_bytes / mib) * 8.0 : 0.0;
                global_dense_roi =
                    global_dense_utility / dense_growth_mib -
                    global_headroom_barrier / std::max(1.0, pressure_margin_mib) -
                    global_dense_headroom_risk -
                    global_dense_pressure_tax;
                if (dense_pressure_gate) {
                    global_optimizer_decision = "dense_high_excess_gate";
                } else if (effective_pressure_state == kv_pressure_state::NORMAL &&
                        dense_growth_bytes > 0 &&
                        available >= dense_growth_bytes &&
                        global_dense_roi >= memory_governor_global_roi_threshold) {
                    global_dense_grant_bytes = dense_growth_bytes;
                    global_optimizer_decision = "dense_prefetch_window";
                } else if (effective_pressure_state != kv_pressure_state::NORMAL) {
                    global_optimizer_decision = "not_normal";
                } else if (dense_growth_bytes == 0) {
                    global_optimizer_decision = "dense_no_window";
                } else if (available < dense_growth_bytes) {
                    global_optimizer_decision = "dense_hard_headroom";
                } else {
                    global_optimizer_decision = "dense_roi_below_threshold";
                }
            }
        }

        if (memory_governor_moe_budget_dynamic_enabled && moe_enabled && moe_context) {
            moe_budget_reason = "seed";
            if (memory_governor_moe_budget_stats_valid) {
                moe_budget_delta_evictions =
                    moe_stats.evictions >= memory_governor_moe_budget_last_evictions
                        ? moe_stats.evictions - memory_governor_moe_budget_last_evictions
                        : 0;
                moe_budget_delta_bytes_read =
                    moe_stats.bytes_read >= memory_governor_moe_budget_last_bytes_read
                        ? moe_stats.bytes_read - memory_governor_moe_budget_last_bytes_read
                        : 0;
                moe_budget_delta_prefetch_late =
                    moe_stats.prefetch_late >= memory_governor_moe_budget_last_prefetch_late
                        ? moe_stats.prefetch_late - memory_governor_moe_budget_last_prefetch_late
                        : 0;
                moe_budget_delta_prefetch_dropped =
                    moe_stats.prefetch_budget_dropped >= memory_governor_moe_budget_last_prefetch_dropped
                        ? moe_stats.prefetch_budget_dropped - memory_governor_moe_budget_last_prefetch_dropped
                        : 0;

                if (effective_pressure_state == kv_pressure_state::NORMAL) {
                    memory_governor_moe_budget_normal_streak++;
                } else {
                    memory_governor_moe_budget_normal_streak = 0;
                }

                uint64_t max_budget = memory_governor_moe_budget_max_bytes;
                if (max_budget == 0 || max_budget > moe_stats.expert_bytes) {
                    max_budget = moe_stats.expert_bytes;
                }
                if (moe_stats.planner_floor_bytes > 0) {
                    max_budget = std::min<uint64_t>(
                            max_budget,
                            (uint64_t) moe_stats.planner_safe_budget_bytes);
                }
                if (moe_stats.planner_floor_bytes == 0 &&
                        max_budget < effective_moe_budget_min_bytes) {
                    max_budget = effective_moe_budget_min_bytes;
                }

                const uint64_t current_budget =
                    moe_stats.budget_unbounded ? max_budget : (uint64_t) moe_stats.budget_bytes;
                const bool unlimited = moe_stats.budget_unbounded;
                const bool cooldown =
                    sample_count != 0 && sample_count < memory_governor_moe_budget_next_sample;

                if (memory_governor_global_optimizer_enabled) {
                    static constexpr double mib = 1024.0 * 1024.0;
                    static constexpr double eps = 1.0e-9;
                    const double current_mib = std::max(1.0, current_budget / mib);
                    const double resident_mib = std::max(1.0, (double) moe_stats.resident_bytes / mib);
                    const double kv_reclaim_mib =
                        (double) kv_effective_reclaimable_resident_bytes / mib;
                    const double kv_resident_mib =
                        (double) kv_effective_resident_bytes / mib;
                    const double pressure_margin_mib =
                        moe_budget_headroom_bytes > effective_hard_headroom_bytes
                            ? (double) (moe_budget_headroom_bytes -
                                    effective_hard_headroom_bytes) / mib
                            : 0.0;
                    global_available_growth_bytes =
                        moe_budget_headroom_bytes > effective_hard_headroom_bytes
                            ? moe_budget_headroom_bytes - effective_hard_headroom_bytes
                            : 0;
                    const uint64_t working_set_deficit =
                        effective_moe_budget_warm_bytes > current_budget
                            ? effective_moe_budget_warm_bytes - current_budget
                            : 0;
                    uint64_t adaptive_grow_bytes = working_set_deficit > 0
                        ? working_set_deficit
                        : effective_moe_budget_grow_bytes;
                    adaptive_grow_bytes = std::min<uint64_t>(
                            adaptive_grow_bytes,
                            max_budget > current_budget ? max_budget - current_budget : 0);
                    const double grant_mib = std::max(
                            1.0,
                            (double) std::max<uint64_t>(adaptive_grow_bytes, effective_moe_budget_grow_bytes) / mib);
                    global_headroom_barrier = std::log1p(
                            1.0 / std::max(eps, pressure_margin_mib));
                    const double moe_scale_mib =
                        std::max(1.0, (double) effective_hard_headroom_bytes / mib);
                    global_moe_utility =
                        memory_governor_global_moe_eviction_weight *
                            std::log1p((double) moe_budget_delta_evictions) +
                        memory_governor_global_moe_read_weight *
                            std::log1p((double) moe_budget_delta_bytes_read /
                                    (moe_scale_mib * mib)) +
                        memory_governor_global_moe_prefetch_weight *
                            std::log1p((double) moe_budget_delta_prefetch_late +
                                    (double) moe_budget_delta_prefetch_dropped);
                    global_kv_utility =
                        memory_governor_global_kv_reclaim_weight *
                            std::log1p(kv_reclaim_mib) +
                        memory_governor_global_kv_resident_weight *
                            std::log1p(kv_resident_mib);
                    global_prefetch_utility =
                        std::log1p((double) moe_budget_delta_prefetch_late +
                                (double) moe_budget_delta_prefetch_dropped);
                    const double diminishing_return =
                        1.0 + current_mib / std::max(
                                std::max(1.0, (double) effective_hard_headroom_bytes / mib),
                                resident_mib);
                    const double pressure_excess_mib =
                        (double) pressure_excess_bytes / mib;
                    const double allocation_risk_tax =
                        std::log1p(pressure_excess_mib) +
                        grant_mib / std::max(1.0, pressure_margin_mib);
                    global_moe_roi =
                        global_moe_utility / (grant_mib * diminishing_return) -
                        global_headroom_barrier / std::max(1.0, pressure_margin_mib) -
                        allocation_risk_tax;
                    if (effective_pressure_state == kv_pressure_state::NORMAL &&
                            pressure_excess_bytes == 0 &&
                            global_available_growth_bytes > 0 &&
                            current_budget < max_budget &&
                            memory_governor_moe_budget_normal_streak >=
                                memory_governor_moe_budget_grow_samples) {
                        bool should_grant = global_moe_roi >=
                            memory_governor_global_roi_threshold;
                        uint64_t target = max_budget;
                        if (memory_governor_global_min_catchup_enabled &&
                                current_budget < effective_moe_budget_min_bytes &&
                                global_moe_utility > 0.0) {
                            should_grant = true;
                            target = effective_moe_budget_min_bytes;
                            global_optimizer_decision = "moe_min_catchup";
                        } else if (should_grant) {
                            global_optimizer_decision =
                                global_moe_utility >= global_kv_utility &&
                                global_moe_utility >= global_prefetch_utility
                                    ? "moe_roi_grant"
                                    : "moe_roi_shadowed";
                        } else {
                            global_optimizer_decision = "roi_below_threshold";
                        }
                        if (should_grant) {
                            global_moe_grant_bytes = std::min<uint64_t>(
                                    adaptive_grow_bytes,
                                    target > current_budget ? target - current_budget : 0);
                            global_moe_grant_bytes = std::min<uint64_t>(
                                    global_moe_grant_bytes, max_budget - current_budget);
                            global_moe_grant_bytes = std::min<uint64_t>(
                                    global_moe_grant_bytes, global_available_growth_bytes);
                        }
                    } else if (effective_pressure_state != kv_pressure_state::NORMAL) {
                        global_optimizer_decision = "not_normal";
                    } else if (pressure_excess_bytes > 0) {
                        global_optimizer_decision = "high_excess";
                    } else if (global_available_growth_bytes == 0) {
                        global_optimizer_decision = "hard_headroom";
                    } else if (current_budget >= max_budget) {
                        global_optimizer_decision = "max";
                    } else {
                        global_optimizer_decision = "normal_streak";
                    }
                    if (global_moe_grant_bytes > 0) {
                        static constexpr double roi_scale = 1000000.0;
                        static constexpr double mib = 1024.0 * 1024.0;
                        const double grant_mib = std::max(
                                1.0,
                                (double) global_moe_grant_bytes / mib);
                        const double normalized_roi = std::max(0.0, global_moe_roi);
                        const double score_d = normalized_roi * roi_scale * grant_mib;
                        const int64_t score = score_d >
                                (double) std::numeric_limits<int64_t>::max()
                            ? std::numeric_limits<int64_t>::max()
                            : (int64_t) score_d;
                        memory_governor_add_candidate(
                                grow_candidates,
                                "moe_expert",
                                "grow",
                                -1,
                                score,
                                global_moe_grant_bytes,
                                global_optimizer_decision);
                    }
                }

                if (cooldown) {
                    moe_budget_reason = "cooldown";
                } else if (effective_pressure_state == kv_pressure_state::CRITICAL ||
                           effective_pressure_state == kv_pressure_state::PRESSURE) {
                    if (current_budget <= effective_moe_budget_min_bytes) {
                        moe_budget_reason = "min";
                    } else {
                        uint64_t next_budget = current_budget;
                        if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                            next_budget = current_budget / 2;
                            moe_budget_reason = "critical";
                        } else {
                            const uint64_t proportional_shrink =
                                memory_governor_moe_budget_pressure_shrink_pct > 0
                                    ? std::max<uint64_t>(
                                            1,
                                            current_budget *
                                                memory_governor_moe_budget_pressure_shrink_pct / 100)
                                    : current_budget;
                            uint64_t shrink = pressure_excess_bytes == 0
                                ? effective_moe_budget_grow_bytes
                                : pressure_excess_bytes;
                            shrink = std::min<uint64_t>(shrink, proportional_shrink);
                            if (effective_moe_budget_pressure_shrink_max_bytes != 0) {
                                shrink = std::min<uint64_t>(
                                        shrink,
                                        effective_moe_budget_pressure_shrink_max_bytes);
                            }
                            next_budget = current_budget > shrink ? current_budget - shrink : 0;
                            moe_budget_reason = "pressure_smooth";
                        }
                        next_budget = std::max<uint64_t>(
                                effective_moe_budget_min_bytes,
                                next_budget);
                        if (next_budget < current_budget) {
                            llama_moe_buffer_set_budget(moe_context, (size_t) next_budget);
                            moe_budget_action = "shrink";
                            moe_budget_new_bytes = next_budget;
                            moe_budget_reclaim_target_bytes = current_budget - next_budget;
                            const auto reclaim = llama_moe_buffer_reclaim_clean(
                                    *moe_context,
                                    moe_budget_reclaim_target_bytes,
                                    memory_governor_clean_reclaim_max_objects);
                            moe_budget_reclaim_released_bytes = reclaim.released_bytes;
                            moe_budget_reclaim_released_groups = reclaim.released_groups;
                            moe_budget_reclaim_target_satisfied = reclaim.target_satisfied;
                            memory_governor_moe_budget_next_sample =
                                sample_count + memory_governor_moe_budget_cooldown_samples;
                        }
                    }
                } else if (unlimited) {
                    moe_budget_reason = "unbounded";
                } else if (memory_governor_moe_budget_fast_start_enabled &&
                           moe_budget_headroom_bytes > effective_moe_budget_headroom_bytes &&
                           current_budget < max_budget &&
                           current_budget < effective_moe_budget_warm_bytes) {
                    const uint64_t safe_headroom =
                        moe_budget_headroom_bytes - effective_moe_budget_headroom_bytes;
                    const uint64_t warm_target = std::min<uint64_t>(
                            std::min<uint64_t>(
                                effective_moe_budget_warm_bytes,
                                max_budget),
                            current_budget + safe_headroom);
                    if (warm_target > current_budget) {
                        llama_moe_buffer_set_budget(moe_context, (size_t) warm_target);
                        moe_budget_action = "grow";
                        moe_budget_reason = "fast_start";
                        moe_budget_new_bytes = warm_target;
                        memory_governor_moe_budget_next_sample =
                            sample_count + memory_governor_moe_budget_cooldown_samples;
                    } else {
                        moe_budget_reason = "headroom";
                    }
                } else if (memory_governor_moe_budget_normal_streak <
                           memory_governor_moe_budget_grow_samples) {
                    moe_budget_reason = "normal_streak";
                } else if (memory_governor_global_optimizer_enabled &&
                           !grow_candidates.empty()) {
                    const char * grow_auction_reason = "no_candidate";
                    const auto selected = memory_governor_auction_select(
                            grow_candidates,
                            effective_pressure_state,
                            memory_governor_is_moe_grow_candidate,
                            &grow_auction_reason);
                    if (memory_governor_is_moe_grow_candidate(selected)) {
                        const uint64_t next_budget = current_budget + selected.bytes;
                        llama_moe_buffer_set_budget(moe_context, (size_t) next_budget);
                        moe_budget_action = "grow";
                        moe_budget_reason = selected.reason;
                        moe_budget_new_bytes = next_budget;
                        memory_governor_moe_budget_next_sample =
                            sample_count + memory_governor_moe_budget_cooldown_samples;
                    } else {
                        moe_budget_reason = grow_auction_reason;
                    }
                } else if (memory_governor_global_optimizer_enabled) {
                    moe_budget_reason = global_optimizer_decision;
                } else if (moe_budget_headroom_bytes <= effective_moe_budget_headroom_bytes) {
                    moe_budget_reason = "headroom";
                } else if (current_budget >= max_budget) {
                    moe_budget_reason = "max";
                } else if (current_budget < effective_moe_budget_warm_bytes) {
                    const uint64_t safe_headroom =
                        moe_budget_headroom_bytes - effective_moe_budget_headroom_bytes;
                    uint64_t grow = std::min<uint64_t>(
                            effective_moe_budget_warm_bytes - current_budget,
                            safe_headroom);
                    grow = std::min<uint64_t>(grow, max_budget - current_budget);
                    if (grow == 0) {
                        moe_budget_reason = "headroom";
                    } else {
                        const uint64_t next_budget = current_budget + grow;
                        llama_moe_buffer_set_budget(moe_context, (size_t) next_budget);
                        moe_budget_action = "grow";
                        moe_budget_reason = "working_set_warm";
                        moe_budget_new_bytes = next_budget;
                        memory_governor_moe_budget_next_sample =
                            sample_count + memory_governor_moe_budget_cooldown_samples;
                    }
                } else {
                    moe_budget_reason = "working_set_ready";
                }
            }

            memory_governor_moe_budget_stats_valid = true;
            memory_governor_moe_budget_last_evictions = moe_stats.evictions;
            memory_governor_moe_budget_last_bytes_read = moe_stats.bytes_read;
            memory_governor_moe_budget_last_prefetch_late = moe_stats.prefetch_late;
            memory_governor_moe_budget_last_prefetch_dropped = moe_stats.prefetch_budget_dropped;

            if (std::string(moe_budget_action) == "grow" ||
                    std::string(moe_budget_action) == "shrink") {
                moe_stats = llama_moe_buffer_get_stats(*moe_context);
            }
        }

        llama_flex_resize_result dense_resize_result;
        dense_resize_result.reason = flex_enabled ? "no_grant" : "disabled";
        uint64_t dense_ring_shrink_required_bytes = 0;
        uint64_t dense_ring_shrink_releasable_bytes = 0;
        int dense_ring_shrink_target_slots = 0;
        if (memory_governor_dense_runtime_ring_shrink_enabled &&
                flex_enabled &&
                model_tgt &&
                model_tgt->get_flex_context() &&
                flex_stats.slot_bytes > 0 &&
                effective_pressure_state != kv_pressure_state::NORMAL) {
            const int current_slots = (int) (flex_stats.ring_bytes / flex_stats.slot_bytes);
            const auto cgroup = memory_governor_read_cgroup_memory();
            const uint64_t observed_current_bytes =
                pressure_current_bytes > 0 ? pressure_current_bytes : cgroup.current_bytes;
            uint64_t headroom_bytes = 0;
            if (cgroup.valid && cgroup.max_bytes > observed_current_bytes) {
                headroom_bytes = cgroup.max_bytes - observed_current_bytes;
            }
            uint64_t required_headroom = effective_hard_headroom_bytes;
            if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                required_headroom += flex_stats.slot_bytes;
            }
            if (required_headroom > headroom_bytes) {
                dense_ring_shrink_required_bytes = required_headroom - headroom_bytes;
            }
            dense_ring_shrink_required_bytes =
                std::max<uint64_t>(dense_ring_shrink_required_bytes, pressure_excess_bytes);
            if (current_slots > 1) {
                dense_ring_shrink_releasable_bytes =
                    (uint64_t) (current_slots - 1) * (uint64_t) flex_stats.slot_bytes;
            }
            if (!cgroup.valid && pressure_excess_bytes == 0) {
                dense_resize_result.reason = "no_headroom_signal";
            } else if (dense_ring_shrink_required_bytes == 0) {
                dense_resize_result.reason = "no_shrink_required";
            } else if (dense_ring_shrink_releasable_bytes == 0) {
                dense_resize_result.reason = "ring_minimum";
            } else {
                const uint64_t release_bytes =
                    std::min<uint64_t>(dense_ring_shrink_required_bytes,
                            dense_ring_shrink_releasable_bytes);
                const int release_slots =
                    (int) ((release_bytes + flex_stats.slot_bytes - 1) / flex_stats.slot_bytes);
                const int target_slots = std::max(1, current_slots - release_slots);
                dense_ring_shrink_target_slots = target_slots;
                dense_resize_result = llama_flex_resize_ring(
                        *model_tgt->get_flex_context(),
                        target_slots);
                flex_stats = llama_flex_get_stats(*model_tgt->get_flex_context());
            }
        } else if (memory_governor_global_optimizer_enabled &&
                global_dense_ring_flowback_bytes > 0 &&
                global_dense_target_ring_slots > 0 &&
                flex_enabled &&
                model_tgt &&
                model_tgt->get_flex_context() &&
                flex_stats.slot_bytes > 0) {
            dense_resize_result = llama_flex_resize_ring(
                    *model_tgt->get_flex_context(),
                    (int) global_dense_target_ring_slots);
            flex_stats = llama_flex_get_stats(*model_tgt->get_flex_context());
        } else if (memory_governor_global_optimizer_enabled &&
                global_dense_grant_bytes > 0 &&
                flex_enabled &&
                model_tgt &&
                model_tgt->get_flex_context() &&
                flex_stats.slot_bytes > 0) {
            const int current_slots = (int) (flex_stats.ring_bytes / flex_stats.slot_bytes);
            const int add_slots = std::max<int>(
                    1,
                    (int) (global_dense_grant_bytes / flex_stats.slot_bytes));
            dense_resize_result = llama_flex_resize_ring(
                    *model_tgt->get_flex_context(),
                    current_slots + add_slots);
            flex_stats = llama_flex_get_stats(*model_tgt->get_flex_context());
        }

        const uint64_t dense_reclaimable_bytes = (uint64_t) flex_stats.ring_bytes;
        const uint64_t dense_ring_floor_bytes = 0;
        const uint64_t dense_reclaimable_above_floor = dense_reclaimable_bytes;
        const uint64_t window_observed_rss =
            window_stats.current_rss != 0 ? (uint64_t) window_stats.current_rss : pressure_current_bytes;
        const uint64_t pressure_reclaim_boost_bytes =
            pressure_excess_bytes +
            (effective_pressure_state == kv_pressure_state::NORMAL
                ? 0
                : effective_hard_headroom_bytes);

        std::vector<memory_governor_would_candidate> reclaim_candidates;
        std::vector<memory_governor_would_candidate> prefetch_candidates;

        if (flex_enabled && dense_reclaimable_bytes > 0) {
            int64_t score = 0;
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(dense_reclaimable_bytes, 4096, 1024));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(flex_stats.stream_per_token, 4096, -64));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(pressure_reclaim_boost_bytes, 4096, 512));
            memory_governor_add_candidate(
                    reclaim_candidates,
                    "dense_layer",
                    "reclaim_clean",
                    -1,
                    score,
                    dense_reclaimable_bytes,
                    "flex_ring");
        }

        if (moe_enabled && moe_stats.resident_bytes > 0) {
            const bool over_budget =
                !moe_stats.budget_unbounded && moe_stats.resident_bytes > moe_stats.budget_bytes;
            int64_t score = 0;
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(moe_stats.resident_bytes, 4096, 512));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(moe_stats.bytes_read, 4096, -16));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(pressure_excess_bytes, 4096, 128));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(pressure_reclaim_boost_bytes, 4096, 512));
            if (over_budget) {
                score = memory_governor_add_score(score, 1000000);
            }
            memory_governor_add_candidate(
                    reclaim_candidates,
                    "moe_expert",
                    "reclaim_clean",
                    -1,
                    score,
                    moe_stats.resident_bytes,
                    over_budget ? "over_budget" : "resident_pool");
        }

        if (kv_budget.valid && kv_effective_reclaimable_resident_bytes > 0) {
            int64_t score = 0;
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(kv_effective_reclaimable_resident_bytes, 4096, 512));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(pressure_excess_bytes, 4096, 128));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(pressure_reclaim_boost_bytes, 4096, 512));
            memory_governor_add_candidate(
                    reclaim_candidates,
                    "kv_global",
                    "release",
                    -1,
                    score,
                    kv_effective_reclaimable_resident_bytes,
                    "kv_reclaimable");
        }

        if (ctx_tgt && llama_get_memory(ctx_tgt)) {
            const int64_t now_us = ggml_time_us();
            for (const auto & slot : slots) {
                const bool active = slot.is_processing() || slot.task != nullptr;
                const bool shared = slot.task && (slot.task->is_parent() || slot.task->is_child());
                const uint64_t logical_tokens = slot.prompt.n_tokens() > 0
                    ? (uint64_t) slot.prompt.n_tokens()
                    : 0;
                const uint64_t reclaimable_bytes = !active && logical_tokens > 0
                    ? (uint64_t) llama_state_seq_get_size_ext(
                            ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE)
                    : 0;
                if (!active && !shared && !slot.kv_resume_protected && reclaimable_bytes > 0) {
                    const uint64_t idle_age_us =
                        slot.t_last_used >= 0 && now_us > slot.t_last_used
                            ? (uint64_t) (now_us - slot.t_last_used)
                            : 0;
                    const uint64_t reuse_hint =
                        slot.kv_reuse_hint_tokens > 0
                            ? (uint64_t) slot.kv_reuse_hint_tokens
                            : 0;
                    int64_t score = 0;
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(reclaimable_bytes, 4096, 512));
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(idle_age_us, 1000, 4));
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(logical_tokens, 1, 256));
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(reuse_hint, 1, -1024));
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(pressure_reclaim_boost_bytes, 4096, 512));
                    memory_governor_add_candidate(
                            reclaim_candidates,
                            "kv_sequence",
                            "offload",
                            slot.id,
                            score,
                            reclaimable_bytes,
                            "idle_seq");
                }

                if ((active || slot.kv_resume_protected) && logical_tokens > 0) {
                    int64_t score = slot.kv_resume_protected ? 2000000 : 0;
                    score = memory_governor_add_score(
                            score, memory_governor_bounded_term(logical_tokens, 1, 256));
                    if (slot.kv_reuse_hint_tokens > 0) {
                        score = memory_governor_add_score(
                                score,
                                memory_governor_bounded_term(
                                        (uint64_t) slot.kv_reuse_hint_tokens, 1, 1024));
                    }
                    memory_governor_add_candidate(
                            prefetch_candidates,
                            "kv_sequence",
                            "prefetch",
                            slot.id,
                            score,
                            logical_tokens,
                            slot.kv_resume_protected ? "resume_protected" : "active_seq");
                }
            }
        }

        if (flex_enabled && flex_stats.stream_per_token > 0) {
            int64_t score = 0;
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(flex_stats.total_wait_us, 1000, 1024));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(flex_stats.demand_loads, 1, 1000000));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(flex_stats.stream_per_token, 4096, 32));
            memory_governor_add_candidate(
                    prefetch_candidates,
                    "dense_layer",
                    "prefetch",
                    -1,
                    score,
                    flex_stats.stream_per_token,
                    "flex_stream");
        }

        if (moe_enabled && moe_stats.expert_bytes > 0) {
            int64_t score = 0;
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(moe_stats.prefetch_late, 1, 1000000));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(moe_stats.cache_misses, 1, 4096));
            score = memory_governor_add_score(
                    score, memory_governor_bounded_term(moe_stats.expert_bytes, 4096, 8));
            memory_governor_add_candidate(
                    prefetch_candidates,
                    "moe_expert",
                    "prefetch",
                    -1,
                    score,
                    moe_stats.expert_bytes,
                    "moe_predictor");
        }

        uint64_t prefetch_budget_tick_bytes = 0;
        uint64_t prefetch_budget_remaining_bytes = 0;
        uint64_t prefetch_budget_dense_bytes = 0;
        uint64_t prefetch_budget_moe_bytes = 0;
        uint64_t prefetch_budget_kv_reserved_bytes =
            memory_governor_prefetch_budget_kv_resume_used_bytes;
        uint64_t prefetch_budget_kv_overruns =
            memory_governor_prefetch_budget_kv_resume_overruns;
        uint64_t prefetch_budget_config_bytes =
            memory_governor_prefetch_budget_bytes_per_tick;
        uint64_t prefetch_budget_effective_bytes =
            memory_governor_prefetch_budget_bytes_per_tick;
        const char * prefetch_budget_clamp_reason = "none";
        const char * prefetch_budget_auto_reason = memory_governor_prefetch_budget_runtime_auto ? "runtime" : "disabled";
        const uint64_t prefetch_budget_runtime_headroom_bytes =
            pressure_low_water_bytes > pressure_current_bytes
                ? pressure_low_water_bytes - pressure_current_bytes
                : 0;
        if (memory_governor_prefetch_budget_enabled) {
            if (memory_governor_prefetch_budget_runtime_auto) {
                uint64_t usable_headroom = 0;
                if (prefetch_budget_runtime_headroom_bytes >
                        effective_prefetch_headroom_bytes) {
                    usable_headroom = prefetch_budget_runtime_headroom_bytes -
                        effective_prefetch_headroom_bytes;
                }
                uint32_t divisor = memory_governor_prefetch_budget_normal_divisor;
                if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                    divisor = memory_governor_prefetch_budget_critical_divisor;
                    prefetch_budget_clamp_reason = "critical";
                } else if (effective_pressure_state == kv_pressure_state::PRESSURE) {
                    divisor = memory_governor_prefetch_budget_pressure_divisor;
                    prefetch_budget_clamp_reason = "pressure";
                }
                prefetch_budget_effective_bytes =
                    divisor > 0 ? usable_headroom / divisor : usable_headroom;
                if (prefetch_budget_effective_bytes > 0) {
                    prefetch_budget_effective_bytes =
                        memory_governor_round_prefetch_budget(prefetch_budget_effective_bytes);
                }
                if (prefetch_budget_effective_bytes > 0 &&
                        prefetch_budget_effective_bytes <
                            memory_governor_prefetch_budget_min_bytes_per_tick) {
                    prefetch_budget_effective_bytes =
                        memory_governor_prefetch_budget_min_bytes_per_tick;
                }
                if (memory_governor_prefetch_budget_max_bytes_per_tick > 0) {
                    prefetch_budget_effective_bytes = std::min<uint64_t>(
                            prefetch_budget_effective_bytes,
                            memory_governor_prefetch_budget_max_bytes_per_tick);
                }
                if (usable_headroom == 0) {
                    prefetch_budget_auto_reason = "headroom";
                }
            } else if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                prefetch_budget_effective_bytes =
                    std::min<uint64_t>(prefetch_budget_effective_bytes, effective_hard_headroom_bytes);
                prefetch_budget_clamp_reason = "critical";
            } else if (effective_pressure_state == kv_pressure_state::PRESSURE) {
                prefetch_budget_effective_bytes =
                    std::min<uint64_t>(prefetch_budget_effective_bytes, effective_hard_headroom_bytes);
                prefetch_budget_clamp_reason = "pressure";
            }
            if (flex_enabled && flex_stats.slot_bytes > 0) {
                const uint64_t dense_slot_bytes = (uint64_t) flex_stats.slot_bytes;
                const uint64_t dense_ahead = (uint64_t) std::max(1, flex_stats.effective_ahead);
                const uint64_t dense_window_bytes =
                    memory_governor_saturating_mul(dense_slot_bytes, dense_ahead);
                if (effective_pressure_state == kv_pressure_state::PRESSURE ||
                        effective_pressure_state == kv_pressure_state::CRITICAL) {
                    const uint64_t safe_floor = std::min<uint64_t>(
                            dense_slot_bytes,
                            dense_window_bytes == 0 ? dense_slot_bytes : dense_window_bytes);
                    prefetch_budget_effective_bytes = std::max<uint64_t>(
                            prefetch_budget_effective_bytes,
                            safe_floor);
                    prefetch_budget_effective_bytes = std::min<uint64_t>(
                            prefetch_budget_effective_bytes,
                            safe_floor);
                    prefetch_budget_clamp_reason =
                        effective_pressure_state == kv_pressure_state::CRITICAL
                            ? "dense_flex_critical_slot"
                            : "dense_flex_pressure_slot";
                }
            }
        }
        if (memory_governor_prefetch_budget_enabled) {
            memory_governor_prefetch_budget_available_bytes =
                prefetch_budget_effective_bytes;
            prefetch_budget_tick_bytes = prefetch_budget_effective_bytes;
            std::vector<memory_governor_would_candidate> sorted_prefetch;
            for (const auto & candidate : prefetch_candidates) {
                if (memory_governor_candidate_allowed_in_state(candidate, effective_pressure_state)) {
                    sorted_prefetch.push_back(candidate);
                }
            }
            std::sort(sorted_prefetch.begin(), sorted_prefetch.end(), memory_governor_candidate_before);
            for (const auto & candidate : sorted_prefetch) {
                if (memory_governor_prefetch_budget_available_bytes == 0) {
                    break;
                }
                if (std::string(candidate.kind) == "dense_layer" &&
                        std::string(candidate.action) == "prefetch" &&
                        prefetch_budget_dense_bytes == 0) {
                    if (flex_enabled && effective_pressure_state == kv_pressure_state::NORMAL) {
                        if (model_tgt && model_tgt->get_flex_context()) {
                            llama_flex_set_prefetch_budget(*model_tgt->get_flex_context(), 0);
                        }
                        continue;
                    }
                    const uint64_t dense_min_grant = flex_stats.slot_bytes > 0
                        ? (uint64_t) flex_stats.slot_bytes
                        : (uint64_t) flex_stats.stream_per_token;
                    const uint64_t dense_ahead = (uint64_t) std::max(1, flex_stats.effective_ahead);
                    const uint64_t dense_window_grant =
                        memory_governor_saturating_mul(dense_min_grant, dense_ahead);
                    const uint64_t dense_target_grant =
                        effective_pressure_state == kv_pressure_state::PRESSURE
                            ? std::max<uint64_t>(candidate.bytes, dense_window_grant)
                            : std::max<uint64_t>(dense_min_grant, std::min<uint64_t>(
                                candidate.bytes,
                                dense_window_grant));
                    if (dense_min_grant > 0 &&
                            memory_governor_prefetch_budget_available_bytes >= dense_min_grant) {
                        const uint64_t grant = std::min<uint64_t>(
                                dense_target_grant,
                                memory_governor_prefetch_budget_available_bytes);
                        prefetch_budget_dense_bytes = grant;
                        memory_governor_prefetch_budget_available_bytes -= grant;
                        if (flex_enabled && model_tgt && model_tgt->get_flex_context()) {
                            llama_flex_set_prefetch_budget(*model_tgt->get_flex_context(), grant);
                        }
                    }
                } else if (std::string(candidate.kind) == "moe_expert" &&
                        std::string(candidate.action) == "prefetch" &&
                        prefetch_budget_moe_bytes == 0) {
                    const uint64_t grant = std::min<uint64_t>(
                            candidate.bytes,
                            memory_governor_prefetch_budget_available_bytes);
                    prefetch_budget_moe_bytes = grant;
                    memory_governor_prefetch_budget_available_bytes -= grant;
                    if (moe_enabled && model_tgt && model_tgt->get_moe_buffer_context()) {
                        llama_moe_buffer_set_prefetch_budget(
                                *model_tgt->get_moe_buffer_context(), grant);
                    }
                }
            }
            if (prefetch_budget_dense_bytes == 0 &&
                    flex_enabled && model_tgt && model_tgt->get_flex_context()) {
                llama_flex_set_prefetch_budget(*model_tgt->get_flex_context(), 0);
            }
            if (prefetch_budget_moe_bytes == 0 &&
                    moe_enabled && model_tgt && model_tgt->get_moe_buffer_context()) {
                llama_moe_buffer_set_prefetch_budget(
                        *model_tgt->get_moe_buffer_context(), 1);
            }
            prefetch_budget_remaining_bytes =
                memory_governor_prefetch_budget_available_bytes;
        } else {
            if (flex_enabled && model_tgt && model_tgt->get_flex_context()) {
                llama_flex_set_prefetch_budget(*model_tgt->get_flex_context(), 0);
            }
            if (moe_enabled && model_tgt && model_tgt->get_moe_buffer_context()) {
                llama_moe_buffer_set_prefetch_budget(
                        *model_tgt->get_moe_buffer_context(), 0);
            }
        }

        memory_governor_clean_reclaim_result clean_result;
        clean_result.reason = memory_governor_clean_reclaim_enabled ? "no_candidate" : "disabled";
        if (memory_governor_clean_reclaim_enabled) {
            if (effective_pressure_state != kv_pressure_state::PRESSURE &&
                    effective_pressure_state != kv_pressure_state::CRITICAL) {
                clean_result.reason = "not_pressure";
            } else {
                uint64_t target = memory_governor_clean_reclaim_target_bytes;
                if (target == 0) {
                    target = pressure_excess_bytes;
                }
                if (target == 0) {
                    clean_result.reason = "zero_target";
                } else if (memory_governor_clean_reclaim_ranked_enabled) {
                    std::vector<memory_governor_would_candidate> sorted_clean;
                    for (const auto & candidate : reclaim_candidates) {
                        if (memory_governor_is_clean_reclaim_candidate(candidate) &&
                                memory_governor_candidate_has_positive_roi(candidate) &&
                                memory_governor_candidate_allowed_in_state(candidate, effective_pressure_state)) {
                            sorted_clean.push_back(candidate);
                        }
                    }
                    std::sort(sorted_clean.begin(), sorted_clean.end(), memory_governor_candidate_before);
                    if (sorted_clean.empty()) {
                        clean_result.reason = "no_positive_candidate";
                    } else {
                        uint64_t remaining = target;
                        clean_result.reason = "ranked_submitted";
                        for (const auto & selected : sorted_clean) {
                            if (remaining == 0 ||
                                    clean_result.passes >= memory_governor_clean_reclaim_max_passes) {
                                break;
                            }
                            const uint64_t pass_target = std::min<uint64_t>(remaining, selected.bytes);
                            if (pass_target == 0) {
                                continue;
                            }
                            clean_result.candidates_tried++;
                            if (!clean_result.attempted) {
                                clean_result.kind = selected.kind;
                                clean_result.score = selected.score;
                            }
                            clean_result.attempted = true;
                            clean_result.target_bytes += pass_target;
                            clean_result.passes++;
                            if (memory_governor_async_actions_enabled) {
                                memory_governor_async_action action;
                                if (std::string(selected.kind) == "dense_layer") {
                                    action.kind = memory_governor_async_action_kind::dense_clean_reclaim;
                                } else if (std::string(selected.kind) == "moe_expert") {
                                    action.kind = memory_governor_async_action_kind::moe_clean_reclaim;
                                } else {
                                    clean_result.reason = "context_unavailable";
                                    continue;
                                }
                                action.score = selected.score;
                                action.roi = selected.roi;
                                action.target_bytes = pass_target;
                                action.max_objects = memory_governor_clean_reclaim_max_objects;
                                action.sample_count = sample_count;
                                action.pressure_state = effective_pressure_state;
                                const bool enqueued = memory_governor_async_submit(action);
                                clean_result.reason =
                                    enqueued ? "async_enqueued" : "async_queue_full";
                                if (enqueued) {
                                    remaining = memory_governor_saturating_sub(
                                            remaining, pass_target);
                                }
                            } else if (std::string(selected.kind) == "dense_layer" &&
                                    flex_enabled && model_tgt && model_tgt->get_flex_context()) {
                                const auto result = llama_flex_reclaim_released(
                                        *model_tgt->get_flex_context(),
                                        pass_target,
                                        memory_governor_clean_reclaim_max_objects);
                                clean_result.released_bytes += result.released_bytes;
                                clean_result.released_objects += result.released_layers;
                                remaining = memory_governor_saturating_sub(
                                        remaining, result.released_bytes);
                            } else if (std::string(selected.kind) == "moe_expert" &&
                                    moe_enabled && model_tgt && model_tgt->get_moe_buffer_context()) {
                                const auto result = llama_moe_buffer_reclaim_clean(
                                        *model_tgt->get_moe_buffer_context(),
                                        pass_target,
                                        memory_governor_clean_reclaim_max_objects);
                                clean_result.released_bytes += result.released_bytes;
                                clean_result.released_objects += result.released_groups;
                                remaining = memory_governor_saturating_sub(
                                        remaining, result.released_bytes);
                            } else {
                                clean_result.reason = "context_unavailable";
                            }
                        }
                        clean_result.target_satisfied = remaining == 0;
                        if (!clean_result.attempted) {
                            clean_result.reason = "no_candidate";
                        }
                    }
                } else {
                    const char * clean_auction_reason = "no_candidate";
                    memory_governor_would_candidate selected;
                    if (flex_enabled) {
                        bool has_dense_clean = false;
                        for (const auto & candidate : reclaim_candidates) {
                            if (std::string(candidate.kind) != "dense_layer" ||
                                    !memory_governor_is_clean_reclaim_candidate(candidate) ||
                                    !memory_governor_candidate_has_positive_roi(candidate) ||
                                    !memory_governor_candidate_allowed_in_state(
                                        candidate, effective_pressure_state)) {
                                continue;
                            }
                            if (!has_dense_clean ||
                                    memory_governor_candidate_before(candidate, selected)) {
                                selected = candidate;
                                has_dense_clean = true;
                            }
                        }
                        if (has_dense_clean) {
                            clean_auction_reason = "dense_clean_first";
                        }
                    }
                    if (!memory_governor_is_clean_reclaim_candidate(selected)) {
                        selected = memory_governor_auction_select(
                                reclaim_candidates,
                                effective_pressure_state,
                                [](const memory_governor_would_candidate & candidate) {
                                    return memory_governor_is_clean_reclaim_candidate(candidate) &&
                                        memory_governor_candidate_has_positive_roi(candidate);
                                },
                                &clean_auction_reason);
                    }
                    if (memory_governor_is_clean_reclaim_candidate(selected)) {
                        target = std::min<uint64_t>(target, selected.bytes);
                        clean_result.attempted = true;
                        clean_result.kind = selected.kind;
                        clean_result.score = selected.score;
                        clean_result.target_bytes = target;
                        clean_result.passes = 1;
                        clean_result.candidates_tried = 1;
                        clean_result.reason = clean_auction_reason;
                        if (memory_governor_async_actions_enabled) {
                            memory_governor_async_action action;
                            bool async_clean_kind_valid = true;
                            if (std::string(selected.kind) == "dense_layer") {
                                action.kind = memory_governor_async_action_kind::dense_clean_reclaim;
                            } else if (std::string(selected.kind) == "moe_expert") {
                                action.kind = memory_governor_async_action_kind::moe_clean_reclaim;
                            } else {
                                async_clean_kind_valid = false;
                                clean_result.reason = "context_unavailable";
                            }
                            if (async_clean_kind_valid) {
                                action.score = selected.score;
                                action.roi = selected.roi;
                                action.target_bytes = target;
                                action.max_objects = memory_governor_clean_reclaim_max_objects;
                                action.sample_count = sample_count;
                                action.pressure_state = effective_pressure_state;
                                clean_result.reason =
                                    memory_governor_async_submit(action)
                                        ? "async_enqueued"
                                        : "async_queue_full";
                            }
                        } else if (std::string(selected.kind) == "dense_layer" &&
                                flex_enabled && model_tgt && model_tgt->get_flex_context()) {
                            const auto result = llama_flex_reclaim_released(
                                    *model_tgt->get_flex_context(),
                                    target,
                                    memory_governor_clean_reclaim_max_objects);
                            clean_result.released_bytes = result.released_bytes;
                            clean_result.released_objects = result.released_layers;
                            clean_result.target_satisfied = result.target_satisfied;
                        } else if (std::string(selected.kind) == "moe_expert" &&
                                moe_enabled && model_tgt && model_tgt->get_moe_buffer_context()) {
                            const auto result = llama_moe_buffer_reclaim_clean(
                                    *model_tgt->get_moe_buffer_context(),
                                    target,
                                    memory_governor_clean_reclaim_max_objects);
                            clean_result.released_bytes = result.released_bytes;
                            clean_result.released_objects = result.released_groups;
                            clean_result.target_satisfied = result.target_satisfied;
                        } else {
                            clean_result.reason = "context_unavailable";
                        }
                    } else {
                        clean_result.reason = clean_auction_reason;
                    }
                }
            }
        }

        memory_governor_kv_release_result kv_release_result;
        kv_release_result.reason = memory_governor_kv_release_enabled ? "no_candidate" : "disabled";
        kv_release_result.max_blocks = memory_governor_kv_release_max_blocks;
        if (memory_governor_kv_release_enabled) {
            const char * release_auction_reason = "no_candidate";
            auto selected = memory_governor_auction_select(
                    reclaim_candidates,
                    effective_pressure_state,
                    memory_governor_is_kv_release_candidate,
                    &release_auction_reason);
            if (!memory_governor_is_kv_release_candidate(selected)) {
                kv_release_result.reason = release_auction_reason;
            } else if (!ctx_tgt || !llama_get_memory(ctx_tgt)) {
                kv_release_result.reason = "no_memory";
            } else if (!telemetry || !telemetry->sample_valid || telemetry->stale) {
                kv_release_result.reason = "stale";
            } else if (effective_pressure_state != kv_pressure_state::PRESSURE &&
                    effective_pressure_state != kv_pressure_state::CRITICAL) {
                kv_release_result.reason = "not_pressure";
            } else if (sample_count < memory_governor_kv_release_next_sample) {
                kv_release_result.reason = "cooldown";
            } else {
                uint64_t pressure_after_clean = pressure_excess_bytes;
                if (clean_result.released_bytes >= pressure_after_clean) {
                    pressure_after_clean = 0;
                } else {
                    pressure_after_clean -= clean_result.released_bytes;
                }
                uint64_t target = memory_governor_kv_release_target_bytes;
                if (target == 0) {
                    target = pressure_after_clean;
                    if (memory_governor_kv_soft_budget_enabled) {
                        const uint64_t kv_soft_after_clean =
                            memory_governor_saturating_sub(
                                    kv_soft_excess_bytes, clean_result.released_bytes);
                        target = std::max<uint64_t>(target, kv_soft_after_clean);
                        kv_soft_release_target_bytes = target;
                    }
                }
                if (target == 0) {
                    kv_release_result.reason = "zero_target";
                } else {
                    target = std::min<uint64_t>(target, selected.bytes);
                    if (memory_governor_async_actions_enabled) {
                        memory_governor_async_action action;
                        action.kind = memory_governor_async_action_kind::kv_global_release;
                        action.score = selected.score;
                        action.roi = selected.roi;
                        action.target_bytes = target;
                        action.max_blocks = memory_governor_kv_release_max_blocks;
                        action.sample_count = sample_count;
                        action.pressure_state = effective_pressure_state;
                        kv_release_result.attempted = true;
                        kv_release_result.score = selected.score;
                        kv_release_result.target_bytes = target;
                        kv_release_result.reason =
                            memory_governor_async_submit(action)
                                ? "async_enqueued"
                                : "async_queue_full";
                        if (std::string(kv_release_result.reason) == "async_enqueued") {
                            memory_governor_kv_release_next_sample =
                                sample_count + std::max<uint32_t>(
                                        memory_governor_kv_release_cooldown_samples, 1);
                        }
                    } else {
                        auto * mem = llama_get_memory(ctx_tgt);
                        const uint64_t decision_id = next_kv_decision_id();
                        kv_release_result.evaluate_attempted = true;
                        const auto evaluation = mem->execute_action({
                                llama_kv_action::evaluate,
                                decision_id,
                                -1,
                                0,
                                0,
                                false,
                                false,
                                llama_kv_memory_claimant::kv,
                                llama_kv_io_class::background_write,
                                0,
                                0,
                        });
                        if (evaluation.decision_id != decision_id) {
                            kv_release_result.reason = "decision_mismatch";
                        } else if (evaluation.capability.context_invalid) {
                            kv_release_result.reason = "context_invalid";
                        } else if (evaluation.capability.write_transaction_open) {
                            kv_release_result.reason = "write_transaction_open";
                        } else if (evaluation.fail_stop) {
                            kv_release_result.reason = "fail_stop";
                            kv_release_result.fail_stop = true;
                        } else if (!evaluation.capability.can_release) {
                            kv_release_result.reason = "release_unsupported";
                        } else if (evaluation.outcome != llama_kv_action_outcome::completed ||
                                evaluation.reason != llama_kv_action_reason::none) {
                            kv_release_result.reason = "evaluate_rejected";
                        } else {
                            kv_release_result.attempted = true;
                            kv_release_result.score = selected.score;
                            kv_release_result.target_bytes = target;
                            const auto release = mem->execute_action({
                                    llama_kv_action::release,
                                    decision_id,
                                    -1,
                                    target,
                                    memory_governor_kv_release_max_blocks,
                                    false,
                                    false,
                                    llama_kv_memory_claimant::kv,
                                    llama_kv_io_class::background_write,
                                    0,
                                    target,
                            });
                            kv_release_result.bytes = release.bytes;
                            kv_release_result.blocks = release.blocks;
                            kv_release_result.relieved_bytes = release.relieved_bytes;
                            kv_release_result.shortfall_bytes = release.shortfall_bytes;
                            kv_release_result.state_changed = release.state_changed;
                            kv_release_result.io_failure = release.io_failure;
                            kv_release_result.fail_stop = release.fail_stop;
                            kv_release_result.io_errno = release.io_errno;
                            kv_release_result.outcome =
                                memory_governor_action_outcome_name(release.outcome);
                            kv_release_result.action_reason =
                                memory_governor_action_reason_name(release.reason);
                            kv_release_result.reason = "release_submitted";
                            memory_governor_kv_release_next_sample =
                                sample_count + std::max<uint32_t>(
                                        memory_governor_kv_release_cooldown_samples, 1);
                        }
                    }
                }
            }
        }

        memory_governor_kv_offload_result kv_offload_result;
        kv_offload_result.reason = memory_governor_kv_offload_enabled ? "no_candidate" : "disabled";
        kv_offload_result.max_blocks = memory_governor_kv_offload_max_blocks;
        if (memory_governor_kv_offload_enabled) {
            const char * offload_auction_reason = "no_candidate";
            auto selected = memory_governor_auction_select(
                    reclaim_candidates,
                    effective_pressure_state,
                    memory_governor_is_kv_offload_candidate,
                    &offload_auction_reason);
            const bool slot_state_offload_fallback =
                !kv_budget.valid && kv_slot_budget_valid;
            uint64_t pressure_after_reclaim = pressure_excess_bytes;
            pressure_after_reclaim = memory_governor_saturating_sub(
                    pressure_after_reclaim, clean_result.released_bytes);
            pressure_after_reclaim = memory_governor_saturating_sub(
                    pressure_after_reclaim, kv_release_result.relieved_bytes);
            uint64_t kv_soft_after_reclaim = kv_soft_excess_bytes;
            kv_soft_after_reclaim = memory_governor_saturating_sub(
                    kv_soft_after_reclaim, clean_result.released_bytes);
            kv_soft_after_reclaim = memory_governor_saturating_sub(
                    kv_soft_after_reclaim, kv_release_result.relieved_bytes);
            const bool kv_soft_offload_needed =
                memory_governor_kv_soft_budget_enabled && kv_soft_after_reclaim > 0;
            const bool slot_state_offload_needed =
                slot_state_offload_fallback &&
                (pressure_after_reclaim > 0 || kv_soft_offload_needed);
            if (!memory_governor_kv_release_enabled && !slot_state_offload_fallback) {
                kv_offload_result.reason = "release_disabled";
            } else if (!memory_governor_is_kv_offload_candidate(selected)) {
                kv_offload_result.reason = offload_auction_reason;
            } else if (!ctx_tgt || !llama_get_memory(ctx_tgt)) {
                kv_offload_result.reason = "no_memory";
            } else if (!telemetry || !telemetry->sample_valid || telemetry->stale) {
                kv_offload_result.reason = "stale";
            } else if (effective_pressure_state != kv_pressure_state::PRESSURE &&
                    effective_pressure_state != kv_pressure_state::CRITICAL &&
                    !slot_state_offload_needed) {
                kv_offload_result.reason = "not_pressure";
            } else if (std::string(kv_release_result.reason) == "cooldown") {
                kv_offload_result.reason = "release_cooldown";
            } else if (sample_count < memory_governor_kv_offload_next_sample) {
                kv_offload_result.reason = "cooldown";
            } else if (pressure_after_reclaim == 0 &&
                    !kv_soft_offload_needed &&
                    !slot_state_offload_needed) {
                kv_offload_result.reason = "pressure_satisfied";
            } else {
                uint64_t target = memory_governor_kv_offload_target_bytes;
                if (target == 0) {
                    target = pressure_after_reclaim;
                    if (memory_governor_kv_soft_budget_enabled) {
                        target = std::max<uint64_t>(target, kv_soft_after_reclaim);
                        kv_soft_offload_target_bytes = target;
                    }
                    if (target == 0 && slot_state_offload_needed) {
                        target = selected.bytes;
                    }
                }
                if (target == 0) {
                    kv_offload_result.reason = "zero_target";
                } else if (slot_state_offload_fallback) {
                    target = std::min<uint64_t>(target, selected.bytes);
                    const auto offload =
                        memory_governor_slot_state_offload(selected.id, target);
                    kv_offload_result.attempted = offload.attempted;
                    kv_offload_result.seq_id = selected.id;
                    kv_offload_result.score = selected.score;
                    kv_offload_result.target_bytes = target;
                    kv_offload_result.bytes = offload.bytes_before;
                    kv_offload_result.relieved_bytes = offload.relieved_bytes;
                    kv_offload_result.shortfall_bytes =
                        target > offload.relieved_bytes ? target - offload.relieved_bytes : 0;
                    kv_offload_result.state_changed = offload.state_changed;
                    kv_offload_result.backend = "slot_state";
                    kv_offload_result.reason = offload.reason;
                    kv_offload_result.outcome =
                        offload.state_changed ? "completed" : "deferred";
                    kv_offload_result.action_reason =
                        offload.state_changed ? "none" : offload.reason;
                    if (offload.attempted) {
                        memory_governor_kv_offload_next_sample =
                            sample_count + std::max<uint32_t>(
                                    memory_governor_kv_offload_cooldown_samples, 1);
                    }
                } else if (memory_governor_async_actions_enabled) {
                    target = std::min<uint64_t>(target, selected.bytes);
                    memory_governor_async_action action;
                    action.kind = memory_governor_async_action_kind::kv_sequence_offload;
                    action.seq_id = selected.id;
                    action.score = selected.score;
                    action.roi = selected.roi;
                    action.target_bytes = target;
                    action.max_blocks = memory_governor_kv_offload_max_blocks;
                    action.sample_count = sample_count;
                    action.pressure_state = effective_pressure_state;
                    kv_offload_result.attempted = true;
                    kv_offload_result.seq_id = selected.id;
                    kv_offload_result.score = selected.score;
                    kv_offload_result.target_bytes = target;
                    kv_offload_result.bytes = selected.bytes;
                    kv_offload_result.backend = "memory_backend";
                    kv_offload_result.reason =
                        memory_governor_async_submit(action)
                            ? "async_enqueued"
                            : "async_queue_full";
                    kv_offload_result.outcome = "enqueued";
                    if (std::string(kv_offload_result.reason) == "async_enqueued") {
                        memory_governor_kv_offload_next_sample =
                            sample_count + std::max<uint32_t>(
                                    memory_governor_kv_offload_cooldown_samples, 1);
                    }
                } else {
                    target = std::min<uint64_t>(target, selected.bytes);
                    auto * mem = llama_get_memory(ctx_tgt);
                    const uint64_t decision_id = next_kv_decision_id();
                    kv_offload_result.evaluate_attempted = true;
                    const auto evaluation = mem->execute_action({
                            llama_kv_action::evaluate,
                            decision_id,
                            -1,
                            0,
                            0,
                            false,
                            false,
                            llama_kv_memory_claimant::kv,
                            llama_kv_io_class::capacity_write,
                            0,
                            0,
                    });
                    if (evaluation.decision_id != decision_id) {
                        kv_offload_result.reason = "decision_mismatch";
                    } else if (evaluation.capability.context_invalid) {
                        kv_offload_result.reason = "context_invalid";
                    } else if (evaluation.capability.write_transaction_open) {
                        kv_offload_result.reason = "write_transaction_open";
                    } else if (evaluation.fail_stop) {
                        kv_offload_result.reason = "fail_stop";
                        kv_offload_result.fail_stop = true;
                    } else if (!evaluation.capability.can_offload) {
                        kv_offload_result.reason = "offload_unsupported";
                    } else if (evaluation.outcome != llama_kv_action_outcome::completed ||
                            evaluation.reason != llama_kv_action_reason::none) {
                        kv_offload_result.reason = "evaluate_rejected";
                    } else {
                        const int64_t bounded_priority = std::max<int64_t>(
                                std::numeric_limits<int32_t>::min(),
                                std::min<int64_t>(
                                        std::numeric_limits<int32_t>::max(),
                                        selected.score));
                        kv_offload_result.attempted = true;
                        kv_offload_result.seq_id = selected.id;
                        kv_offload_result.score = selected.score;
                        kv_offload_result.target_bytes = target;
                        kv_offload_result.backend = "memory_backend";
                        const auto offload = mem->execute_action({
                                llama_kv_action::offload,
                                decision_id,
                                selected.id,
                                target,
                                memory_governor_kv_offload_max_blocks,
                                false,
                                false,
                                llama_kv_memory_claimant::kv,
                                llama_kv_io_class::capacity_write,
                                (int32_t) bounded_priority,
                                target,
                        });
                        kv_offload_result.bytes = offload.bytes;
                        kv_offload_result.blocks = offload.blocks;
                        kv_offload_result.relieved_bytes = offload.relieved_bytes;
                        kv_offload_result.shortfall_bytes = offload.shortfall_bytes;
                        kv_offload_result.state_changed = offload.state_changed;
                        kv_offload_result.io_failure = offload.io_failure;
                        kv_offload_result.fail_stop = offload.fail_stop;
                        kv_offload_result.io_errno = offload.io_errno;
                        kv_offload_result.outcome =
                            memory_governor_action_outcome_name(offload.outcome);
                        kv_offload_result.action_reason =
                            memory_governor_action_reason_name(offload.reason);
                        kv_offload_result.reason = "offload_submitted";
                        memory_governor_kv_offload_next_sample =
                            sample_count + std::max<uint32_t>(
                                    memory_governor_kv_offload_cooldown_samples, 1);
                    }
                }
            }
        }

        uint64_t reallocation_credit_earned_bytes = 0;
        uint64_t reallocation_pending_added_bytes = 0;
        uint64_t reallocation_pending_before_confirm_bytes = memory_governor_reallocation_pending_bytes;
        uint64_t reallocation_pending_remaining_bytes = memory_governor_reallocation_pending_bytes;
        uint64_t reallocation_observed_drop_bytes = 0;
        uint64_t reallocation_confirm_limit_bytes = 0;
        uint64_t reallocation_credit_decayed_bytes = 0;
        uint64_t reallocation_credit_available_bytes = memory_governor_reallocation_credit_bytes;
        uint64_t reallocation_credit_spent_bytes = 0;
        uint64_t reallocation_dense_grant_bytes = 0;
        uint64_t reallocation_moe_grant_bytes = 0;
        uint64_t reallocation_moe_old_budget_bytes = moe_stats.budget_bytes;
        uint64_t reallocation_moe_new_budget_bytes = moe_stats.budget_bytes;
        const uint64_t effective_reallocation_unit_bytes =
            flex_stats.slot_bytes > 0
                ? (uint64_t) flex_stats.slot_bytes
                : (effective_moe_budget_grow_bytes > 0
                    ? effective_moe_budget_grow_bytes
                    : effective_hard_headroom_bytes);
        const uint64_t effective_reallocation_credit_cap_bytes =
            memory_governor_reallocation_credit_cap_bytes > 0
                ? memory_governor_reallocation_credit_cap_bytes
                : effective_reallocation_unit_bytes;
        const uint64_t effective_reallocation_max_grant_bytes =
            memory_governor_reallocation_max_grant_bytes > 0
                ? memory_governor_reallocation_max_grant_bytes
                : effective_reallocation_credit_cap_bytes;
        const uint64_t effective_reallocation_min_grant_bytes =
            memory_governor_reallocation_min_grant_bytes > 0
                ? memory_governor_reallocation_min_grant_bytes
                : effective_reallocation_unit_bytes;
        const uint64_t effective_reallocation_hard_guard_bytes =
            memory_governor_reallocation_hard_guard_bytes > 0
                ? memory_governor_reallocation_hard_guard_bytes
                : effective_hard_headroom_bytes;
        const uint64_t effective_reallocation_confirm_slack_bytes =
            memory_governor_reallocation_confirm_slack_bytes > 0
                ? memory_governor_reallocation_confirm_slack_bytes
                : effective_hard_headroom_bytes;
        const char * reallocation_best_kind =
            memory_governor_reallocation_enabled ? "none" : "disabled";
        const char * reallocation_reason =
            memory_governor_reallocation_enabled ? "no_credit" : "disabled";
        if (memory_governor_reallocation_enabled) {
            const uint64_t raw_relieved =
                clean_result.released_bytes +
                kv_release_result.relieved_bytes +
                kv_offload_result.relieved_bytes +
                moe_budget_reclaim_released_bytes +
                async_relieved_drained_bytes;
            double discount = memory_governor_reallocation_normal_discount;
            if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                discount = 0.0;
            } else if (effective_pressure_state == kv_pressure_state::PRESSURE) {
                discount = memory_governor_reallocation_pressure_discount;
            }
            reallocation_pending_added_bytes = (uint64_t) ((double) raw_relieved * discount);
            if (reallocation_pending_added_bytes > 0) {
                memory_governor_reallocation_pending_bytes = std::min<uint64_t>(
                        effective_reallocation_credit_cap_bytes,
                        memory_governor_reallocation_pending_bytes +
                            reallocation_pending_added_bytes);
            }
            reallocation_pending_before_confirm_bytes =
                memory_governor_reallocation_pending_bytes;
            if (memory_governor_reallocation_last_current_valid &&
                    memory_governor_reallocation_last_current_bytes > pressure_current_bytes) {
                reallocation_observed_drop_bytes =
                    memory_governor_reallocation_last_current_bytes - pressure_current_bytes;
            }
            if (memory_governor_reallocation_confirm_enabled) {
                reallocation_confirm_limit_bytes =
                    reallocation_observed_drop_bytes +
                    effective_reallocation_confirm_slack_bytes;
                reallocation_credit_earned_bytes = std::min<uint64_t>(
                        memory_governor_reallocation_pending_bytes,
                        reallocation_confirm_limit_bytes);
                memory_governor_reallocation_pending_bytes =
                    memory_governor_saturating_sub(
                            memory_governor_reallocation_pending_bytes,
                            reallocation_credit_earned_bytes);
            } else {
                reallocation_confirm_limit_bytes = memory_governor_reallocation_pending_bytes;
                reallocation_credit_earned_bytes =
                    memory_governor_reallocation_pending_bytes;
                memory_governor_reallocation_pending_bytes = 0;
            }
            reallocation_pending_remaining_bytes =
                memory_governor_reallocation_pending_bytes;
            reallocation_credit_decayed_bytes =
                (uint64_t) ((double) memory_governor_reallocation_credit_bytes *
                        memory_governor_reallocation_decay);
            memory_governor_reallocation_credit_bytes = std::min<uint64_t>(
                    effective_reallocation_credit_cap_bytes,
                    reallocation_credit_decayed_bytes + reallocation_credit_earned_bytes);
            reallocation_credit_available_bytes = memory_governor_reallocation_credit_bytes;

            if (!memory_governor_reallocation_apply_moe) {
                reallocation_reason = "apply_moe_disabled";
            } else if (!memory_governor_moe_budget_dynamic_enabled || !moe_enabled || !moe_context) {
                if (flex_enabled && effective_pressure_state != kv_pressure_state::CRITICAL) {
                    const uint64_t dense_slot_bytes = flex_stats.slot_bytes > 0
                        ? (uint64_t) flex_stats.slot_bytes
                        : (uint64_t) flex_stats.stream_per_token;
                    const uint64_t dense_window_bytes =
                        memory_governor_saturating_mul(
                                dense_slot_bytes,
                                (uint64_t) std::max(1, flex_stats.effective_ahead));
                    const uint64_t grant = std::min<uint64_t>(
                            std::min<uint64_t>(
                                memory_governor_reallocation_credit_bytes,
                                effective_reallocation_max_grant_bytes),
                            dense_window_bytes);
                    if (grant >= effective_reallocation_min_grant_bytes) {
                        reallocation_dense_grant_bytes = grant;
                        memory_governor_reallocation_credit_bytes -= grant;
                        reallocation_credit_spent_bytes = grant;
                        reallocation_best_kind = "dense_prefetch_window";
                        reallocation_reason = "dense_credit_accounted";
                    } else {
                        reallocation_reason = "dense_grant_below_min";
                    }
                } else {
                    reallocation_reason = "moe_unavailable";
                }
            } else if (effective_pressure_state == kv_pressure_state::CRITICAL) {
                reallocation_reason = "critical";
            } else if ((moe_budget_delta_evictions > 0 ||
                        moe_budget_delta_bytes_read > 0 ||
                        moe_budget_delta_prefetch_late > 0) &&
                    (moe_stats.budget_unbounded ||
                     moe_stats.warm_working_set_bytes > moe_stats.budget_bytes) &&
                    moe_budget_headroom_bytes > effective_reallocation_hard_guard_bytes) {
                uint64_t max_budget = memory_governor_moe_budget_max_bytes;
                if (max_budget == 0 || max_budget > moe_stats.expert_bytes) {
                    max_budget = moe_stats.expert_bytes;
                }
                if (moe_stats.planner_floor_bytes > 0) {
                    max_budget = std::min<uint64_t>(
                            max_budget,
                            (uint64_t) moe_stats.planner_safe_budget_bytes);
                }
                const uint64_t current_budget =
                    moe_stats.budget_unbounded ? max_budget : (uint64_t) moe_stats.budget_bytes;
                reallocation_moe_old_budget_bytes = current_budget;
                const uint64_t guard_headroom =
                    effective_reallocation_hard_guard_bytes == 0
                        ? moe_budget_headroom_bytes
                        : moe_budget_headroom_bytes - effective_reallocation_hard_guard_bytes;
                const uint64_t warm_deficit =
                    (uint64_t) moe_stats.warm_working_set_bytes > current_budget
                        ? (uint64_t) moe_stats.warm_working_set_bytes - current_budget
                        : 0;
                uint64_t grant = std::min<uint64_t>(warm_deficit, guard_headroom);
                grant = std::min<uint64_t>(grant, max_budget > current_budget ? max_budget - current_budget : 0);
                if (grant == 0) {
                    reallocation_reason = "emergency_no_headroom";
                } else {
                    const uint64_t next_budget = current_budget + grant;
                    llama_moe_buffer_set_budget(moe_context, (size_t) next_budget);
                    reallocation_moe_grant_bytes = grant;
                    reallocation_moe_new_budget_bytes = next_budget;
                    reallocation_best_kind = "moe_resident";
                    reallocation_reason = "moe_emergency_working_set";
                    moe_stats = llama_moe_buffer_get_stats(*moe_context);
                }
            } else if (memory_governor_reallocation_credit_bytes <
                    effective_reallocation_min_grant_bytes) {
                reallocation_reason = "below_min_grant";
            } else {
                uint64_t max_budget = memory_governor_moe_budget_max_bytes;
                if (max_budget == 0 || max_budget > moe_stats.expert_bytes) {
                    max_budget = moe_stats.expert_bytes;
                }
                if (moe_stats.planner_floor_bytes > 0) {
                    max_budget = std::min<uint64_t>(
                            max_budget,
                            (uint64_t) moe_stats.planner_safe_budget_bytes);
                }
                const uint64_t current_budget =
                    moe_stats.budget_unbounded ? max_budget : (uint64_t) moe_stats.budget_bytes;
                reallocation_moe_old_budget_bytes = current_budget;
                if (current_budget >= max_budget) {
                    reallocation_reason = "moe_max";
                } else if (effective_reallocation_hard_guard_bytes > 0 &&
                        moe_budget_headroom_bytes <= effective_reallocation_hard_guard_bytes) {
                    reallocation_reason = "hard_guard";
                } else {
                    const uint64_t guard_headroom =
                        effective_reallocation_hard_guard_bytes == 0
                            ? memory_governor_reallocation_credit_bytes
                            : moe_budget_headroom_bytes - effective_reallocation_hard_guard_bytes;
                    uint64_t grant = std::min<uint64_t>(
                            memory_governor_reallocation_credit_bytes,
                            effective_reallocation_max_grant_bytes);
                    grant = std::min<uint64_t>(grant, guard_headroom);
                    grant = std::min<uint64_t>(grant, max_budget - current_budget);
                    if (grant < effective_reallocation_min_grant_bytes) {
                        reallocation_reason = "grant_below_min";
                    } else {
                        const uint64_t next_budget = current_budget + grant;
                        llama_moe_buffer_set_budget(moe_context, (size_t) next_budget);
                        memory_governor_reallocation_credit_bytes -= grant;
                        reallocation_credit_spent_bytes = grant;
                        reallocation_moe_grant_bytes = grant;
                        reallocation_moe_new_budget_bytes = next_budget;
                        reallocation_best_kind = "moe_resident";
                        reallocation_reason = "moe_credit_grant";
                        moe_stats = llama_moe_buffer_get_stats(*moe_context);
                    }
                }
            }
            memory_governor_reallocation_last_current_bytes = pressure_current_bytes;
            memory_governor_reallocation_last_current_valid = true;
        }

        llama_flex_delta_pin_result dense_repin_result;
        dense_repin_result.reason = memory_governor_dense_repin_enabled ? "not_attempted" : "disabled";
        uint64_t dense_repin_grant_bytes = 0;
        uint64_t dense_repin_headroom_bytes = 0;
        if (memory_governor_dense_repin_enabled &&
                flex_enabled &&
                model_tgt &&
                model_tgt->get_flex_context()) {
            const auto cgroup = memory_governor_read_cgroup_memory();
            dense_repin_headroom_bytes =
                cgroup.valid && cgroup.max_bytes > pressure_current_bytes
                    ? cgroup.max_bytes - pressure_current_bytes
                    : 0;
            const uint64_t safe_headroom_grant =
                dense_repin_headroom_bytes > effective_dense_repin_headroom_bytes
                    ? dense_repin_headroom_bytes - effective_dense_repin_headroom_bytes
                    : 0;
            const uint64_t cap_remaining =
                effective_dense_repin_max_bytes > flex_stats.delta_locked_bytes
                    ? effective_dense_repin_max_bytes - flex_stats.delta_locked_bytes
                    : 0;
            dense_repin_grant_bytes = std::min<uint64_t>(
                    effective_dense_repin_step_bytes,
                    std::min<uint64_t>(cap_remaining,
                        std::max<uint64_t>(safe_headroom_grant, reallocation_dense_grant_bytes)));
            if (effective_pressure_state != kv_pressure_state::NORMAL) {
                dense_repin_result.reason = "not_normal";
                dense_repin_grant_bytes = 0;
            } else if (memory_governor_dense_repin_idle_only && !idle) {
                dense_repin_result.reason = "busy";
                dense_repin_grant_bytes = 0;
            } else if (sample_count < memory_governor_dense_repin_next_sample) {
                dense_repin_result.reason = "cooldown";
                dense_repin_grant_bytes = 0;
            } else if (cap_remaining == 0) {
                dense_repin_result.reason = "cap_reached";
            } else if (dense_repin_headroom_bytes <= effective_dense_repin_headroom_bytes) {
                dense_repin_result.reason = "headroom_guard";
            } else if (dense_repin_grant_bytes < effective_reallocation_min_grant_bytes) {
                dense_repin_result.reason = "grant_below_min";
            } else {
                dense_repin_result = memory_governor_dense_repin_async_enabled
                    ? llama_flex_delta_pin_async(
                            *model_tgt->get_flex_context(),
                            dense_repin_grant_bytes,
                            memory_governor_dense_repin_min_roi)
                    : llama_flex_delta_pin(
                            *model_tgt->get_flex_context(),
                            dense_repin_grant_bytes,
                            memory_governor_dense_repin_min_roi);
                flex_stats = llama_flex_get_stats(*model_tgt->get_flex_context());
                if (std::string(dense_repin_result.reason) == "async_enqueued" ||
                        dense_repin_result.changed) {
                    memory_governor_dense_repin_next_sample =
                        sample_count + memory_governor_dense_repin_cooldown_samples;
                }
            }
        }

        const std::string would_reclaim =
            memory_governor_format_candidates(reclaim_candidates, 3);
        const std::string would_prefetch =
            memory_governor_format_candidates(prefetch_candidates, 3);
        std::vector<memory_governor_would_candidate> auction_candidates = reclaim_candidates;
        auction_candidates.insert(
                auction_candidates.end(),
                prefetch_candidates.begin(),
                prefetch_candidates.end());
        auction_candidates.insert(
                auction_candidates.end(),
                grow_candidates.begin(),
                grow_candidates.end());
        const char * auction_allocation_reason = "no_candidate";
        const auto auction_selected_allocation = memory_governor_auction_select(
                auction_candidates,
                effective_pressure_state,
                memory_governor_is_allocation_candidate,
                &auction_allocation_reason);
        const char * auction_reclaim_reason = "no_candidate";
        const auto auction_selected_reclaim = memory_governor_auction_select(
                auction_candidates,
                effective_pressure_state,
                memory_governor_is_reclaim_candidate,
                &auction_reclaim_reason);
        const std::string auction_top =
            memory_governor_format_candidates(auction_candidates, 5);
        const std::string auction_selected_allocation_log =
            memory_governor_format_candidate(auction_selected_allocation);
        const std::string auction_selected_reclaim_log =
            memory_governor_format_candidate(auction_selected_reclaim);
        const uint64_t memory_governor_observe_elapsed_us =
            (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - memory_governor_observe_t0).count();
        const char * planner_model_kind =
            moe_enabled ? "moe" : (flex_enabled ? "dense" : "none");
        const uint64_t planner_floor_bytes =
            moe_enabled ? (uint64_t) moe_stats.planner_floor_bytes :
            (flex_enabled ? (uint64_t) flex_stats.sched_fixed_bytes : 0);
        const uint64_t planner_dynamic_budget_bytes =
            moe_enabled
                ? (moe_stats.budget_unbounded
                    ? (uint64_t) moe_stats.expert_bytes
                    : (uint64_t) moe_stats.budget_bytes)
                : (flex_enabled
                    ? (uint64_t) (flex_stats.locked_bytes + flex_stats.ring_bytes)
                    : 0);

        std::ostringstream out;
        out << "memory_governor_observe"
            << " sample_count=" << sample_count
            << " planner_auto_enabled=" << (memory_governor_auto_backends_enabled ? 1 : 0)
            << " planner_model_kind=" << planner_model_kind
            << " planner_floor_bytes=" << planner_floor_bytes
            << " planner_dynamic_budget_bytes=" << planner_dynamic_budget_bytes
            << " governor_observe_elapsed_us=" << memory_governor_observe_elapsed_us
            << " idle=" << (idle ? 1 : 0)
            << " pressure_valid=" << (pressure_valid ? 1 : 0)
            << " pressure_state=" << kv_pressure_state_name(pressure_state)
            << " effective_pressure_state=" << kv_pressure_state_name(effective_pressure_state)
            << " effective_pressure_reason=" << effective_pressure_reason
            << " pressure_source=" << kv_pressure_source_name(pressure_source)
            << " pressure_stale=" << (pressure_stale ? 1 : 0)
            << " pressure_current_bytes=" << pressure_current_bytes
            << " pressure_low_water_bytes=" << pressure_low_water_bytes
            << " pressure_raw_excess_bytes=" << pressure_raw_excess_bytes
            << " pressure_excess_bytes=" << pressure_excess_bytes
            << " effective_pressure_critical_excess_bytes=" << effective_pressure_critical_excess_bytes
            << " rss_observed_bytes=" << window_observed_rss
            << " dense_flex_enabled=" << (flex_enabled ? 1 : 0)
            << " dense_resident_bytes=" << ((uint64_t) flex_stats.ring_bytes + (uint64_t) flex_stats.locked_bytes)
            << " dense_reclaimable_bytes=" << dense_reclaimable_bytes
            << " dense_reclaimable_above_floor_bytes=" << dense_reclaimable_above_floor
            << " dense_ring_floor_bytes=" << dense_ring_floor_bytes
            << " dense_ring_bytes=" << flex_stats.ring_bytes
            << " dense_slot_bytes=" << flex_stats.slot_bytes
            << " dense_locked_bytes=" << flex_stats.locked_bytes
            << " dense_delta_locked_bytes=" << flex_stats.delta_locked_bytes
            << " dense_delta_locked_tensors=" << flex_stats.delta_locked_tensors
            << " dense_delta_pin_saved_per_token_bytes=" << flex_stats.delta_pin_saved_per_token
            << " dense_delta_pin_attempts=" << flex_stats.delta_pin_attempts
            << " dense_delta_pin_failures=" << flex_stats.delta_pin_failures
            << " dense_delta_pin_async_submitted=" << flex_stats.delta_pin_async_submitted
            << " dense_delta_pin_async_completed=" << flex_stats.delta_pin_async_completed
            << " dense_delta_pin_async_rejected=" << flex_stats.delta_pin_async_rejected
            << " dense_delta_pin_async_pending=" << flex_stats.delta_pin_async_pending
            << " dense_delta_pin_async_last_io_us=" << flex_stats.delta_pin_async_last_io_us
            << " dense_delta_pin_async_last_elapsed_us=" << flex_stats.delta_pin_async_last_elapsed_us
            << " dense_lock_unused_bytes=" << flex_stats.lock_budget_unused
            << " dense_stream_per_token_bytes=" << flex_stats.stream_per_token
            << " dense_effective_ahead=" << flex_stats.effective_ahead
            << " dense_read_cost_bytes=" << flex_stats.read_cost_bytes
            << " dense_global_rebalance_bytes=" << flex_stats.global_rebalance_bytes
            << " dense_global_rebalance_tensors=" << flex_stats.global_rebalance_tensors
            << " dense_bytes_streamed=" << flex_stats.bytes_streamed
            << " dense_bytes_read_phys=" << flex_stats.bytes_read_phys
            << " dense_read_ops=" << flex_stats.read_ops
            << " window_enabled=" << (window_enabled ? 1 : 0)
            << " window_current_rss_bytes=" << window_stats.current_rss
            << " window_peak_rss_bytes=" << window_stats.peak_rss
            << " window_prefetched_bytes=" << window_stats.bytes_prefetched
            << " window_reclaimed_bytes=" << window_stats.bytes_reclaimed
            << " moe_enabled=" << (moe_enabled ? 1 : 0)
            << " moe_resident_bytes=" << moe_stats.resident_bytes
            << " moe_budget_bytes=" << moe_stats.budget_bytes
            << " moe_budget_unbounded=" << (moe_stats.budget_unbounded ? 1 : 0)
            << " moe_planner_safe_budget_bytes=" << moe_stats.planner_safe_budget_bytes
            << " moe_planner_floor_bytes=" << moe_stats.planner_floor_bytes
            << " moe_expert_bytes=" << moe_stats.expert_bytes
            << " moe_streams=" << moe_stats.streams
            << " moe_hits=" << moe_stats.hits
            << " moe_evictions=" << moe_stats.evictions
            << " moe_bytes_read=" << moe_stats.bytes_read
            << " moe_cache_hits=" << moe_stats.cache_hits
            << " moe_cache_misses=" << moe_stats.cache_misses
            << " moe_prefetch_hits=" << moe_stats.prefetch_hits
            << " moe_prefetch_late=" << moe_stats.prefetch_late
            << " moe_prefetch_unused=" << moe_stats.prefetch_unused
            << " moe_budget_dynamic_enabled=" << (memory_governor_moe_budget_dynamic_enabled ? 1 : 0)
            << " moe_budget_fast_start_enabled=" << (memory_governor_moe_budget_fast_start_enabled ? 1 : 0)
            << " moe_budget_action=" << moe_budget_action
            << " moe_budget_reason=" << moe_budget_reason
            << " moe_budget_old_bytes=" << moe_budget_old_bytes
            << " moe_budget_new_bytes=" << moe_budget_new_bytes
            << " moe_budget_warm_bytes=" << effective_moe_budget_warm_bytes
            << " moe_warm_working_set_bytes=" << moe_stats.warm_working_set_bytes
            << " moe_warm_working_set_groups=" << moe_stats.warm_working_set_groups
            << " moe_warm_working_set_coverage=" << moe_stats.warm_working_set_coverage
            << " moe_budget_headroom_bytes=" << moe_budget_headroom_bytes
            << " moe_budget_pressure_shrink_pct=" << memory_governor_moe_budget_pressure_shrink_pct
            << " moe_budget_pressure_shrink_max_bytes=" <<
                effective_moe_budget_pressure_shrink_max_bytes
            << " moe_budget_delta_evictions=" << moe_budget_delta_evictions
            << " moe_budget_delta_bytes_read=" << moe_budget_delta_bytes_read
            << " moe_budget_delta_prefetch_late=" << moe_budget_delta_prefetch_late
            << " moe_budget_delta_prefetch_dropped=" << moe_budget_delta_prefetch_dropped
            << " moe_budget_reclaim_target_bytes=" << moe_budget_reclaim_target_bytes
            << " moe_budget_reclaim_released_bytes=" << moe_budget_reclaim_released_bytes
            << " moe_budget_reclaim_released_groups=" << moe_budget_reclaim_released_groups
            << " moe_budget_reclaim_target_satisfied=" << (moe_budget_reclaim_target_satisfied ? 1 : 0)
            << " global_optimizer_enabled=" << (memory_governor_global_optimizer_enabled ? 1 : 0)
            << " global_optimizer_decision=" << global_optimizer_decision
            << " global_hard_headroom_bytes=" << memory_governor_global_hard_headroom_bytes
            << " global_effective_hard_headroom_bytes=" << effective_hard_headroom_bytes
            << " global_derived_hard_headroom_bytes=" << derived_hard_headroom_bytes
            << " global_available_growth_bytes=" << global_available_growth_bytes
            << " global_dense_grant_bytes=" << global_dense_grant_bytes
            << " global_dense_utility=" << global_dense_utility
            << " global_dense_roi=" << global_dense_roi
            << " global_dense_headroom_risk=" << global_dense_headroom_risk
            << " global_dense_pressure_tax=" << global_dense_pressure_tax
            << " global_dense_io_bound_ratio=" << global_dense_io_bound_ratio
            << " global_dense_ring_flowback_bytes=" << global_dense_ring_flowback_bytes
            << " global_dense_recommended_lock_bytes=" << global_dense_recommended_lock_bytes
            << " global_dense_target_ring_slots=" << global_dense_target_ring_slots
            << " dense_repin_enabled=" << (memory_governor_dense_repin_enabled ? 1 : 0)
            << " dense_repin_async_enabled=" << (memory_governor_dense_repin_async_enabled ? 1 : 0)
            << " dense_repin_idle_only=" << (memory_governor_dense_repin_idle_only ? 1 : 0)
            << " dense_repin_cooldown_samples=" << memory_governor_dense_repin_cooldown_samples
            << " dense_repin_next_sample=" << memory_governor_dense_repin_next_sample
            << " dense_repin_grant_bytes=" << dense_repin_grant_bytes
            << " dense_repin_headroom_bytes=" << dense_repin_headroom_bytes
            << " dense_repin_attempted=" << (dense_repin_result.attempted ? 1 : 0)
            << " dense_repin_changed=" << (dense_repin_result.changed ? 1 : 0)
            << " dense_repin_requested_bytes=" << dense_repin_result.requested_bytes
            << " dense_repin_pinned_bytes=" << dense_repin_result.pinned_bytes
            << " dense_repin_saved_per_token_bytes=" << dense_repin_result.saved_per_token_bytes
            << " dense_repin_candidates=" << dense_repin_result.candidates
            << " dense_repin_pinned_tensors=" << dense_repin_result.pinned_tensors
            << " dense_repin_io_us=" << dense_repin_result.io_us
            << " dense_repin_elapsed_us=" << dense_repin_result.elapsed_us
            << " dense_repin_roi=" << dense_repin_result.roi
            << " dense_repin_reason=" << dense_repin_result.reason
            << " dense_resize_attempted=" << (dense_resize_result.attempted ? 1 : 0)
            << " dense_resize_changed=" << (dense_resize_result.changed ? 1 : 0)
            << " dense_resize_old_slots=" << dense_resize_result.old_slots
            << " dense_resize_new_slots=" << dense_resize_result.new_slots
            << " dense_resize_old_bytes=" << dense_resize_result.old_bytes
            << " dense_resize_new_bytes=" << dense_resize_result.new_bytes
            << " dense_resize_reason=" << dense_resize_result.reason
            << " dense_ring_shrink_enabled=" << (memory_governor_dense_runtime_ring_shrink_enabled ? 1 : 0)
            << " dense_ring_shrink_required_bytes=" << dense_ring_shrink_required_bytes
            << " dense_ring_shrink_releasable_bytes=" << dense_ring_shrink_releasable_bytes
            << " dense_ring_shrink_target_slots=" << dense_ring_shrink_target_slots
            << " global_moe_grant_bytes=" << global_moe_grant_bytes
            << " global_moe_utility=" << global_moe_utility
            << " global_kv_utility=" << global_kv_utility
            << " global_prefetch_utility=" << global_prefetch_utility
            << " global_headroom_barrier=" << global_headroom_barrier
            << " global_moe_roi=" << global_moe_roi
            << " reallocation_enabled=" << (memory_governor_reallocation_enabled ? 1 : 0)
            << " reallocation_apply_moe=" << (memory_governor_reallocation_apply_moe ? 1 : 0)
            << " reallocation_confirm_enabled=" << (memory_governor_reallocation_confirm_enabled ? 1 : 0)
            << " reallocation_pending_added_bytes=" << reallocation_pending_added_bytes
            << " reallocation_pending_before_confirm_bytes=" <<
                reallocation_pending_before_confirm_bytes
            << " reallocation_pending_remaining_bytes=" << reallocation_pending_remaining_bytes
            << " reallocation_observed_drop_bytes=" << reallocation_observed_drop_bytes
            << " reallocation_confirm_limit_bytes=" << reallocation_confirm_limit_bytes
            << " reallocation_credit_earned_bytes=" << reallocation_credit_earned_bytes
            << " reallocation_credit_decayed_bytes=" << reallocation_credit_decayed_bytes
            << " reallocation_credit_available_bytes=" << reallocation_credit_available_bytes
            << " reallocation_credit_remaining_bytes=" << memory_governor_reallocation_credit_bytes
            << " reallocation_credit_spent_bytes=" << reallocation_credit_spent_bytes
            << " reallocation_best_kind=" << reallocation_best_kind
            << " reallocation_dense_grant_bytes=" << reallocation_dense_grant_bytes
            << " reallocation_moe_grant_bytes=" << reallocation_moe_grant_bytes
            << " reallocation_moe_old_budget_bytes=" << reallocation_moe_old_budget_bytes
            << " reallocation_moe_new_budget_bytes=" << reallocation_moe_new_budget_bytes
            << " reallocation_reason=" << reallocation_reason
            << " kv_memory_present=" << (kv_memory_present ? 1 : 0)
            << " kv_release_budget_valid=" << (kv_budget.valid ? 1 : 0)
            << " kv_resident_bytes=" << kv_budget.resident_bytes
            << " kv_reclaimable_resident_bytes=" << kv_budget.reclaimable_resident_bytes
            << " kv_slot_budget_valid=" << (kv_slot_budget_valid ? 1 : 0)
            << " kv_slot_resident_bytes=" << kv_slot_resident_bytes
            << " kv_slot_reclaimable_resident_bytes=" << kv_slot_reclaimable_resident_bytes
            << " kv_slot_sequences=" << kv_slot_sequences
            << " kv_slot_reclaimable_sequences=" << kv_slot_reclaimable_sequences
            << " kv_effective_budget_valid=" << (kv_effective_budget_valid ? 1 : 0)
            << " kv_effective_budget_source=" << kv_effective_budget_source
            << " kv_effective_resident_bytes=" << kv_effective_resident_bytes
            << " kv_effective_reclaimable_resident_bytes=" << kv_effective_reclaimable_resident_bytes
            << " kv_soft_budget_enabled=" << (memory_governor_kv_soft_budget_enabled ? 1 : 0)
            << " kv_soft_target_bytes=" << kv_soft_target_bytes
            << " kv_soft_idle_bytes=" << memory_governor_kv_soft_idle_bytes
            << " kv_soft_protected_bytes=" << kv_soft_protected_bytes
            << " kv_soft_excess_bytes=" << kv_soft_excess_bytes
            << " kv_soft_release_target_bytes=" << kv_soft_release_target_bytes
            << " kv_soft_offload_target_bytes=" << kv_soft_offload_target_bytes
            << " kv_soft_reason=" << kv_soft_reason
            << " auction_candidates=" << auction_top
            << " auction_selected_allocation=" << auction_selected_allocation_log
            << " auction_allocation_reason=" << auction_allocation_reason
            << " auction_selected_reclaim=" << auction_selected_reclaim_log
            << " auction_reclaim_reason=" << auction_reclaim_reason
            << " would_reclaim_candidates=" << would_reclaim
            << " would_prefetch_candidates=" << would_prefetch
            << " prefetch_budget_enabled=" << (memory_governor_prefetch_budget_enabled ? 1 : 0)
            << " prefetch_budget_auto=" << (memory_governor_prefetch_budget_auto ? 1 : 0)
            << " prefetch_budget_runtime_auto=" << (memory_governor_prefetch_budget_runtime_auto ? 1 : 0)
            << " prefetch_budget_auto_reason=" << prefetch_budget_auto_reason
            << " prefetch_budget_config_bytes=" << prefetch_budget_config_bytes
            << " prefetch_budget_effective_bytes=" << prefetch_budget_effective_bytes
            << " prefetch_budget_clamp_reason=" << prefetch_budget_clamp_reason
            << " prefetch_budget_runtime_headroom_bytes=" << prefetch_budget_runtime_headroom_bytes
            << " prefetch_budget_headroom_guard_bytes=" << memory_governor_prefetch_budget_headroom_bytes
            << " prefetch_budget_min_bytes_per_tick=" << memory_governor_prefetch_budget_min_bytes_per_tick
            << " prefetch_budget_max_bytes_per_tick=" << memory_governor_prefetch_budget_max_bytes_per_tick
            << " prefetch_budget_tick_bytes=" << prefetch_budget_tick_bytes
            << " prefetch_budget_dense_bytes=" << prefetch_budget_dense_bytes
            << " prefetch_budget_moe_bytes=" << prefetch_budget_moe_bytes
            << " prefetch_budget_remaining_bytes=" << prefetch_budget_remaining_bytes
            << " prefetch_budget_kv_resume_used_bytes=" << prefetch_budget_kv_reserved_bytes
            << " prefetch_budget_kv_resume_overruns=" << prefetch_budget_kv_overruns
            << " dense_prefetch_budget_dropped=" << flex_stats.prefetch_budget_dropped
            << " moe_prefetch_budget_dropped=" << moe_stats.prefetch_budget_dropped
            << " governor_async_actions_enabled=" << (memory_governor_async_actions_enabled ? 1 : 0)
            << " governor_async_queue_depth=" << memory_governor_async_queue_depth.load(std::memory_order_relaxed)
            << " governor_async_submitted=" << memory_governor_async_submitted.load(std::memory_order_relaxed)
            << " governor_async_completed=" << memory_governor_async_completed.load(std::memory_order_relaxed)
            << " governor_async_rejected=" << memory_governor_async_rejected.load(std::memory_order_relaxed)
            << " governor_async_dropped=" << memory_governor_async_dropped.load(std::memory_order_relaxed)
            << " governor_async_relieved_drained_bytes=" << async_relieved_drained_bytes
            << " clean_reclaim_enabled=" << (memory_governor_clean_reclaim_enabled ? 1 : 0)
            << " clean_reclaim_ranked_enabled=" << (memory_governor_clean_reclaim_ranked_enabled ? 1 : 0)
            << " clean_reclaim_attempted=" << (clean_result.attempted ? 1 : 0)
            << " clean_reclaim_kind=" << clean_result.kind
            << " clean_reclaim_score=" << clean_result.score
            << " clean_reclaim_target_bytes=" << clean_result.target_bytes
            << " clean_reclaim_released_bytes=" << clean_result.released_bytes
            << " clean_reclaim_released_objects=" << clean_result.released_objects
            << " clean_reclaim_passes=" << clean_result.passes
            << " clean_reclaim_candidates_tried=" << clean_result.candidates_tried
            << " clean_reclaim_max_passes=" << memory_governor_clean_reclaim_max_passes
            << " clean_reclaim_target_satisfied=" << (clean_result.target_satisfied ? 1 : 0)
            << " clean_reclaim_reason=" << clean_result.reason
            << " kv_release_enabled=" << (memory_governor_kv_release_enabled ? 1 : 0)
            << " kv_release_evaluate_attempted=" << (kv_release_result.evaluate_attempted ? 1 : 0)
            << " kv_release_attempted=" << (kv_release_result.attempted ? 1 : 0)
            << " kv_release_score=" << kv_release_result.score
            << " kv_release_target_bytes=" << kv_release_result.target_bytes
            << " kv_release_max_blocks=" << kv_release_result.max_blocks
            << " kv_release_blocks=" << kv_release_result.blocks
            << " kv_release_bytes=" << kv_release_result.bytes
            << " kv_release_relieved_bytes=" << kv_release_result.relieved_bytes
            << " kv_release_shortfall_bytes=" << kv_release_result.shortfall_bytes
            << " kv_release_state_changed=" << (kv_release_result.state_changed ? 1 : 0)
            << " kv_release_io_failure=" << (kv_release_result.io_failure ? 1 : 0)
            << " kv_release_fail_stop=" << (kv_release_result.fail_stop ? 1 : 0)
            << " kv_release_io_errno=" << kv_release_result.io_errno
            << " kv_release_outcome=" << kv_release_result.outcome
            << " kv_release_action_reason=" << kv_release_result.action_reason
            << " kv_release_reason=" << kv_release_result.reason
            << " kv_offload_enabled=" << (memory_governor_kv_offload_enabled ? 1 : 0)
            << " kv_offload_evaluate_attempted=" << (kv_offload_result.evaluate_attempted ? 1 : 0)
            << " kv_offload_attempted=" << (kv_offload_result.attempted ? 1 : 0)
            << " kv_offload_seq_id=" << kv_offload_result.seq_id
            << " kv_offload_score=" << kv_offload_result.score
            << " kv_offload_target_bytes=" << kv_offload_result.target_bytes
            << " kv_offload_max_blocks=" << kv_offload_result.max_blocks
            << " kv_offload_blocks=" << kv_offload_result.blocks
            << " kv_offload_bytes=" << kv_offload_result.bytes
            << " kv_offload_relieved_bytes=" << kv_offload_result.relieved_bytes
            << " kv_offload_shortfall_bytes=" << kv_offload_result.shortfall_bytes
            << " kv_offload_state_changed=" << (kv_offload_result.state_changed ? 1 : 0)
            << " kv_offload_io_failure=" << (kv_offload_result.io_failure ? 1 : 0)
            << " kv_offload_fail_stop=" << (kv_offload_result.fail_stop ? 1 : 0)
            << " kv_offload_io_errno=" << kv_offload_result.io_errno
            << " kv_offload_backend=" << kv_offload_result.backend
            << " kv_offload_outcome=" << kv_offload_result.outcome
            << " kv_offload_action_reason=" << kv_offload_result.action_reason
            << " kv_offload_reason=" << kv_offload_result.reason;
        SRV_INF("%s\n", out.str().c_str());
        memory_governor_observe_last = server_kv_pressure_runtime::clock::now();
        return kv_release_result.attempted || kv_offload_result.attempted;
    }
```

### KV pressure sampler 初始化：init_kv_pressure_sampler()

来源：`tools/server/server-context.cpp:4990-5176`

```cpp
    bool init_kv_pressure_sampler() {
        kv_pressure_sampler_owner.reset();
        kv_pressure_runtime.disable();
        kv_pressure_runtime.dry_run_disable();
        kv_pressure_runtime.bounded_release_disable();
        kv_pressure_dry_run_config = server_kv_pressure_dry_run_config{};
        kv_pressure_bounded_release_config = server_kv_pressure_bounded_release_config{};
        kv_pressure_bounded_release_episode = 0;
        kv_pressure_unified_action_config = server_kv_pressure_unified_action_config{};
        kv_governor_state.reset();
        init_memory_governor_observer_from_env();

        if (memory_governor_kv_release_enabled || memory_governor_kv_offload_enabled) {
            const bool legacy_active = (std::getenv("LLAMA_KV_PAGED_RELEASE") &&
                std::strcmp(std::getenv("LLAMA_KV_PAGED_RELEASE"), "1") == 0);
            if (legacy_active) {
                SRV_ERR("%s",
                        "LLAMA_MEMORY_GOVERNOR_KV_RELEASE/OFFLOAD and "
                        "LLAMA_KV_PAGED_RELEASE=1 are mutually exclusive; "
                        "server initialization rejected\n");
                return false;
            }
            if (std::getenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION")) {
                SRV_WRN("%s",
                        "LLAMA_MEMORY_GOVERNOR_KV_RELEASE/OFFLOAD supersedes "
                        "LLAMA_KV_PRESSURE_UNIFIED_ACTION; old unified action "
                        "path disabled\n");
            }
        } else {
            const auto unified_decision =
                    server_kv_pressure_unified_action_startup_decide_from_env();
            if (unified_decision.status == server_kv_pressure_unified_action_startup_status::invalid ||
                    unified_decision.status == server_kv_pressure_unified_action_startup_status::conflict) {
                SRV_ERR("KV pressure unified action configuration error: %s; server initialization rejected\n",
                        unified_decision.error.c_str());
                return false;
            }
            if (unified_decision.status == server_kv_pressure_unified_action_startup_status::enabled) {
                kv_pressure_unified_action_config = unified_decision.config;
                SRV_INF("KV pressure unified action enabled: target_bytes=%" PRIu64
                        " max_blocks=%" PRIu32 "\n",
                        kv_pressure_unified_action_config.target_bytes,
                        kv_pressure_unified_action_config.max_blocks);
            }
        }

        kv_pressure_enablement enablement = kv_pressure_sampler_environment_enablement();
        if (enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_DISABLED &&
                memory_governor_observe_enabled &&
                std::getenv("LLAMA_KV_PRESSURE_SAMPLER") == nullptr) {
            enablement = kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_ENABLED;
            SRV_INF("%s", "KV pressure sampler auto-enabled by LLAMA_MEMORY_GOVERNOR\n");
        }
        if (enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_DISABLED) {
            return true;
        }
        if (enablement == kv_pressure_enablement::KV_PRESSURE_ENABLEMENT_INVALID) {
            SRV_WRN("%s", "invalid LLAMA_KV_PRESSURE_SAMPLER value; pressure telemetry disabled\n");
            return true;
        }

        server_kv_pressure_config config;
        std::string error;
        if (!server_kv_pressure_config_from_env(config, error)) {
            SRV_WRN("KV pressure telemetry disabled: %s\n", error.c_str());
            return true;
        }

        // Parse dry-run config independently — it is not gated on sampler init
        // success because the dry-run could conceptually run with a different
        // pressure input source in the future.  For now it shares the sampler
        // lifecycle but fails independently.
        {
            server_kv_pressure_dry_run_config dry_cfg;
            std::string dry_error;
            if (!server_kv_pressure_dry_run_config_from_env(dry_cfg, dry_error)) {
                SRV_WRN("KV pressure dry-run disabled: %s\n", dry_error.c_str());
            } else if (dry_cfg.enabled) {
                kv_pressure_dry_run_config = dry_cfg;
                // Dry-run piggybacks on the pressure sampler lifecycle but does
                // not require it — the sampler must also be initialized for the
                // dry-run to fire because it reads pressure state.
                kv_pressure_runtime.dry_run_enable(dry_cfg);
                SRV_INF("KV pressure dry-run enabled: target_bytes=%" PRIu64
                        " max_scan_blocks=%" PRIu32 " cooldown_ms=%" PRIu32
                        " backoff_ms=%" PRIu32 "\n",
                        dry_cfg.target_bytes, dry_cfg.max_scan_blocks,
                        dry_cfg.cooldown_ms, dry_cfg.backoff_ms);
            }
        }

        // Parse bounded destructive release config independently.
        // Mutual exclusion checks:
        // - LLAMA_KV_PAGED_RELEASE=1 AND LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 → fail-closed
        // - LLAMA_KV_PRESSURE_DRY_RUN=1 AND LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 → fail-closed
        {
            server_kv_pressure_bounded_release_config bounded_cfg;
            std::string bounded_error;
            if (!server_kv_pressure_bounded_release_config_from_env(
                        bounded_cfg, bounded_error)) {
                SRV_WRN("KV pressure bounded release disabled: %s\n",
                        bounded_error.c_str());
            } else if (bounded_cfg.enabled) {
                bool bounded_disabled = false;

                // Legacy mutual exclusion
                const bool legacy_active = (std::getenv("LLAMA_KV_PAGED_RELEASE") &&
                    std::strcmp(std::getenv("LLAMA_KV_PAGED_RELEASE"), "1") == 0);
                if (legacy_active) {
                    SRV_ERR("%s",
                            "LLAMA_KV_PAGED_RELEASE=1 and "
                            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 are mutually "
                            "exclusive; bounded release disabled\n");
                    bounded_disabled = true;
                }

                // Dry-run mutual exclusion: cannot run dry-run AND destructive
                // bounded release simultaneously — they are alternative paths.
                if (!bounded_disabled && kv_pressure_dry_run_config.enabled) {
                    SRV_ERR("%s",
                            "LLAMA_KV_PRESSURE_DRY_RUN=1 and "
                            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 are mutually "
                            "exclusive; bounded release disabled\n");
                    bounded_disabled = true;
                }

                if (!bounded_disabled && (memory_governor_kv_release_enabled ||
                            memory_governor_kv_offload_enabled)) {
                    SRV_ERR("%s",
                            "LLAMA_MEMORY_GOVERNOR_KV_RELEASE/OFFLOAD and "
                            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 are mutually "
                            "exclusive; bounded release disabled\n");
                    bounded_disabled = true;
                }

                if (!bounded_disabled) {
                    kv_pressure_bounded_release_config = bounded_cfg;
                    kv_pressure_runtime.bounded_release_enable(bounded_cfg);
                    kv_pressure_bounded_release_episode = 0;
                    SRV_INF("KV pressure bounded release enabled: target_mode=%s max_release_bytes=%"
                            PRIu64 " max_scan_blocks=%" PRIu32 " cooldown_ms=%"
                            PRIu32 " backoff_ms=%" PRIu32 "\n",
                            bounded_cfg.dynamic_target ? "dynamic" : "fixed",
                            bounded_cfg.target_bytes,
                            bounded_cfg.max_scan_blocks,
                            bounded_cfg.cooldown_ms,
                            bounded_cfg.backoff_ms);

                    // Emit structural capability diagnostic at init time so
                    // every skip reason can be traced back to a specific condition.
                    if (ctx_tgt) {
                        auto * mem = llama_get_memory(ctx_tgt);
                        if (mem) {
                            const auto cap = mem->bounded_release_can_enable_diagnose();
                            SRV_INF("kv_pressure_bounded_release_capability"
                                    " can_enable=%d"
                                    " paged=%d ingraph=%d layers_supported=%d"
                                    " row_idx=%d swap_disabled=%d layout_supported=%d\n",
                                    cap.can_enable ? 1 : 0,
                                    cap.paged ? 1 : 0,
                                    cap.ingraph ? 1 : 0,
                                    cap.layers_supported ? 1 : 0,
                                    cap.row_idx ? 1 : 0,
                                    cap.swap_disabled ? 1 : 0,
                                    cap.layout_supported ? 1 : 0);
                        }
                    }
                }
            }
        }

        try {
            auto sampler = std::make_unique<kv_pressure_sampler>();
            if (!sampler->init(enablement)) {
                SRV_WRN("%s", "KV pressure sampler initialization failed; pressure telemetry disabled\n");
                return true;
            }

            kv_pressure_sampler_owner = std::move(sampler);
            kv_pressure_runtime.enable(config);
        } catch (const std::exception & e) {
            SRV_WRN("KV pressure sampler initialization failed; pressure telemetry disabled: %s\n", e.what());
        } catch (...) {
            SRV_WRN("%s", "KV pressure sampler initialization failed; pressure telemetry disabled\n");
        }
        return true;
    }
```

### 运行期采样入口：maybe_sample_kv_pressure()

来源：`tools/server/server-context.cpp:5178-5579`

```cpp
    void maybe_sample_kv_pressure(bool idle) {
        const auto now = server_kv_pressure_runtime::clock::now();
        const bool observe_due = memory_governor_observe_due(now);

        if (!kv_pressure_sampler_owner) {
            if (observe_due) {
                publish_memory_governor_observation(idle, 0, nullptr);
            }
            return;
        }

        if (!kv_pressure_runtime.sample_due(now)) {
            if (observe_due) {
                const auto telemetry = kv_pressure_sampler_owner->telemetry();
                publish_memory_governor_observation(
                        idle, kv_pressure_runtime.sample_count(), &telemetry);
            }
            return;
        }

        kv_pressure_sampler_owner->sample();
        const auto event = kv_pressure_runtime.record_sample(
                now, idle, kv_pressure_sampler_owner->telemetry());
        if (event.should_log()) {
            const std::string marker = server_kv_pressure_format_marker(event);
            SRV_INF("%s\n", marker.c_str());
        }
        bool destructive_phase_did_work = false;
        if (observe_due) {
            const auto telemetry = kv_pressure_sampler_owner->telemetry();
            destructive_phase_did_work = publish_memory_governor_observation(
                    idle, kv_pressure_runtime.sample_count(), &telemetry);
        }

        // Per-sample defensive gating: at most one of dry-run or bounded
        // release may be evaluated per sample cycle.  Under normal operation the
        // init-time mutual-exclusion checks guarantee that only one path is
        // enabled, but this flag provides defense-in-depth against config errors
        // or runtime state corruption.

        // --- Dry-run bounded release evaluation (read-only, never mutates KV) ---
        {
            const auto & telemetry = kv_pressure_sampler_owner->telemetry();

            server_kv_pressure_dry_run_event dry_event;
            dry_event.pressure_state   = telemetry.state;
            dry_event.pressure_source  = telemetry.source;
            dry_event.stale            = telemetry.stale;
            dry_event.idle             = idle;
            dry_event.release_enabled  = false;  // set below after status check
            dry_event.sample_count     = kv_pressure_runtime.sample_count();
            dry_event.target_bytes     = kv_pressure_dry_run_config.target_bytes;
            dry_event.max_scan_blocks  = kv_pressure_dry_run_config.max_scan_blocks;

            const auto now = server_kv_pressure_runtime::clock::now();

            // Explicit evaluation gate — no empty branches, no sentinel strings.
            // Only when should_evaluate==true is the dry-run scanner called.
            bool should_evaluate = false;
            const char * skip_reason = nullptr;

            // Master switch: enabled + non-zero target.
            if (!kv_pressure_dry_run_config.enabled ||
                    kv_pressure_dry_run_config.target_bytes == 0) {
                // Dry-run is disabled — zero markers, zero scanner calls.
            } else if (!ctx_tgt || !llama_get_memory(ctx_tgt)) {
                skip_reason = "no_memory";
            } else {
                // Release capability check: only hard blockers set skip_reason.
                const auto release_status =
                    llama_get_memory(ctx_tgt)->paged_release_status();
                switch (release_status) {
                case llama_kv_release_status::available:
                    dry_event.release_enabled = true;
                    break;
                case llama_kv_release_status::disabled:
                    break;  // observational, not a skip
                case llama_kv_release_status::swap_enabled:
                    skip_reason = "swap_enabled";        break;
                case llama_kv_release_status::not_paged:
                    skip_reason = "not_paged";           break;
                case llama_kv_release_status::layout_unsupported:
                    skip_reason = "layout_unsupported";  break;
                }

                if (!skip_reason && telemetry.stale) {
                    skip_reason = "stale";
                }

                // Trigger-state gate: only PRESSURE / CRITICAL are eligible.
                if (!skip_reason &&
                        telemetry.state != kv_pressure_state::PRESSURE &&
                        telemetry.state != kv_pressure_state::CRITICAL) {
                    // NORMAL or RECOVERY — no evaluation, no marker.
                } else if (!skip_reason) {
                    // Cooldown / state-entry check.  dry_run_due() returns true
                    // on state entry (immediate evaluation) or when the cooldown
                    // has elapsed.  Returns false when still within the cooldown
                    // window — skip this sample, no marker.
                    should_evaluate = kv_pressure_runtime.dry_run_due(
                            now, telemetry.state, telemetry.stale);
                }
            }

            if (should_evaluate) {
                // Evaluate dry-run scan
                auto * mem = llama_get_memory(ctx_tgt);
                dry_event.result = mem->bounded_release_dry_run(
                        kv_pressure_dry_run_config.target_bytes,
                        kv_pressure_dry_run_config.max_scan_blocks);
                const bool had_shortfall = dry_event.result.shortfall_bytes > 0 &&
                    dry_event.result.block_scan_exhausted;
                kv_pressure_runtime.dry_run_record(had_shortfall, now);
                dry_event.cooldown_ms = kv_pressure_dry_run_config.cooldown_ms;
                skip_reason = "none";
            }

            if (skip_reason) {
                destructive_phase_did_work = true;
                dry_event.skipped_reason = skip_reason;
                const std::string marker =
                    server_kv_pressure_dry_run_format_marker(dry_event);
                // Ownership ABORT is a correctness anomaly — escalate to WARNING.
                if (dry_event.result.ownership_aborted) {
                    SRV_WRN("%s\n", marker.c_str());
                } else {
                    SRV_INF("%s\n", marker.c_str());
                }
            }
        }

        // --- EdgeKV Governor: RELEASE first, scored OFFLOAD only after a
        // prior core no_candidate result arms a later decision. ---
        {
            const auto telemetry = kv_pressure_sampler_owner->telemetry();
            if (kv_pressure_unified_action_config.enabled &&
                    !memory_governor_kv_release_enabled &&
                    !memory_governor_kv_offload_enabled) {
                auto * mem = ctx_tgt ? llama_get_memory(ctx_tgt) : nullptr;
                const uint64_t decision_id = ++kv_decision_next;
                const server_kv_pressure_snapshot pressure_snapshot {
                    telemetry.state,
                    telemetry.source,
                    telemetry.sample_valid,
                    telemetry.stale,
                    telemetry.pressure_basis_valid,
                    telemetry.pressure_current_bytes,
                    telemetry.pressure_low_water_bytes,
                    telemetry.pressure_basis_generation,
                    kv_pressure_runtime.sample_count(),
                    decision_id,
                };

                std::vector<server_kv_claimant_snapshot> claimant_snapshots;
                claimant_snapshots.reserve(slots.size());
                const int64_t now_us = ggml_time_us();
                for (const auto & slot : slots) {
                    const bool active = slot.is_processing() || slot.task != nullptr;
                    const bool shared = slot.task && (slot.task->is_parent() || slot.task->is_child());
                    const uint64_t idle_age_us = !active && slot.t_last_used >= 0 && now_us > slot.t_last_used
                        ? (uint64_t) (now_us - slot.t_last_used)
                        : 0;
                    const uint64_t logical_tokens = slot.prompt.n_tokens() > 0
                        ? (uint64_t) slot.prompt.n_tokens()
                        : 0;
                    const uint64_t reclaimable_bytes = mem && !active && logical_tokens > 0
                        ? (uint64_t) llama_state_seq_get_size_ext(
                                ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE)
                        : 0;
                    claimant_snapshots.push_back({
                            llama_kv_memory_claimant::kv,
                            slot.id,
                            kv_governor_state.claimant_epoch(slot.id),
                            active,
                            slot.kv_resume_protected,
                            shared,
                            idle_age_us,
                            logical_tokens,
                            reclaimable_bytes,
                            slot.kv_reuse_hint_tokens > 0
                                ? (uint64_t) slot.kv_reuse_hint_tokens
                                : 0,
                            reclaimable_bytes,
                    });
                }

                auto result = server_kv_pressure_execute_governor(
                        kv_pressure_unified_action_config,
                        kv_governor_state,
                        mem ? server_kv_pressure_action_ops {
                            [mem](const llama_kv_action_request & request) {
                                return mem->execute_action(request);
                            },
                        } : server_kv_pressure_action_ops {},
                        pressure_snapshot,
                        claimant_snapshots);
                result.observation.idle = idle;
                const std::string marker =
                    server_kv_pressure_unified_action_format_marker(result);
                if (result.release.io_failure || result.release.fail_stop ||
                        result.offload.io_failure || result.offload.fail_stop) {
                    SRV_WRN("%s\n", marker.c_str());
                } else {
                    SRV_INF("%s\n", marker.c_str());
                }
                destructive_phase_did_work = true;
            }
        }

        // --- Bounded destructive release evaluation (Phase C) ---
        // Mutually exclusive with dry-run Phase B at the per-sample level:
        // if dry-run already ran, bounded release is skipped this sample.
        //
        // Also performs a defensive per-sample legacy contamination check:
        // if LLAMA_KV_PAGED_RELEASE=1 is detected at sample time, bounded
        // release is skipped even if the init-time check missed it.
        {
            const auto & telemetry = kv_pressure_sampler_owner->telemetry();

            server_kv_pressure_bounded_release_event bounded_event;
            bounded_event.pressure_state   = telemetry.state;
            bounded_event.pressure_source  = telemetry.source;
            bounded_event.stale            = telemetry.stale;
            bounded_event.idle             = idle;
            bounded_event.legacy_enabled   = false;  // set below after check
            bounded_event.sample_count     = kv_pressure_runtime.sample_count();
            bounded_event.episode          = kv_pressure_bounded_release_episode;
            bounded_event.target_bytes     = kv_pressure_bounded_release_config.target_bytes;
            bounded_event.dynamic_target   = kv_pressure_bounded_release_config.dynamic_target;
            bounded_event.max_release_bytes = kv_pressure_bounded_release_config.target_bytes;
            bounded_event.max_scan_blocks  = kv_pressure_bounded_release_config.max_scan_blocks;
            bounded_event.pressure_basis_valid = telemetry.pressure_basis_valid;
            bounded_event.pressure_current_bytes = telemetry.pressure_current_bytes;
            bounded_event.pressure_low_water_bytes = telemetry.pressure_low_water_bytes;
            bounded_event.pressure_basis_generation = telemetry.pressure_basis_generation;

            const auto now2 = server_kv_pressure_runtime::clock::now();

            bool should_evaluate_bounded = false;
            const char * bounded_skip_reason = nullptr;

            // Master switch
            if (!kv_pressure_bounded_release_config.enabled ||
                    kv_pressure_bounded_release_config.target_bytes == 0) {
                // Bounded release disabled — no marker.
            } else if (destructive_phase_did_work) {
                // Defense-in-depth: another governor path already touched KV
                // this sample — skip bounded release to prevent double-execution.
                bounded_skip_reason = "dry_run_active";
            } else if (!ctx_tgt || !llama_get_memory(ctx_tgt)) {
                bounded_skip_reason = "no_memory";
            } else {
                // Structural capability check (independent of legacy policy).
                // Use paged_release_status for precise reason granularity
                // and supplement with per-condition diagnose for exact attribution.
                auto * mem = llama_get_memory(ctx_tgt);

                // Populate per-condition capability fields for marker attribution
                {
                    const auto cap = mem->bounded_release_can_enable_diagnose();
                    bounded_event.can_enable        = cap.can_enable ? 1 : 0;
                    bounded_event.cap_paged         = cap.paged ? 1 : 0;
                    bounded_event.cap_ingraph       = cap.ingraph ? 1 : 0;
                    bounded_event.cap_layers        = cap.layers_supported ? 1 : 0;
                    bounded_event.cap_row_idx       = cap.row_idx ? 1 : 0;
                    bounded_event.cap_swap_disabled = cap.swap_disabled ? 1 : 0;
                    bounded_event.cap_layout        = cap.layout_supported ? 1 : 0;
                }

                if (!mem->bounded_release_can_enable()) {
                    const auto status = mem->paged_release_status();
                    switch (status) {
                    case llama_kv_release_status::not_paged:
                        bounded_skip_reason = "not_paged"; break;
                    case llama_kv_release_status::layout_unsupported:
                        bounded_skip_reason = "layout_unsupported"; break;
                    case llama_kv_release_status::swap_enabled:
                        bounded_skip_reason = "swap_enabled"; break;
                    default: {
                        // available or disabled means the basic paged+layout+!swap
                        // checks passed.  Use per-condition diagnose for exact
                        // skip reason — no catch-all "structurally_disabled".
                        const auto cap = mem->bounded_release_can_enable_diagnose();
                        if (!cap.ingraph) {
                            bounded_skip_reason = "not_ingraph";
                        } else if (!cap.layers_supported) {
                            bounded_skip_reason = "no_layers";
                        } else if (!cap.row_idx) {
                            bounded_skip_reason = "no_row_idx";
                        } else {
                            // Defensive: all diagnose fields are true but
                            // can_enable is still false — this should not
                            // happen; report as unknown structural block.
                            bounded_skip_reason = "structurally_disabled";
                        }
                        break;
                    }
                    }
                } else {
                    // Per-sample legacy contamination check: if legacy release
                    // is active, bounded release must not execute (fail-closed).
                    const bool legacy_active =
                        (std::getenv("LLAMA_KV_PAGED_RELEASE") &&
                         std::strcmp(std::getenv("LLAMA_KV_PAGED_RELEASE"), "1") == 0);
                    bounded_event.legacy_enabled = legacy_active;
                    if (legacy_active) {
                        bounded_skip_reason = "legacy_active";
                    }
                }

                if (!bounded_skip_reason && telemetry.stale) {
                    bounded_skip_reason = "stale";
                }

                if (!bounded_skip_reason &&
                        telemetry.state != kv_pressure_state::PRESSURE &&
                        telemetry.state != kv_pressure_state::CRITICAL) {
                    // NORMAL or RECOVERY — no evaluation, no marker.
                } else if (!bounded_skip_reason) {
                    should_evaluate_bounded =
                        kv_pressure_runtime.bounded_release_due(
                                now2, telemetry.state, telemetry.stale,
                                telemetry.pressure_basis_generation);
                }
            }

            if (should_evaluate_bounded) {
                auto * mem = llama_get_memory(ctx_tgt);
                if (kv_pressure_bounded_release_config.dynamic_target) {
                    const auto budget = mem->sample_kv_release_budget();
                    bounded_event.kv_budget_valid = budget.valid;
                    bounded_event.kv_budget_ownership_aborted = budget.ownership_aborted;
                    bounded_event.kv_resident_bytes = budget.resident_bytes;
                    bounded_event.kv_reclaimable_resident_bytes = budget.reclaimable_resident_bytes;
                    const auto target = server_kv_pressure_compute_dynamic_target(
                            telemetry, budget, kv_pressure_bounded_release_config.target_bytes);
                    bounded_event.water_excess_bytes = target.water_excess_bytes;
                    bounded_event.target_bytes = target.effective_target_bytes;
                    bounded_event.target_clamp = target.clamp_reason;
                    bounded_event.decision_reason = target.decision_reason;
                    if (!target.valid || target.effective_target_bytes == 0) {
                        bounded_skip_reason = target.decision_reason;
                        kv_pressure_runtime.bounded_release_record(
                                !target.valid || budget.ownership_aborted, now2);
                        bounded_event.cooldown_ms =
                            kv_pressure_runtime.bounded_release_current_cooldown_ms();
                        should_evaluate_bounded = false;
                    }
                }
            }

            if (should_evaluate_bounded) {
                auto * mem = llama_get_memory(ctx_tgt);
                // Capture independent counter values before release for delta
                // attribution — these counters are the authoritative source,
                // distinct from process-wide strace page-discard syscalls.
                const uint64_t cnt_bytes_before  = mem->bounded_release_counter_bytes();
                const uint64_t cnt_blocks_before = mem->bounded_release_counter_blocks();
                // Sample KV resident pages before release (mincore hard gate)
                bounded_event.mincore_before_bytes = mem->sample_kv_resident_bytes();
                bounded_event.result = mem->bounded_release(
                        bounded_event.target_bytes,
                        kv_pressure_bounded_release_config.max_scan_blocks);
                // Sample KV resident pages after release
                bounded_event.mincore_after_bytes = mem->sample_kv_resident_bytes();
                const uint64_t cnt_bytes_after  = mem->bounded_release_counter_bytes();
                const uint64_t cnt_blocks_after = mem->bounded_release_counter_blocks();
                bounded_event.bounded_cnt_bytes_delta  = cnt_bytes_after - cnt_bytes_before;
                bounded_event.bounded_cnt_blocks_delta = cnt_blocks_after - cnt_blocks_before;
                const bool had_shortfall = bounded_event.result.ownership_aborted ||
                    bounded_event.result.madvise_failures > 0 ||
                    (bounded_event.result.shortfall_bytes > 0 &&
                     bounded_event.result.block_scan_exhausted);
                kv_pressure_runtime.bounded_release_record(had_shortfall, now2);
                bounded_event.cooldown_ms =
                    kv_pressure_runtime.bounded_release_current_cooldown_ms();
                if (bounded_event.dynamic_target) {
                    bounded_event.water_shortfall_bytes =
                        bounded_event.water_excess_bytes > bounded_event.result.released_bytes ?
                        bounded_event.water_excess_bytes - bounded_event.result.released_bytes : 0;
                    bounded_event.water_overshoot_bytes =
                        bounded_event.result.released_bytes > bounded_event.water_excess_bytes ?
                        bounded_event.result.released_bytes - bounded_event.water_excess_bytes : 0;
                }
                bounded_skip_reason = "none";
                // Advance episode counter on successful evaluation
                kv_pressure_bounded_release_episode += 1;
                bounded_event.episode = kv_pressure_bounded_release_episode;
            }

            if (bounded_skip_reason) {
                bounded_event.skipped_reason = bounded_skip_reason;
                const std::string marker =
                    server_kv_pressure_bounded_release_format_marker(bounded_event);
                if (bounded_event.result.ownership_aborted) {
                    SRV_WRN("%s\n", marker.c_str());
                } else {
                    SRV_INF("%s\n", marker.c_str());
                }
            }
        }
    }
```

## 16.4 实验取证脚本完整关键代码

### global memory sweep 脚本

来源：`scripts/profile-global-memory-sweep.sh:1-273`

```bash
#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${SERVER:-$ROOT/build/bin/llama-server}"
DENSE_MODEL="${DENSE_MODEL:-/root/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.ffn-bplus-b.gguf}"
MOE_MODEL="${MOE_MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
MOE_SIDECAR="${MOE_SIDECAR:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.mwq-v2-hier-legacy.sidecar}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/global-memory-sweep-$(date +%Y%m%d-%H%M%S)}"
MEMORY_MBS="${MEMORY_MBS:-1000 1250 1500 1750 2000 2250 2500 2750 3000}"
HIGH_MARGIN_MB="${HIGH_MARGIN_MB:-0}"
TOKENS="${TOKENS:-16}"
CTX="${CTX:-1024}"
BATCH="${BATCH:-16}"
THREADS="${THREADS:-8}"
BASE_PORT="${BASE_PORT:-41000}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-240}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-300}"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf 'model\tmem_mb\thigh_mb\tstatus\tfinish\terror\tpred_tok_s\tpred_ms\tprompt_tok_s\tserver_eval_tok_s\tserver_prompt_tok_s\tlast_state\tpressure_source\tmem_current\tmem_peak\tmax_events\toom\toom_kill\tdense_planner\tdense_ring\tdense_locked\tdense_stream_per_token\tdense_ahead\tdense_decision\tdense_repin_reason\tmoe_enabled\tmoe_budget\tmoe_resident\tmoe_evictions\tmoe_bytes_read\tmoe_prefetch_dropped\tmoe_action\tmoe_reason\tglobal_decision\treallocation_reason\tkv_offload_reason\trun_dir\n' > "$SUMMARY"

need_file() {
    local path="$1"
    [[ -e "$path" ]] || {
        echo "missing: $path" >&2
        exit 2
    }
}

kv() {
    local key="$1"
    local file="$2"
    grep -o "${key}=[^ ]*" "$file" 2>/dev/null | tail -1 | cut -d= -f2-
}

json_num() {
    local key="$1"
    local file="$2"
    grep -o "\"${key}\":[0-9.]*" "$file" 2>/dev/null | tail -1 | cut -d: -f2
}

json_str() {
    local key="$1"
    local file="$2"
    grep -o "\"${key}\":\"[^\"]*\"" "$file" 2>/dev/null | tail -1 | cut -d: -f2- | tr -d '"'
}

server_eval_tok_s() {
    local file="$1"
    grep 'eval time' "$file" 2>/dev/null | grep 'tokens per second' | tail -1 | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second).*/\1/p'
}

server_prompt_tok_s() {
    local file="$1"
    grep 'prompt eval time' "$file" 2>/dev/null | tail -1 | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second).*/\1/p'
}

wait_health() {
    local port="$1"
    local pid="$2"
    local log="$3"
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 2
        fi
        if grep -q "couldn't bind HTTP server socket\\|exiting due to HTTP server error" "$log" 2>/dev/null; then
            return 3
        fi
        if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

summarize() {
    local model="$1"
    local mem_mb="$2"
    local high_mb="$3"
    local status="$4"
    local run_dir="$5"
    local log="$run_dir/server.log"
    local out="$run_dir/completion.json"
    local events="$run_dir/memory.events"
    local current="$run_dir/memory.current"
    local flex_line="$run_dir/flex.line"

    grep 'llama_flex: layers=' "$log" 2>/dev/null | tail -1 > "$flex_line"
    local finish error pred_tok_s pred_ms prompt_tok_s eval_tok_s prompt_srv
    finish="$(json_str finish_reason "$out")"; finish="${finish:-NA}"
    error="$(json_str error "$out")"; error="${error:-0}"
    pred_tok_s="$(json_num predicted_per_second "$out")"; pred_tok_s="${pred_tok_s:-NA}"
    pred_ms="$(json_num predicted_ms "$out")"; pred_ms="${pred_ms:-NA}"
    prompt_tok_s="$(json_num prompt_per_second "$out")"; prompt_tok_s="${prompt_tok_s:-NA}"
    eval_tok_s="$(server_eval_tok_s "$log")"; eval_tok_s="${eval_tok_s:-NA}"
    prompt_srv="$(server_prompt_tok_s "$log")"; prompt_srv="${prompt_srv:-NA}"

    local max_events oom oom_kill mem_current mem_peak
    max_events="$(awk '$1=="max"{print $2}' "$events" 2>/dev/null | tail -1)"; max_events="${max_events:-NA}"
    oom="$(awk '$1=="oom"{print $2}' "$events" 2>/dev/null | tail -1)"; oom="${oom:-NA}"
    oom_kill="$(awk '$1=="oom_kill"{print $2}' "$events" 2>/dev/null | tail -1)"; oom_kill="${oom_kill:-NA}"
    mem_current="$(cat "$current" 2>/dev/null)"; mem_current="${mem_current:-NA}"
    mem_peak="$(awk '/memory_current/ {print}' "$run_dir/cgroup-samples.tsv" 2>/dev/null | tail -1 | cut -f2)"; mem_peak="${mem_peak:-NA}"

    local last_obs="$run_dir/last-observe.log"
    grep 'memory_governor_observe' "$log" 2>/dev/null | tail -1 > "$last_obs"

    local dense_planner dense_ring dense_locked dense_stream dense_ahead
    dense_planner="$(grep -q 'planner=1' "$flex_line" && echo 1 || echo 0)"
    dense_ring="$(grep -o 'ring=[0-9]*' "$flex_line" | tail -1 | cut -d= -f2)"; dense_ring="${dense_ring:-NA}"
    dense_locked="$(kv dense_locked_bytes "$last_obs")"; dense_locked="${dense_locked:-NA}"
    dense_stream="$(kv dense_stream_per_token_bytes "$last_obs")"; dense_stream="${dense_stream:-NA}"
    dense_ahead="$(kv dense_effective_ahead "$last_obs")"; dense_ahead="${dense_ahead:-NA}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model" "$mem_mb" "$high_mb" "$status" "$finish" "$error" "$pred_tok_s" "$pred_ms" "$prompt_tok_s" \
        "$eval_tok_s" "$prompt_srv" \
        "$(kv effective_pressure_state "$last_obs")" \
        "$(kv pressure_source "$last_obs")" \
        "$mem_current" "$mem_peak" "$max_events" "$oom" "$oom_kill" \
        "$dense_planner" "$dense_ring" "$dense_locked" "$dense_stream" "$dense_ahead" \
        "$(kv global_optimizer_decision "$last_obs")" \
        "$(kv dense_repin_reason "$last_obs")" \
        "$(kv moe_enabled "$last_obs")" \
        "$(kv moe_budget_bytes "$last_obs")" \
        "$(kv moe_resident_bytes "$last_obs")" \
        "$(kv moe_evictions "$last_obs")" \
        "$(kv moe_bytes_read "$last_obs")" \
        "$(kv moe_prefetch_budget_dropped "$last_obs")" \
        "$(kv moe_budget_action "$last_obs")" \
        "$(kv moe_budget_reason "$last_obs")" \
        "$(kv global_optimizer_decision "$last_obs")" \
        "$(kv reallocation_reason "$last_obs")" \
        "$(kv kv_offload_reason "$last_obs")" \
        "$run_dir" >> "$SUMMARY"
}

run_case() {
    local model_kind="$1"
    local mem_mb="$2"
    local idx="$3"
    local port=$((BASE_PORT + idx))
    local high_mb="$mem_mb"
    if (( HIGH_MARGIN_MB > 0 && mem_mb > HIGH_MARGIN_MB )); then
        high_mb=$((mem_mb - HIGH_MARGIN_MB))
    fi
    local run_dir="$OUT_DIR/${model_kind}-${mem_mb}M"
    local cg="/sys/fs/cgroup/global-sweep-${model_kind}-${mem_mb}M-$$"
    local log="$run_dir/server.log"
    local out="$run_dir/completion.json"
    local samples="$run_dir/cgroup-samples.tsv"

    mkdir -p "$run_dir" "$cg" 2>/dev/null
    echo $((mem_mb * 1024 * 1024)) > "$cg/memory.max"
    if (( HIGH_MARGIN_MB > 0 )); then
        echo $((high_mb * 1024 * 1024)) > "$cg/memory.high" 2>/dev/null || true
    else
        echo max > "$cg/memory.high" 2>/dev/null || true
    fi
    echo 0 > "$cg/memory.oom.group" 2>/dev/null || true

    local model="$DENSE_MODEL"
    local prompt="Explain memory scheduling briefly."
    local -a env_cmd
    env_cmd=(
        env
        -u LLAMA_FLEX_LOCK_GB
        -u LLAMA_FLEX_RING
        -u LLAMA_FLEX_AHEAD
        -u LLAMA_FLEX_MAX_AHEAD
        -u LLAMA_FLEX_SCHED
        -u LLAMA_FLEX_AUTO
        LLAMA_MEMORY_GOVERNOR=1
        LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=1
        LLAMA_MEMORY_GOVERNOR_REALLOCATION=1
        LLAMA_MEMORY_GOVERNOR_DENSE_REPIN=1
        LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_ASYNC=1
        LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500
        LLAMA_FLEX_DEBUG=1
    )

    if [[ "$model_kind" == "moe" ]]; then
        model="$MOE_MODEL"
        prompt="Explain why expert caching matters."
        env_cmd+=(
            LLAMA_LAZY_V2=1
            LLAMA_LAZY_CLG=1
            LLAMA_LAZY_MOE_BUFFER=1
            LLAMA_LAZY_MOE_BUFFER_AUTO=1
            LLAMA_LAZY_MOE_BUFFER_WORKERS=4
            LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0
            LLAMA_LAZY_MOE_SIDECAR="$MOE_SIDECAR"
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED=1
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1
            LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=1
            LLAMA_MEMORY_GOVERNOR_MOE_FAST_START=1
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1
        )
    fi

    echo -e "memory_current\tbytes" > "$samples"
    (
        echo "$BASHPID" > "$cg/cgroup.procs"
        exec "${env_cmd[@]}" "$SERVER" \
            -m "$model" \
            --host 127.0.0.1 \
            --port "$port" \
            -c "$CTX" \
            -b "$BATCH" \
            -ub "$BATCH" \
            -t "$THREADS" \
            -np 1 \
            --no-webui
    ) > "$log" 2>&1 &
    local pid=$!

    local status="ok"
    if ! wait_health "$port" "$pid" "$log"; then
        status="startup_failed"
    else
        local mon_pid=""
        (
            while kill -0 "$pid" 2>/dev/null; do
                printf 'memory_current\t%s\n' "$(cat "$cg/memory.current" 2>/dev/null || echo 0)" >> "$samples"
                sleep 1
            done
        ) &
        mon_pid=$!

        curl --noproxy '*' -sS --max-time "$REQUEST_TIMEOUT_SEC" "http://127.0.0.1:${port}/v1/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"default\",\"prompt\":\"${prompt}\",\"max_tokens\":${TOKENS},\"temperature\":0}" > "$out" 2>"$run_dir/curl.err" || status="request_failed"

        kill "$mon_pid" 2>/dev/null || true
    fi

    cat "$cg/memory.events" > "$run_dir/memory.events" 2>/dev/null || true
    cat "$cg/memory.current" > "$run_dir/memory.current" 2>/dev/null || true

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    summarize "$model_kind" "$mem_mb" "$high_mb" "$status" "$run_dir"
    echo -e "${model_kind}\t${mem_mb}M\t${status}\t${run_dir}"
}

main() {
    need_file "$SERVER"
    need_file "$DENSE_MODEL"
    need_file "$MOE_MODEL"
    need_file "$MOE_SIDECAR"

    pkill -x llama-server 2>/dev/null || true
    sleep 2

    local idx=0
    local mem
    for mem in $MEMORY_MBS; do
        run_case dense "$mem" "$idx"
        idx=$((idx + 1))
        run_case moe "$mem" "$idx"
        idx=$((idx + 1))
    done

    echo "summary: $SUMMARY"
}

main "$@"
```

### memory behavior profile 脚本

来源：`scripts/run-memory-behavior-profile.py:1-1081`

```python
#!/usr/bin/env python3
"""Profile LLM inference memory behavior from llama-server logs and runtime samples.

The script is evidence-first: by default it enables only observation markers and
explicitly disables legacy destructive KV actions.  It does not create or modify
cgroups and only terminates the process group it starts.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build/bin/llama-server"
DEFAULT_PROMPT = "Explain memory scheduling briefly."

MARKERS = (
    "memory_governor_observe",
    "kv_pressure_telemetry",
    "kv_pressure_dry_run",
    "kv_pressure_bounded_release",
    "kv_pressure_unified_action",
    "KV_GOVERNOR_CAPABILITY",
    "kv_g0_s1_resident_observation",
    "kv_resume_order_event",
)

RECOMMENDED_FIELDS = {
    "memory_governor_observe": {
        "sample_count",
        "effective_pressure_state",
        "dense_resident_bytes",
        "moe_resident_bytes",
        "kv_effective_resident_bytes",
    },
    "kv_pressure_telemetry": {
        "state",
        "source",
        "sample_valid",
        "rss_kb",
        "sample_count",
    },
}

FORBIDDEN_DEFAULT_ENV = {
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0",
    "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0",
    "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
}

ACTION_ENV_KEYS = {
    "LLAMA_MEMORY_GOVERNOR_KV_RELEASE",
    "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD",
    "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM",
    "LLAMA_KV_PAGED_RELEASE",
    "LLAMA_KV_PAGED_SWAP",
}

KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not path.is_file():
        return {"path": str(path), "exists": True, "is_file": False}
    return {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256(path)}


def write_inventory(out_dir: pathlib.Path) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "inventory.sha256.json":
            continue
        rel = path.relative_to(out_dir)
        records.append({"path": str(rel), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(out_dir / "inventory.sha256.json", {"files": records})


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        fail(f"expected KEY=VALUE, got: {value}")
    key, val = value.split("=", 1)
    if not key:
        fail(f"empty env key in: {value}")
    return key, val


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "LLAMA_MEMORY_GOVERNOR": "1",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE": "1",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE_MS": str(args.observe_ms),
    })
    if args.auto_backends:
        env["LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS"] = "1"
    if args.flex_trace:
        env["LLAMA_FLEX_TRACE"] = str(pathlib.Path(args.flex_trace))

    if not args.allow_unsafe_env:
        env.update(FORBIDDEN_DEFAULT_ENV)

    for item in args.env or []:
        key, val = parse_key_value(item)
        env[key] = val

    validate_env_safety(env, args)
    return env


def validate_env_safety(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.enable_actions:
        return
    unsafe = []
    for key in ACTION_ENV_KEYS:
        if env.get(key) not in (None, "", "0", "false", "False", "FALSE"):
            unsafe.append(f"{key}={env[key]}")
    if unsafe:
        fail("action env requires --enable-actions: " + ", ".join(sorted(unsafe)))


def build_server_argv(args: argparse.Namespace, port: int) -> list[str]:
    argv = [
        str(pathlib.Path(args.binary)),
        "--model", str(pathlib.Path(args.model)),
        "--host", args.host,
        "--port", str(port),
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--threads", str(args.threads),
        "--parallel", str(args.parallel),
        "--seed", str(args.seed),
        "--temp", str(args.temp),
        "--log-verbosity", str(args.log_verbosity),
        "--no-log-prefix",
        "--no-log-timestamps",
        "--no-webui",
    ]
    if args.no_warmup:
        argv.append("--no-warmup")
    argv.extend(args.extra_server_arg or [])
    return argv


def wait_health(host: str, port: int, proc: subprocess.Popen[bytes], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    attempts = 0
    last_error = ""
    while time.monotonic() < deadline:
        attempts += 1
        if proc.poll() is not None:
            return {"ok": False, "attempts": attempts, "error": f"server exited rc={proc.returncode}"}
        conn: http.client.HTTPConnection | None = None
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1.0)
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = response.read()
            if response.status == 200:
                parsed: Any = None
                try:
                    parsed = json.loads(payload) if payload else None
                except ValueError:
                    parsed = payload.decode("utf-8", errors="replace")
                return {"ok": True, "attempts": attempts, "status": response.status, "body": parsed}
            last_error = f"HTTP {response.status}"
        except OSError as exc:
            last_error = str(exc)
        finally:
            if conn is not None:
                conn.close()
        time.sleep(0.2)
    return {"ok": False, "attempts": attempts, "error": last_error or "timeout"}


def terminate_process_group(proc: subprocess.Popen[bytes], timeout_s: float = 10.0) -> dict[str, Any]:
    if proc.poll() is not None:
        return {"already_exited": True, "returncode": proc.returncode}
    result: dict[str, Any] = {"already_exited": False}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        result["sigterm_sent"] = True
    except ProcessLookupError:
        result["sigterm_sent"] = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            result["returncode"] = proc.returncode
            return result
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        result["sigkill_sent"] = True
    except ProcessLookupError:
        result["sigkill_sent"] = False
    proc.wait(timeout=5)
    result["returncode"] = proc.returncode
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def stream_completion(host: str, port: int, prompt: str, args: argparse.Namespace,
                      raw_path: pathlib.Path, events_path: pathlib.Path) -> dict[str, Any]:
    body = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": args.stream,
        "seed": args.seed,
        "temperature": args.temp,
        "cache_prompt": args.cache_prompt,
    }
    encoded = json.dumps(body).encode("utf-8")
    started = time.monotonic_ns()
    conn = http.client.HTTPConnection(host, port, timeout=args.request_timeout_sec)
    raw = bytearray()
    records: list[dict[str, Any]] = []
    http_status: int | None = None
    response_body: Any = None
    try:
        conn.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = conn.getresponse()
        http_status = response.status
        if response.status != 200:
            payload = response.read()
            raw.extend(payload)
            fail(f"completion returned HTTP {response.status}: {payload[:200]!r}")
        if args.stream:
            while True:
                line = response.readline()
                if not line:
                    break
                raw.extend(line)
                stripped = line.rstrip(b"\r\n")
                if not stripped:
                    continue
                if not stripped.startswith(b"data: "):
                    records.append({
                        "arrival_monotonic_ns": time.monotonic_ns(),
                        "raw_payload": stripped.decode("utf-8", errors="replace"),
                        "parse_error": "non_sse_data_line",
                    })
                    continue
                payload = stripped[6:]
                record: dict[str, Any] = {
                    "arrival_monotonic_ns": time.monotonic_ns(),
                    "raw_payload": payload.decode("utf-8", errors="replace"),
                }
                try:
                    record["parsed"] = json.loads(payload)
                except ValueError as exc:
                    record["parse_error"] = str(exc)
                records.append(record)
        else:
            payload = response.read()
            raw.extend(payload)
            response_body = json.loads(payload) if payload else {}
            records.append({"arrival_monotonic_ns": time.monotonic_ns(), "parsed": response_body})
    finally:
        conn.close()
        raw_path.write_bytes(raw)
        write_json(events_path, {
            "request_started_monotonic_ns": started,
            "stream": args.stream,
            "http_status": http_status,
            "events": records,
            "response_body": response_body,
            "stream_ended_monotonic_ns": time.monotonic_ns(),
            "clock": "time.monotonic_ns",
        })
    return derive_runner_metrics(events_path)


def derive_runner_metrics(events_path: pathlib.Path) -> dict[str, Any]:
    evidence = json.loads(events_path.read_text(encoding="utf-8"))
    events = evidence.get("events", [])
    parsed = [e.get("parsed") for e in events if isinstance(e.get("parsed"), dict)]
    timings = None
    for obj in reversed(parsed):
        if isinstance(obj.get("timings"), dict):
            timings = obj["timings"]
            break
    arrivals = [int(e["arrival_monotonic_ns"]) for e in events if "arrival_monotonic_ns" in e]
    intervals_ms = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])]
    metrics: dict[str, Any] = {
        "http_status": evidence.get("http_status"),
        "event_count": len(events),
        "ttft_ms": ((arrivals[0] - evidence["request_started_monotonic_ns"]) / 1e6) if arrivals else None,
        "wall_ms": ((evidence["stream_ended_monotonic_ns"] - evidence["request_started_monotonic_ns"]) / 1e6),
        "sse_chunk_interval_p50_ms": percentile(intervals_ms, 0.50),
        "sse_chunk_interval_p95_ms": percentile(intervals_ms, 0.95),
        "sse_chunk_interval_p99_ms": percentile(intervals_ms, 0.99),
        "server_timings": timings,
    }
    if timings:
        for key in ("predicted_n", "predicted_ms", "predicted_per_second", "prompt_n", "prompt_ms", "prompt_per_second"):
            if key in timings:
                metrics[key] = timings[key]
        predicted_n = timings.get("predicted_n")
        predicted_ms = timings.get("predicted_ms")
        if isinstance(predicted_n, (int, float)) and isinstance(predicted_ms, (int, float)) and predicted_n > 1:
            metrics["server_tpot_ms"] = predicted_ms / max(1, predicted_n - 1)
    return metrics


def parse_colon_file(path: pathlib.Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    except OSError:
        pass
    return data


def read_proc_status(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/status"))


def read_proc_smaps_rollup(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/smaps_rollup"))


def read_proc_io(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/io"))


def kb_value(value: str | None) -> str:
    if not value:
        return "NA"
    return value.split()[0] if value.split() else "NA"


def plain_int(value: str | None) -> str:
    if value is None:
        return "NA"
    first = value.split()[0] if value.split() else value
    return first if re.fullmatch(r"-?\d+", first) else "NA"


def read_cgroup_events(path: pathlib.Path) -> dict[str, str]:
    result = {"low": "NA", "high": "NA", "max": "NA", "oom": "NA", "oom_kill": "NA"}
    try:
        for line in (path / "memory.events").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in result:
                result[parts[0]] = parts[1]
    except OSError:
        pass
    return result


def read_cgroup_scalar(path: pathlib.Path, name: str) -> str:
    try:
        return (path / name).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "NA"


class SamplerThread(threading.Thread):
    def __init__(self, pid: int, interval_ms: int, out_dir: pathlib.Path, cgroup_path: pathlib.Path | None):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_s = max(0.01, interval_ms / 1000.0)
        self.out_dir = out_dir
        self.cgroup_path = cgroup_path
        self.stop_event = threading.Event()
        self.warnings: list[str] = []

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        proc_path = self.out_dir / "proc-samples.tsv"
        cg_path = self.out_dir / "cgroup-samples.tsv"
        with proc_path.open("w", encoding="utf-8") as proc_handle:
            proc_handle.write("monotonic_ns\tpid\tVmRSS_kB\tVmHWM_kB\tRssAnon_kB\tRssFile_kB\tRssShmem_kB\tsmaps_Rss_kB\tsmaps_Pss_kB\tread_bytes\twrite_bytes\tsyscr\tsyscw\n")
            cg_handle = cg_path.open("w", encoding="utf-8")
            try:
                cg_handle.write("monotonic_ns\tmemory_current\tmemory_peak\tmemory_high\tmemory_max\tevents_low\tevents_high\tevents_max\tevents_oom\tevents_oom_kill\n")
                while not self.stop_event.is_set():
                    ts = time.monotonic_ns()
                    status = read_proc_status(self.pid)
                    smaps = read_proc_smaps_rollup(self.pid)
                    io = read_proc_io(self.pid)
                    proc_handle.write("\t".join([
                        str(ts), str(self.pid),
                        kb_value(status.get("VmRSS")), kb_value(status.get("VmHWM")),
                        kb_value(status.get("RssAnon")), kb_value(status.get("RssFile")), kb_value(status.get("RssShmem")),
                        kb_value(smaps.get("Rss")), kb_value(smaps.get("Pss")),
                        plain_int(io.get("read_bytes")), plain_int(io.get("write_bytes")),
                        plain_int(io.get("syscr")), plain_int(io.get("syscw")),
                    ]) + "\n")
                    proc_handle.flush()
                    if self.cgroup_path is not None:
                        events = read_cgroup_events(self.cgroup_path)
                        cg_handle.write("\t".join([
                            str(ts),
                            read_cgroup_scalar(self.cgroup_path, "memory.current"),
                            read_cgroup_scalar(self.cgroup_path, "memory.peak"),
                            read_cgroup_scalar(self.cgroup_path, "memory.high"),
                            read_cgroup_scalar(self.cgroup_path, "memory.max"),
                            events["low"], events["high"], events["max"], events["oom"], events["oom_kill"],
                        ]) + "\n")
                        cg_handle.flush()
                    time.sleep(self.interval_s)
            finally:
                cg_handle.close()


def coerce_value(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE]-?\d+)?", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_marker_line(line: str, marker: str) -> dict[str, Any] | None:
    idx = line.find(marker)
    if idx < 0:
        return None
    before = line[:idx]
    after_idx = idx + len(marker)
    if before and not before[-1].isspace():
        return None
    if after_idx < len(line) and not line[after_idx].isspace():
        return None
    rest = line[after_idx:].strip()
    fields: dict[str, Any] = {}
    duplicate_keys: list[str] = []
    unparsed_tokens: list[str] = []
    for token in rest.split():
        if token.count("=") != 1:
            unparsed_tokens.append(token)
            continue
        key, value = token.split("=", 1)
        if not key or not KEY_RE.match(key):
            unparsed_tokens.append(token)
            continue
        if key in fields:
            duplicate_keys.append(key)
        fields[key] = coerce_value(value)
    missing = sorted(RECOMMENDED_FIELDS.get(marker, set()) - set(fields))
    return {
        "marker": marker,
        "fields": fields,
        "duplicate_keys": duplicate_keys,
        "unparsed_tokens": unparsed_tokens,
        "missing_recommended_fields": missing,
    }


def parse_log_file(log_path: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    offset = 0
    line_no = 0
    try:
        with log_path.open("rb") as handle:
            for raw_line in handle:
                line_no += 1
                start = offset
                end = start + len(raw_line)
                offset = end
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                for marker in MARKERS:
                    parsed = parse_marker_line(line, marker)
                    if parsed is None:
                        continue
                    parsed.update({
                        "line_no": line_no,
                        "byte_start": start,
                        "byte_end": end,
                        "raw": line,
                    })
                    if parsed["duplicate_keys"]:
                        warnings.append(f"line {line_no}: duplicate keys in {marker}: {','.join(parsed['duplicate_keys'])}")
                    if parsed["unparsed_tokens"]:
                        warnings.append(f"line {line_no}: unparsed tokens in {marker}: {','.join(parsed['unparsed_tokens'][:5])}")
                    if parsed["missing_recommended_fields"]:
                        warnings.append(f"line {line_no}: missing recommended fields in {marker}: {','.join(parsed['missing_recommended_fields'])}")
                    events.append(parsed)
                    break
    except OSError as exc:
        fail(f"cannot read log {log_path}: {exc}")
    if not any(e["marker"] == "memory_governor_observe" for e in events):
        warnings.append("no memory_governor_observe markers found")
    return events, warnings


def write_events_jsonl(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def marker_events(events: list[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("marker") == marker]


def field_values(events: list[dict[str, Any]], key: str) -> list[Any]:
    values = []
    for event in events:
        fields = event.get("fields", {})
        if key in fields:
            values.append(fields[key])
    return values


def numeric_values(events: list[dict[str, Any]], key: str) -> list[int | float]:
    return [v for v in field_values(events, key) if isinstance(v, (int, float))]


def last_value(events: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    values = field_values(events, key)
    return values[-1] if values else default


def max_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    return max(values) if values else None


def min_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    return min(values) if values else None


def delta_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    if not values:
        return None
    return values[-1] - values[0]


def count_truthy(events: list[dict[str, Any]], key: str) -> int:
    count = 0
    for value in field_values(events, key):
        if value in (1, "1", True, "true", "True", "yes", "on"):
            count += 1
    return count


def aggregate_dense(obs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "enabled_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "dense_flex_enabled")),
        "resident_bytes_max": max_value(obs, "dense_resident_bytes"),
        "resident_bytes_last": last_value(obs, "dense_resident_bytes"),
        "reclaimable_bytes_max": max_value(obs, "dense_reclaimable_bytes"),
        "ring_bytes_max": max_value(obs, "dense_ring_bytes"),
        "locked_bytes_last": last_value(obs, "dense_locked_bytes"),
        "delta_locked_bytes_last": last_value(obs, "dense_delta_locked_bytes"),
        "stream_per_token_bytes_last": last_value(obs, "dense_stream_per_token_bytes"),
        "bytes_streamed_delta": delta_value(obs, "dense_bytes_streamed"),
        "bytes_read_phys_delta": delta_value(obs, "dense_bytes_read_phys"),
        "read_ops_delta": delta_value(obs, "dense_read_ops"),
        "effective_ahead_last": last_value(obs, "dense_effective_ahead"),
        "prefetch_budget_dropped_delta": delta_value(obs, "dense_prefetch_budget_dropped"),
        "repin_attempts": count_truthy(obs, "dense_repin_attempted"),
        "resize_attempts": count_truthy(obs, "dense_resize_attempted"),
        "repin_reason_last": last_value(obs, "dense_repin_reason"),
        "resize_reason_last": last_value(obs, "dense_resize_reason"),
    }


def aggregate_moe(obs: list[dict[str, Any]]) -> dict[str, Any]:
    hits_last = last_value(obs, "moe_cache_hits", 0)
    misses_last = last_value(obs, "moe_cache_misses", 0)
    hit_rate = None
    if isinstance(hits_last, (int, float)) and isinstance(misses_last, (int, float)) and hits_last + misses_last > 0:
        hit_rate = hits_last / (hits_last + misses_last)
    actions = Counter(str(v) for v in field_values(obs, "moe_budget_action"))
    return {
        "enabled_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "moe_enabled")),
        "resident_bytes_max": max_value(obs, "moe_resident_bytes"),
        "resident_bytes_last": last_value(obs, "moe_resident_bytes"),
        "budget_bytes_last": last_value(obs, "moe_budget_bytes"),
        "budget_unbounded_last": last_value(obs, "moe_budget_unbounded"),
        "expert_bytes_last": last_value(obs, "moe_expert_bytes"),
        "streams_delta": delta_value(obs, "moe_streams"),
        "hits_delta": delta_value(obs, "moe_hits"),
        "evictions_delta": delta_value(obs, "moe_evictions"),
        "bytes_read_delta": delta_value(obs, "moe_bytes_read"),
        "cache_hits_last": hits_last,
        "cache_misses_last": misses_last,
        "cache_hit_rate": hit_rate,
        "prefetch_hits_delta": delta_value(obs, "moe_prefetch_hits"),
        "prefetch_late_delta": delta_value(obs, "moe_prefetch_late"),
        "prefetch_unused_delta": delta_value(obs, "moe_prefetch_unused"),
        "warm_working_set_bytes_last": last_value(obs, "moe_warm_working_set_bytes"),
        "warm_working_set_groups_last": last_value(obs, "moe_warm_working_set_groups"),
        "budget_actions": dict(actions),
        "budget_reason_last": last_value(obs, "moe_budget_reason"),
    }


def aggregate_kv(obs: list[dict[str, Any]]) -> dict[str, Any]:
    sources = sorted({str(v) for v in field_values(obs, "kv_effective_budget_source")})
    return {
        "memory_present_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "kv_memory_present")),
        "resident_bytes_max": max_value(obs, "kv_resident_bytes"),
        "resident_bytes_last": last_value(obs, "kv_resident_bytes"),
        "reclaimable_resident_bytes_max": max_value(obs, "kv_reclaimable_resident_bytes"),
        "slot_budget_valid_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "kv_slot_budget_valid")),
        "slot_resident_bytes_max": max_value(obs, "kv_slot_resident_bytes"),
        "slot_reclaimable_resident_bytes_max": max_value(obs, "kv_slot_reclaimable_resident_bytes"),
        "effective_budget_sources_seen": sources,
        "effective_resident_bytes_max": max_value(obs, "kv_effective_resident_bytes"),
        "effective_reclaimable_resident_bytes_max": max_value(obs, "kv_effective_reclaimable_resident_bytes"),
        "soft_excess_bytes_max": max_value(obs, "kv_soft_excess_bytes"),
        "release_attempts": count_truthy(obs, "kv_release_attempted"),
        "release_relieved_bytes_sum": sum(v for v in numeric_values(obs, "kv_release_relieved_bytes")),
        "release_reason_last": last_value(obs, "kv_release_reason"),
        "offload_attempts": count_truthy(obs, "kv_offload_attempted"),
        "offload_relieved_bytes_sum": sum(v for v in numeric_values(obs, "kv_offload_relieved_bytes")),
        "offload_backend_last": last_value(obs, "kv_offload_backend"),
        "offload_reason_last": last_value(obs, "kv_offload_reason"),
    }


def aggregate_pressure(obs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "states_seen": sorted({str(v) for v in field_values(obs, "effective_pressure_state")}),
        "reason_last": last_value(obs, "effective_pressure_reason"),
        "pressure_current_bytes_max": max_value(obs, "pressure_current_bytes"),
        "pressure_excess_bytes_max": max_value(obs, "pressure_excess_bytes"),
        "rss_observed_bytes_max": max_value(obs, "rss_observed_bytes"),
        "prefetch_budget_effective_bytes_last": last_value(obs, "prefetch_budget_effective_bytes"),
        "prefetch_budget_tick_bytes_last": last_value(obs, "prefetch_budget_tick_bytes"),
        "prefetch_budget_dense_bytes_last": last_value(obs, "prefetch_budget_dense_bytes"),
        "prefetch_budget_moe_bytes_last": last_value(obs, "prefetch_budget_moe_bytes"),
        "prefetch_budget_kv_resume_used_bytes_delta": delta_value(obs, "prefetch_budget_kv_resume_used_bytes"),
        "auction_selected_allocation_counts": dict(Counter(str(v) for v in field_values(obs, "auction_selected_allocation"))),
        "auction_selected_reclaim_counts": dict(Counter(str(v) for v in field_values(obs, "auction_selected_reclaim"))),
        "global_optimizer_decision_last": last_value(obs, "global_optimizer_decision"),
        "reallocation_reason_last": last_value(obs, "reallocation_reason"),
    }


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        records.append({key: parts[i] if i < len(parts) else "NA" for i, key in enumerate(header)})
    return records


def int_or_none(value: str | None) -> int | None:
    if value is None or value == "NA" or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def aggregate_tsv_samples(path: pathlib.Path, fields: list[str], deltas: list[str] | None = None) -> dict[str, Any]:
    records = read_tsv(path)
    result: dict[str, Any] = {"sample_count": len(records)}
    for field in fields:
        vals = [v for v in (int_or_none(r.get(field)) for r in records) if v is not None]
        result[f"{field}_max"] = max(vals) if vals else None
        result[f"{field}_last"] = vals[-1] if vals else None
    for field in deltas or []:
        vals = [v for v in (int_or_none(r.get(field)) for r in records) if v is not None]
        result[f"{field}_delta"] = vals[-1] - vals[0] if vals else None
    return result


def aggregate_all(events: list[dict[str, Any]], out_dir: pathlib.Path, mode: str, warnings: list[str],
                  request_metrics: dict[str, Any] | None, action_enabled: bool,
                  inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    obs = marker_events(events, "memory_governor_observe")
    marker_counts = Counter(e["marker"] for e in events)
    summary = {
        "schema": "memory_behavior_profile/v1",
        "mode": mode,
        "observation_only": not action_enabled,
        "action_enabled": action_enabled,
        "inputs": inputs or {},
        "request_metrics": request_metrics or {},
        "markers": {"counts": dict(marker_counts)},
        "dense": aggregate_dense(obs),
        "moe": aggregate_moe(obs),
        "kv": aggregate_kv(obs),
        "pressure": aggregate_pressure(obs),
        "proc": aggregate_tsv_samples(out_dir / "proc-samples.tsv", ["VmRSS_kB", "VmHWM_kB", "RssAnon_kB", "RssFile_kB", "smaps_Rss_kB"], ["read_bytes", "write_bytes", "syscr", "syscw"]),
        "cgroup": aggregate_tsv_samples(out_dir / "cgroup-samples.tsv", ["memory_current", "memory_peak"], ["events_high", "events_max", "events_oom", "events_oom_kill"]),
        "warnings": warnings,
        "evidence_limits": [
            "memory_governor_observe marker is sampled; it is not a per-token complete trace.",
            "MoE expert activation is inferred from runtime counters; exact router logits are not captured.",
            "Dense parameter loading is inferred from Dense Flex counters; this is not hardware PMU-level memory tracing.",
            "KV usage is inferred from resident/reclaimable/slot/action markers; observation-only runs do not validate release/offload correctness.",
            "This script records evidence and does not claim performance improvement.",
        ],
    }
    return summary


def write_summary_tsv(path: pathlib.Path, summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    def get(path_str: str) -> Any:
        cur: Any = summary
        for part in path_str.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    columns = [
        ("status", "ok"),
        ("mode", "mode"),
        ("observation_only", "observation_only"),
        ("marker_memory_governor_observe_count", "markers.counts.memory_governor_observe"),
        ("marker_kv_pressure_telemetry_count", "markers.counts.kv_pressure_telemetry"),
        ("ttft_ms", "request_metrics.ttft_ms"),
        ("tpot_ms", "request_metrics.server_tpot_ms"),
        ("predicted_per_second", "request_metrics.predicted_per_second"),
        ("rss_observed_bytes_max", "pressure.rss_observed_bytes_max"),
        ("proc_VmRSS_kB_max", "proc.VmRSS_kB_max"),
        ("cgroup_memory_current_max", "cgroup.memory_current_max"),
        ("dense_enabled", "dense.enabled_seen"),
        ("dense_resident_bytes_max", "dense.resident_bytes_max"),
        ("dense_locked_bytes_last", "dense.locked_bytes_last"),
        ("dense_bytes_read_phys_delta", "dense.bytes_read_phys_delta"),
        ("dense_read_ops_delta", "dense.read_ops_delta"),
        ("dense_prefetch_budget_dropped_delta", "dense.prefetch_budget_dropped_delta"),
        ("moe_enabled", "moe.enabled_seen"),
        ("moe_resident_bytes_max", "moe.resident_bytes_max"),
        ("moe_bytes_read_delta", "moe.bytes_read_delta"),
        ("moe_streams_delta", "moe.streams_delta"),
        ("moe_evictions_delta", "moe.evictions_delta"),
        ("moe_cache_hit_rate", "moe.cache_hit_rate"),
        ("kv_memory_present", "kv.memory_present_seen"),
        ("kv_resident_bytes_max", "kv.resident_bytes_max"),
        ("kv_slot_resident_bytes_max", "kv.slot_resident_bytes_max"),
        ("kv_reclaimable_resident_bytes_max", "kv.reclaimable_resident_bytes_max"),
        ("kv_release_attempts", "kv.release_attempts"),
        ("kv_offload_attempts", "kv.offload_attempts"),
        ("pressure_states_seen", "pressure.states_seen"),
        ("prefetch_budget_effective_bytes_last", "pressure.prefetch_budget_effective_bytes_last"),
        ("warnings_count", "warnings"),
        ("out_dir", str(out_dir)),
    ]
    headers = [name for name, _ in columns]
    values = []
    for _, spec in columns:
        if spec == "ok" or spec == str(out_dir):
            value = spec
        elif spec == "warnings":
            value = len(summary.get("warnings", []))
        else:
            value = get(spec)
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif value is None:
            value = "NA"
        values.append(str(value))
    path.write_text("\t".join(headers) + "\n" + "\t".join(values) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "NA"
    return str(value)


def write_report_md(path: pathlib.Path, summary: dict[str, Any]) -> None:
    dense = summary["dense"]
    moe = summary["moe"]
    kv = summary["kv"]
    pressure = summary["pressure"]
    req = summary.get("request_metrics", {})
    lines = [
        "# LLM Memory Behavior Profile",
        "",
        "## 结论摘要",
        "",
        f"- 运行模式：`{summary['mode']}`。",
        f"- observation-only：`{summary['observation_only']}`；action-enabled：`{summary['action_enabled']}`。",
        f"- `memory_governor_observe` 条数：`{summary['markers']['counts'].get('memory_governor_observe', 0)}`。",
        f"- `kv_pressure_telemetry` 条数：`{summary['markers']['counts'].get('kv_pressure_telemetry', 0)}`。",
        "",
        "## 请求指标",
        "",
        f"- TTFT(ms)：{fmt(req.get('ttft_ms'))}",
        f"- Server TPOT(ms)：{fmt(req.get('server_tpot_ms'))}",
        f"- predicted tokens/s：{fmt(req.get('predicted_per_second'))}",
        "",
        "## Dense 参数加载观察",
        "",
        f"- Dense Flex enabled seen：{fmt(dense.get('enabled_seen'))}",
        f"- resident bytes max：{fmt(dense.get('resident_bytes_max'))}",
        f"- locked bytes last：{fmt(dense.get('locked_bytes_last'))}",
        f"- bytes read phys delta：{fmt(dense.get('bytes_read_phys_delta'))}",
        f"- read ops delta：{fmt(dense.get('read_ops_delta'))}",
        f"- effective ahead last：{fmt(dense.get('effective_ahead_last'))}",
        f"- prefetch dropped delta：{fmt(dense.get('prefetch_budget_dropped_delta'))}",
        "",
        "## MoE 专家激活/缓存观察",
        "",
        f"- MoE enabled seen：{fmt(moe.get('enabled_seen'))}",
        f"- resident bytes max：{fmt(moe.get('resident_bytes_max'))}",
        f"- budget bytes last：{fmt(moe.get('budget_bytes_last'))}",
        f"- streams delta：{fmt(moe.get('streams_delta'))}",
        f"- bytes read delta：{fmt(moe.get('bytes_read_delta'))}",
        f"- evictions delta：{fmt(moe.get('evictions_delta'))}",
        f"- cache hit rate：{fmt(moe.get('cache_hit_rate'))}",
        f"- prefetch late delta：{fmt(moe.get('prefetch_late_delta'))}",
        "",
        "## KV Cache 使用观察",
        "",
        f"- KV memory present seen：{fmt(kv.get('memory_present_seen'))}",
        f"- resident bytes max：{fmt(kv.get('resident_bytes_max'))}",
        f"- slot resident bytes max：{fmt(kv.get('slot_resident_bytes_max'))}",
        f"- reclaimable resident bytes max：{fmt(kv.get('reclaimable_resident_bytes_max'))}",
        f"- effective budget sources：{fmt(kv.get('effective_budget_sources_seen'))}",
        f"- release attempts：{fmt(kv.get('release_attempts'))}",
        f"- offload attempts：{fmt(kv.get('offload_attempts'))}",
        "",
        "## Pressure / Prefetch / Auction 观察",
        "",
        f"- pressure states seen：{fmt(pressure.get('states_seen'))}",
        f"- pressure current bytes max：{fmt(pressure.get('pressure_current_bytes_max'))}",
        f"- pressure excess bytes max：{fmt(pressure.get('pressure_excess_bytes_max'))}",
        f"- rss observed bytes max：{fmt(pressure.get('rss_observed_bytes_max'))}",
        f"- prefetch budget effective bytes last：{fmt(pressure.get('prefetch_budget_effective_bytes_last'))}",
        f"- global optimizer decision last：{fmt(pressure.get('global_optimizer_decision_last'))}",
        "",
        "## Proc / Cgroup 观察",
        "",
        f"- proc samples：{fmt(summary['proc'].get('sample_count'))}",
        f"- proc VmRSS_kB max：{fmt(summary['proc'].get('VmRSS_kB_max'))}",
        f"- proc read_bytes delta：{fmt(summary['proc'].get('read_bytes_delta'))}",
        f"- cgroup samples：{fmt(summary['cgroup'].get('sample_count'))}",
        f"- cgroup memory.current max：{fmt(summary['cgroup'].get('memory_current_max'))}",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("evidence_limits", []))
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in summary["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: pathlib.Path, events: list[dict[str, Any]], warnings: list[str], mode: str,
                  request_metrics: dict[str, Any] | None, action_enabled: bool,
                  inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    write_events_jsonl(out_dir / "memory-events.jsonl", events)
    summary = aggregate_all(events, out_dir, mode, warnings, request_metrics, action_enabled, inputs)
    write_json(out_dir / "summary.json", summary)
    write_summary_tsv(out_dir / "summary.tsv", summary, out_dir)
    write_report_md(out_dir / "report.md", summary)
    write_inventory(out_dir)
    return summary


def command_parse(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = pathlib.Path(args.log)
    write_json(out_dir / "parse-input.json", {"log": str(log), "stdout": args.stdout})
    events, warnings = parse_log_file(log)
    if args.strict and not marker_events(events, "memory_governor_observe"):
        fail("strict mode requires at least one memory_governor_observe marker")
    write_outputs(out_dir, events, warnings, "parse", None, False, {"log": identity(log)})
    print(f"wrote {out_dir}")


def command_run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = pathlib.Path(args.binary)
    model = pathlib.Path(args.model)
    if not binary.is_file():
        fail(f"missing binary: {binary}")
    if not model.is_file():
        fail(f"missing model: {model}")
    port = args.port or free_port()
    argv = build_server_argv(args, port)
    env = build_env(args)
    prompt = load_prompt(args)
    action_enabled = bool(args.enable_actions)
    execution = {
        "argv": argv,
        "cwd": str(ROOT),
        "host": args.host,
        "port": port,
        "dry_run": args.dry_run,
        "started_wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    effective_env = {key: env[key] for key in sorted(env) if key.startswith("LLAMA_")}
    write_json(out_dir / "execution.json", execution)
    write_json(out_dir / "environment.json", {"effective_llama_env": effective_env})
    write_json(out_dir / "request.json", {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "n_predict": args.n_predict, "stream": args.stream})
    if args.dry_run:
        write_inventory(out_dir)
        print(f"dry-run wrote {out_dir}")
        return

    stdout_path = out_dir / "server.stdout"
    stderr_path = out_dir / "server.stderr"
    proc: subprocess.Popen[bytes] | None = None
    sampler: SamplerThread | None = None
    request_metrics: dict[str, Any] | None = None
    lifecycle: dict[str, Any] = {}
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            write_json(out_dir / "process.json", {"pid": proc.pid, "binary": identity(binary), "model": identity(model)})
            cgroup_path = pathlib.Path(args.cgroup_path) if args.cgroup_path else None
            if cgroup_path is not None and not cgroup_path.exists():
                cgroup_path = None
                lifecycle.setdefault("warnings", []).append("cgroup path does not exist; cgroup sampling disabled")
            sampler = SamplerThread(proc.pid, args.sample_ms, out_dir, cgroup_path)
            sampler.start()
            health = wait_health(args.host, port, proc, args.startup_timeout_sec)
            write_json(out_dir / "health.json", health)
            if not health.get("ok"):
                fail(f"server health failed: {health}")
            request_metrics = stream_completion(args.host, port, prompt, args, out_dir / ("response.sse" if args.stream else "response.json"), out_dir / "response.events.json")
            lifecycle["request_metrics"] = request_metrics
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.join(timeout=5)
        if proc is not None:
            lifecycle["termination"] = terminate_process_group(proc)
        write_json(out_dir / "execution-result.json", lifecycle)

    events, warnings = parse_log_file(stderr_path)
    warnings.extend(lifecycle.get("warnings", []))
    inputs = {
        "binary": identity(binary),
        "model": identity(model),
        "argv": argv,
        "env_effective": effective_env,
    }
    write_outputs(out_dir, events, warnings, "run", request_metrics, action_enabled, inputs)
    print(f"wrote {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile llama-server memory behavior from runtime markers and samples.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start llama-server, run one completion, and profile memory behavior")
    run.add_argument("--binary", default=str(DEFAULT_BINARY))
    run.add_argument("--model", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--ubatch-size", type=int, default=64)
    run.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--temp", type=float, default=0.0)
    run.add_argument("--n-predict", type=int, default=64)
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--prompt-file")
    stream_group = run.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=True)
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    run.add_argument("--cache-prompt", action="store_true")
    run.add_argument("--startup-timeout-sec", type=float, default=240.0)
    run.add_argument("--request-timeout-sec", type=float, default=300.0)
    run.add_argument("--log-verbosity", type=int, default=4)
    run.add_argument("--no-warmup", action="store_true", default=True)
    run.add_argument("--observe-ms", type=int, default=500)
    run.add_argument("--auto-backends", action="store_true")
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--flex-trace")
    run.add_argument("--sample-ms", type=int, default=200)
    run.add_argument("--cgroup-path")
    run.add_argument("--enable-actions", action="store_true")
    run.add_argument("--allow-unsafe-env", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--extra-server-arg", action="append", default=[])

    parse = sub.add_parser("parse", help="parse an existing llama-server log")
    parse.add_argument("--log", required=True)
    parse.add_argument("--stdout")
    parse.add_argument("--out-dir", required=True)
    parse.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        if args.prompt_file and args.prompt != DEFAULT_PROMPT:
            fail("--prompt and --prompt-file are mutually exclusive")
        command_run(args)
    elif args.command == "parse":
        command_parse(args)
    else:
        fail(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
```

### memory behavior profile parser 测试

来源：`tests/test-memory-behavior-profile-parser.py:1-90`

```python
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run-memory-behavior-profile.py"

spec = importlib.util.spec_from_file_location("memory_behavior_profile", SCRIPT)
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class MemoryBehaviorProfileParserTest(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.stderr"
            path.write_text(text, encoding="utf-8")
            return profile.parse_log_file(path)

    def test_parse_prefixed_memory_governor_marker(self):
        events, warnings = self.parse_text(
            "srv update_slots: memory_governor_observe sample_count=1 "
            "effective_pressure_state=NORMAL dense_resident_bytes=100 "
            "moe_resident_bytes=0 kv_effective_resident_bytes=200\n"
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["marker"], "memory_governor_observe")
        self.assertEqual(events[0]["fields"]["dense_resident_bytes"], 100)
        self.assertEqual(events[0]["fields"]["kv_effective_resident_bytes"], 200)
        self.assertFalse(any("no memory_governor_observe" in item for item in warnings))

    def test_aggregate_dense_moe_kv_deltas(self):
        events, _ = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_flex_enabled=1 dense_resident_bytes=100 dense_bytes_read_phys=10 dense_read_ops=1 "
            "moe_enabled=1 moe_resident_bytes=50 moe_bytes_read=20 moe_streams=2 moe_evictions=0 "
            "moe_cache_hits=3 moe_cache_misses=1 kv_memory_present=1 kv_resident_bytes=70 "
            "kv_reclaimable_resident_bytes=7 kv_slot_resident_bytes=30 kv_effective_resident_bytes=70\n"
            "memory_governor_observe sample_count=2 effective_pressure_state=PRESSURE "
            "dense_flex_enabled=1 dense_resident_bytes=150 dense_bytes_read_phys=40 dense_read_ops=4 "
            "moe_enabled=1 moe_resident_bytes=80 moe_bytes_read=50 moe_streams=5 moe_evictions=2 "
            "moe_cache_hits=7 moe_cache_misses=3 kv_memory_present=1 kv_resident_bytes=90 "
            "kv_reclaimable_resident_bytes=9 kv_slot_resident_bytes=45 kv_effective_resident_bytes=90\n"
        )
        obs = profile.marker_events(events, "memory_governor_observe")
        dense = profile.aggregate_dense(obs)
        moe = profile.aggregate_moe(obs)
        kv = profile.aggregate_kv(obs)
        pressure = profile.aggregate_pressure(obs)
        self.assertEqual(dense["resident_bytes_max"], 150)
        self.assertEqual(dense["bytes_read_phys_delta"], 30)
        self.assertEqual(dense["read_ops_delta"], 3)
        self.assertEqual(moe["resident_bytes_max"], 80)
        self.assertEqual(moe["bytes_read_delta"], 30)
        self.assertEqual(moe["streams_delta"], 3)
        self.assertEqual(moe["evictions_delta"], 2)
        self.assertAlmostEqual(moe["cache_hit_rate"], 0.7)
        self.assertEqual(kv["resident_bytes_max"], 90)
        self.assertEqual(kv["slot_resident_bytes_max"], 45)
        self.assertEqual(pressure["states_seen"], ["NORMAL", "PRESSURE"])

    def test_unknown_field_is_preserved(self):
        events, _ = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_resident_bytes=1 moe_resident_bytes=2 kv_effective_resident_bytes=3 future_new_field=abc\n"
        )
        self.assertEqual(events[0]["fields"]["future_new_field"], "abc")

    def test_duplicate_key_records_warning(self):
        events, warnings = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_resident_bytes=1 dense_resident_bytes=2 moe_resident_bytes=0 kv_effective_resident_bytes=0\n"
        )
        self.assertEqual(events[0]["fields"]["dense_resident_bytes"], 2)
        self.assertIn("dense_resident_bytes", events[0]["duplicate_keys"])
        self.assertTrue(any("duplicate keys" in item for item in warnings))

    def test_no_marker_warns_but_parses(self):
        events, warnings = self.parse_text("ordinary log line\n")
        self.assertEqual(events, [])
        self.assertTrue(any("no memory_governor_observe markers found" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
```

