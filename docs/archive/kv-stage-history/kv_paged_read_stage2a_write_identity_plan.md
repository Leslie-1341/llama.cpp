# Stage 2A — write path identity-consumed mapping 实现计划

前置：

- [docs/kv_paged_read_stage0_source_read.md](kv_paged_read_stage0_source_read.md)
- [docs/kv_paged_read_stage1_metadata_plan.md](kv_paged_read_stage1_metadata_plan.md)
- [docs/kv_paged_read_stage1_metadata_results.md](kv_paged_read_stage1_metadata_results.md)

Stage 1 已建好 metadata scaffold（`LLAMA_KV_PAGED`、identity `block_table`、physical block metadata、free-list、identity 自证），且验证 baseline / `=0` / `=1` 三者逐 token 一致、`identity_fail=0`、graph reuse 未破坏、与 lazy-tail 正交。Stage 1 的 `paged_resolve` **只在 assert 中被调用，未被任何真实寻址消费**。

Stage 2A 让写路径**第一次真正消费** `block_table`：写入 idxs 前先经 `paged_resolve(cell)` 得到 `phys_cell`。因 `block_table` 仍是 identity，`phys_cell == cell`，真实写入位置不变，输出必须与 baseline 完全一致。

---

## 1. Stage 2A 目标

1. 让写路径的 runtime-input idxs（`k_idxs` / `v_idxs`）的值**经 `paged_resolve(cell)` 解析后再落入张量**，而非直接用 `sinfo.idxs[s][i]`；
2. `block_table` 保持 identity，故解析结果恒等于原 cell index → **零行为变化**；
3. 新增写路径解析统计与自证（`write_resolve_checks/fail/changed`），证明"链路已接通且仍恒等"；
4. `LLAMA_KV_PAGED=0` 字节级等价当前分支；`LLAMA_KV_PAGED=1` 输出与 baseline 逐 token 一致、sha256 相同；
5. 把"`block_table` 被写路径实际寻址消费"这条链路在 identity 下打通并回归，为 Stage 2B（非连续块式分配）与 Stage 3（staging paged-read）铺接口。

---

## 2. 非目标（硬约束）

| # | 不做 | 理由 |
|---|---|---|
| 1 | 不改 `get_k/get_v` | 读路径继续冻结，留给 Stage 3 |
| 2 | 不改 attention kernel | 复用现有 FA |
| 3 | 不改 graph / context | 不碰 build_attn / reserve / reuse |
| 4 | 不改 ggml | 不新增 op，不动 `ggml_set_rows` |
| 5 | 不做非 identity block allocation | `block_table` 维持 `[i]=i`，Stage 2B 才动 |
| 6 | 不做 staging paged-read | Stage 3 |
| 7 | 不做 release / madvise | Stage 4，且不与 lazy madvise 抢占 |
| 8 | 不做 swap | Stage 4 |
| 9 | 不做 prefetch | Stage 5 |
| 10 | 不改 `sinfo` 本身 | 只读 `sinfo.idxs`，解析结果写入张量数据，绝不回写 `sinfo` |
| 11 | `LLAMA_KV_PAGED=0` 行为完全不变 | 零回归底线，挂载点早退 |
| 12 | `LLAMA_KV_PAGED=1` 解析结果仍等于原 cell | identity 不变，写入位置不变 |

**关键边界**：Stage 2A 只把"取 idxs 值"这一步从"直接读 `sinfo.idxs`"换成"`paged_resolve(sinfo.idxs)`"，identity 下两者逐位相等。不触碰 `find_slot`、不触碰 `sinfo` 生成、不改 `cpy_k/cpy_v` 的 `ggml_set_rows` 结构。

---

## 3. 当前写路径定位（已读源码确认）

写路径从 cell 选择到 runtime input 张量的链路：

1. **`find_slot()`**（[llama-kv-cache.cpp:1358](../src/llama-kv-cache.cpp#L1358) 起）填 `res.idxs[s]`，即本 ubatch 每个 token 的 cell 行号；
2. **`build_input_k_idxs/v_idxs`**（[:2481](../src/llama-kv-cache.cpp#L2481) / [:2491](../src/llama-kv-cache.cpp#L2491)）构造 I64 1-D runtime input 张量（`ggml_set_input`），**只定形状，不填值**；
3. **`set_input_k_idxs`**（[:2548](../src/llama-kv-cache.cpp#L2548)）/ **`set_input_v_idxs`**（[:2564](../src/llama-kv-cache.cpp#L2564)）在每步 set_inputs 时把值写进张量 host buffer：

   ```cpp
   // set_input_k_idxs（:2559）
   data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
   // set_input_v_idxs 非转置（:2576）同式；v_trans 分支（:2590）按 head 展开
   ```

   `offs = sinfo.strm[s]*get_size()` 是 stream 基址偏移。**`sinfo.idxs[s][i]` 就是要被 `paged_resolve` 包住的 cell index。**
4. **`cpy_k/cpy_v`**（[:2390](../src/llama-kv-cache.cpp#L2390) / [:2425](../src/llama-kv-cache.cpp#L2425)）用上面填好的 idxs 张量做 `ggml_set_rows(k, k_cur, k_idxs)`，把每 token 散写到对应行。
5. `llama_kv_cache_context::set_input_k_idxs`（[:3720](../src/llama-kv-cache.cpp#L3720)）只是转发到 `kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur])`。

**结论**：唯一需要改的精确点是 **`set_input_k_idxs` / `set_input_v_idxs` 里把 `sinfo.idxs[s][i]` 取出、加 `offs` 写入 `data[]` 的那几行**。这是 cell index → 张量数值的唯一出口，且只读 `sinfo`、不改 `sinfo`、不改张量形状、不动 graph。`cpy_k/cpy_v` 与 `build_input_*` 完全不需要改。

---

## 4. 计划改动点

### 4.1 在 `set_input_k_idxs` / `set_input_v_idxs` 内消费 `paged_resolve`

把"直接用 `sinfo.idxs[s][i]`"替换为"先 `paged_resolve(cell)`，再加 `offs` 写入"。

- **`set_input_k_idxs`（[:2559](../src/llama-kv-cache.cpp#L2559)）**：

  ```cpp
  const uint32_t cell = sinfo.idxs[s][i];
  const uint32_t phys = paged_write_resolve(cell);   // 关闭时直接返回 cell
  data[s*sinfo.size() + i] = offs + phys;
  ```

- **`set_input_v_idxs` 非转置分支（[:2576](../src/llama-kv-cache.cpp#L2576)）**：同样把 `sinfo.idxs[s][i]` 换成 `paged_write_resolve(cell)`。
- **`set_input_v_idxs` 转置分支（[:2590](../src/llama-kv-cache.cpp#L2590)）**：`offs + j*kv_size + phys`，仅替换最内层的 cell 项；`j*kv_size`（head 内偏移）不变。

> v_trans 在 `-fa on` 单序列目标场景下不触发（Stage 1 支持性检查仅 `n_stream==1 && !v_trans` 才启用），但为完整与防御，转置分支同样按 `phys` 改，identity 下逐位相等。

### 4.2 不新增临时 resolved idxs buffer

**不需要**额外 buffer。解析是逐元素的纯函数替换：在写 `data[...]` 的那一行内联调用 `paged_write_resolve(cell)` 即可，无需先 gather 到中间 vector。这样改动最小、不引入新内存、不改循环结构。

### 4.3 不修改 `sinfo`

`set_input_*` 全程把 `sinfo` 当 `const &` 读，只读 `sinfo.idxs[s][i]`、`sinfo.strm[s]`、`sinfo.size()`、`sinfo.n_stream()`。解析结果只写进 `dst->data`（runtime input 张量），**绝不回写 `sinfo.idxs`**。`find_slot` / `apply_ubatch` 生成的 `sinfo` 不动。

### 4.4 新增 `paged_write_resolve` 包装（统计 + 自证一体）

`paged_resolve` 是 `const`，无法累加统计。新增一个非 const 小包装专供写路径，内联做"解析 + 计数 + identity 自证"，避免污染 `paged_resolve` 与 `paged_assert_identity`：

```cpp
uint32_t llama_kv_cache::paged_write_resolve(uint32_t cell) {
    if (!kv_paged_enabled) {
        return cell;                       // 关闭：早退，零开销，字节级等价
    }
    const uint32_t phys = paged_resolve(cell);
    paged_write_resolve_checks += 1;
    if (phys == PAGED_BLOCK_INVALID) {
        paged_write_resolve_fail += 1;
        return cell;                       // 防御：解析失败回退原 cell，不破坏写入
    }
    if (phys != cell) {
        paged_write_resolve_changed += 1;  // identity 下应恒为 0
    }
    return phys;
}
```

> 注意：`set_input_k_idxs` 等当前签名为 `const`。引入非 const 的 `paged_write_resolve` 需把这三个 `set_input_*` 成员（及 `llama_kv_cache_context` 的转发）去掉 `const`，或把统计字段标记 `mutable`。**推荐 `mutable` 方案**：统计计数器加 `mutable`、`paged_write_resolve` 保持可在 const 上下文调用，**完全不动 `set_input_*` 的 const 签名与调用方**，改动面最小、最不易触发连锁修改。

---

## 5. 新增统计字段

挂在 `llama_kv_cache` 私有段，紧邻 Stage 1 的 `paged_identity_*`（[llama-kv-cache.h](../src/llama-kv-cache.h)）。采用 §4.4 推荐方案则标 `mutable`：

```cpp
mutable uint64_t paged_write_resolve_checks  = 0; // 写路径解析次数（应 > 0）
mutable uint64_t paged_write_resolve_fail    = 0; // 解析返回 INVALID 次数（应 = 0）
mutable uint64_t paged_write_resolve_changed = 0; // phys != cell 次数（identity 下应 = 0）
```

在 `paged_log_stats`（[:1654](../src/llama-kv-cache.cpp#L1654)）追加一行打印这三个值，走 `LLAMA_LOG_*`（stderr，不污染 stdout）。

---

## 6. correctness 验证命令

> 仅记录命令，Stage 2A 编码完成后执行；本计划阶段不跑、不 build。

### 6.1 默认关闭零回归

```bash
./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on --seed 42 --temp 0 \
    > /tmp/base.txt 2>/tmp/base.log
LLAMA_KV_PAGED=0 ./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on \
    --seed 42 --temp 0 > /tmp/off.txt 2>/tmp/off.log
diff /tmp/base.txt /tmp/off.txt                       # 必须为空
```

### 6.2 开启 identity-consumed 写路径输出一致

```bash
LLAMA_KV_PAGED=1 ./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on \
    --seed 42 --temp 0 > /tmp/on.txt 2>/tmp/on.log
diff /tmp/base.txt /tmp/on.txt                        # identity 下必须为空
sha256sum /tmp/base.txt /tmp/off.txt /tmp/on.txt      # 三者一致
grep -E 'paged|identity|write_resolve' /tmp/on.log    # 看统计
```

### 6.3 与 lazy-tail 正交

```bash
LLAMA_KV_PAGED=1 LLAMA_KV_LAZY_TAIL=1 ./build/bin/llama-cli ... --temp 0 \
    > /tmp/on_lazy.txt; diff /tmp/base.txt /tmp/on_lazy.txt   # 应为空
```

---

## 7. 通过标准（全部满足才算 Stage 2A 通过）

1. ✅ 6.1 `diff` 空（默认关闭字节级等价）；
2. ✅ 6.2 `diff` 空、三输出 sha256 一致（identity-consumed 写路径不改输出）；
3. ✅ `paged_write_resolve_checks > 0`（写路径**确实消费了** `block_table`，链路接通）；
4. ✅ `paged_write_resolve_fail == 0`（解析无 INVALID）；
5. ✅ `paged_write_resolve_changed == 0`（identity 下 `phys == cell`）；
6. ✅ `paged_identity_fail == 0`（Stage 1 自证仍成立）；
7. ✅ graph reuse 未被破坏（`graphs reused` 与 Stage 1 持平）；
8. ✅ 6.3 与 lazy-tail 同开仍 `diff` 空。

判定核心：**`write_resolve_checks > 0` 且 `changed == 0` 且 `diff` 空** —— 证明"消费了 block_table 但仍恒等"。

---

## 8. 风险表

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| S2A-R1 | 误改 `set_input_*` 偏移/循环结构致输出不一致 | 高 | 只替换最内层 cell 项，`offs`/`j*kv_size`/索引步进保持原样；先跑 6.2 逐 token diff |
| S2A-R2 | const 签名连锁修改面扩大 | 中 | 采用 `mutable` 统计字段方案，不动 `set_input_*` 与调用方 const 签名 |
| S2A-R3 | `paged_resolve` 返回 INVALID 致写错行 | 中 | 包装内 INVALID 回退原 cell + `fail++`，identity 下不触发；`fail==0` 守门 |
| S2A-R4 | v_trans 分支漏改或改错 | 中 | 同步按 `phys` 改，仅替换 cell 项；目标场景 `!v_trans` 不触发，但保持一致 |
| S2A-R5 | 统计计数进热路径影响性能 | 低 | 仅整型自增，关闭时早退；非判定项可后续条件编译 |
| S2A-R6 | 统计日志污染 stdout 影响 diff | 低 | 全走 `LLAMA_LOG_*`（stderr） |

**RSS 立场**：Stage 2A 仍是 identity，物理寻址与分配完全不变，**不期待也不声称 RSS 下降**。收益是"写路径消费 block_table"的链路接通，为 Stage 2B/3 铺路。

---

## 9. Codex 实现提示词草案（暂不改代码）

> **任务：在 `src/llama-kv-cache.{h,cpp}` 实现 Stage 2A write path identity-consumed mapping。让写入 idxs 经 `paged_resolve` 消费 block_table，但 identity 下解析结果恒等于原 cell，输出与 baseline 逐 token 一致。**
>
> **硬约束**：① 只改 `llama-kv-cache.h`/`llama-kv-cache.cpp`，禁止改 `get_k/get_v`、attention kernel、graph/context、ggml；② 不做非 identity block allocation、不做 staging read、不做 release/madvise/swap/prefetch；③ 不改 `sinfo` 本身（只读 `sinfo.idxs`）；④ `LLAMA_KV_PAGED=0` 字节级等价当前分支（包装早退）；⑤ `LLAMA_KV_PAGED=1` 解析结果仍等于原 cell。
>
> **实现**：
> 1. `llama-kv-cache.h` 私有段加 `mutable` 计数器 `paged_write_resolve_checks/fail/changed`，声明 `uint32_t paged_write_resolve(uint32_t cell) const;`（用 `mutable` 故可 const）。
> 2. `llama-kv-cache.cpp` 实现 `paged_write_resolve`：关闭早退返回 `cell`；否则 `paged_resolve(cell)`，`checks++`，INVALID 则 `fail++` 回退 `cell`，`phys!=cell` 则 `changed++`，返回 `phys`。
> 3. `set_input_k_idxs`（:2559）、`set_input_v_idxs` 非转置（:2576）与转置（:2590）三处：把 `sinfo.idxs[s][i]` 替换为 `paged_write_resolve(sinfo.idxs[s][i])`，其余偏移项不动。
> 4. `paged_log_stats`（:1654）追加打印三个 write_resolve 计数。
> 5. 所有日志走 `LLAMA_LOG_*`（stderr）。
>
> **验证**（编码后执行，见 §6）：6.1/6.2/6.3 `diff` 全空、三输出 sha256 一致；日志中 `write_resolve_checks>0`、`write_resolve_fail==0`、`write_resolve_changed==0`、`identity_fail==0`；graph reuse 持平。
>
> **对照（只读，勿照搬代码）**：`/root/oscomp/llama.cpp-paged-ref` 的 `calculate_global_slot_index`（`block_table[bid]*block_size + offset`）。
