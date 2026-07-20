# Lazy Loading 完整修复方案

## 问题根因分析

经过深入调查，发现了**三个关键问题**：

### 🔴 问题1: 数据生命周期不匹配
**现象**: 
- `get_tensor_addr()` 返回mmap地址后，该层可能在推理过程中被LRU驱逐
- 被驱逐后，tensor仍持有已munmap的无效指针

**证据**:
```cpp
// llama-model-loader.cpp:1498-1502
if (cur->data == nullptr) {
    cur->data = (uint8_t *)addr;  // ❌ 直接指向mmap内存
} else {
    memcpy(cur->data, addr, ggml_nbytes(cur));  // ✅ 拷贝数据
}
```

**问题**: 
- 第一个分支直接使用mmap地址，但该地址可能在推理时被munmap
- 只有第二个分支（已分配buffer的情况）才会拷贝数据

### 🔴 问题2: 映射驱逐策略过于激进
**现象**:
- `max_mapped_layers=8` 对于32层模型来说太小
- 推理时需要快速访问多个层，频繁的map/unmap会导致竞态

**证据**:
```cpp
// llama-lazy.cpp:242
while (ctx.currently_mapped.size() >= ctx.params.max_mapped_layers) {
    if (!llama_lazy_loading_evict_lru(ctx)) {
        return false;
    }
}
```

### 🔴 问题3: 缺少数据完整性验证
**现象**:
- 没有验证mmap后的数据是否正确
- 没有与正常mmap模式的数据进行对比

## 修复方案

### 方案A: 强制数据拷贝（推荐）✅

**核心思想**: Lazy loading只在需要时映射数据，但**必须拷贝到tensor的buffer**中，然后立即释放映射。

**优点**:
- ✅ 解决生命周期问题
- ✅ 数据安全，不会被驱逐影响
- ✅ 与现有backend完全兼容
- ✅ 实现简单，风险低

**缺点**:
- ❌ 需要额外内存拷贝
- ❌ 首次加载时间略长

**实现步骤**:

#### 1. 修改 `llama-model-loader.cpp` 的集成代码

**当前代码**:
```cpp
void * addr = llama_lazy_loading_get_tensor_addr(...);
if (cur->data == nullptr) {
    cur->data = (uint8_t *)addr;  // ❌ 问题所在
} else {
    memcpy(cur->data, addr, ggml_nbytes(cur));
}
```

**修复后**:
```cpp
// 总是分配buffer并拷贝数据
if (cur->data == nullptr) {
    cur->data = (uint8_t *)malloc(ggml_nbytes(cur));
    if (cur->data == nullptr) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
}

void * addr = llama_lazy_loading_get_tensor_addr(...);
if (addr == nullptr) {
    throw std::runtime_error("lazy loading failed");
}

// 拷贝数据
memcpy(cur->data, addr, ggml_nbytes(cur));

// 立即释放映射（可选优化）
llama_lazy_loading_release_tensor(lazy_ctx, tensor_name);
```

#### 2. 添加立即释放机制

在 `llama-lazy.h` 添加:
```cpp
// 释放tensor的映射（可选，用于立即释放）
void llama_lazy_loading_release_tensor(
    llama_lazy_loading_context & ctx,
    const std::string & tensor_name
);
```

在 `llama-lazy.cpp` 实现:
```cpp
void llama_lazy_loading_release_tensor(
    llama_lazy_loading_context & ctx,
    const std::string & tensor_name
) {
    int layer_id = llama_lazy_loading_parse_layer_id(tensor_name);
    
    std::lock_guard<std::mutex> lock(ctx.mapping_mutex);
    
    // 如果这一层的所有tensor都已加载，可以立即unmap
    // 这是一个优化：减少内存占用
    // 注意：需要跟踪每个层的tensor加载状态
}
```

#### 3. 增加数据验证（调试模式）

```cpp
void* llama_lazy_loading_get_tensor_addr_with_verify(
    llama_lazy_loading_context & ctx,
    const std::string & tensor_name,
    size_t file_offset,
    uint16_t file_idx,
    size_t expected_size
) {
    void* addr = llama_lazy_loading_get_tensor_addr(ctx, tensor_name, file_offset, file_idx);
    
    if (addr && ctx.params.debug_log) {
        // 计算简单的checksum
        uint64_t checksum = 0;
        const uint64_t* data = (const uint64_t*)addr;
        size_t n_qwords = std::min(expected_size / 8, size_t(1024));  // 只检查前8KB
        for (size_t i = 0; i < n_qwords; i++) {
            checksum ^= data[i];
        }
        fprintf(stderr, "[LAZY-VERIFY] %s: checksum=0x%016lx\n", 
                tensor_name.c_str(), checksum);
    }
    
    return addr;
}
```

### 方案B: 智能缓存管理（高级）

**核心思想**: 保持映射直到确认不再需要，使用引用计数。

**优点**:
- ✅ 避免重复拷贝
- ✅ 内存效率高

**缺点**:
- ❌ 实现复杂
- ❌ 需要跟踪tensor生命周期
- ❌ 风险较高

**实现概要**:
```cpp
struct llama_layer_mapping {
    // ... 现有字段
    std::atomic<int> ref_count;  // 引用计数
    std::unordered_set<std::string> active_tensors;  // 正在使用的tensors
};

void* llama_lazy_loading_get_tensor_addr(/* ... */) {
    // ... 映射逻辑
    mapping.ref_count++;
    mapping.active_tensors.insert(tensor_name);
    return addr;
}

void llama_lazy_loading_release_tensor(/* ... */) {
    mapping.ref_count--;
    mapping.active_tensors.erase(tensor_name);
    
    if (mapping.ref_count == 0) {
        // 可以安全驱逐
    }
}
```

### 方案C: 混合策略（最优）

**核心思想**: 
- 全局tensors (embedding, output): 永久映射
- Layer tensors: 按需映射+拷贝，或根据配置保持映射

**实现**:
```cpp
void * addr = llama_lazy_loading_get_tensor_addr(...);

if (layer_id == -1) {
    // 全局tensor: 直接使用mmap地址（不驱逐）
    cur->data = (uint8_t *)addr;
} else {
    // Layer tensor: 拷贝数据
    if (cur->data == nullptr) {
        cur->data = (uint8_t *)malloc(ggml_nbytes(cur));
    }
    memcpy(cur->data, addr, ggml_nbytes(cur));
    llama_lazy_loading_release_tensor(lazy_ctx, tensor_name);
}
```

## 推荐实施路径

### 第一阶段：快速修复（1小时）
1. ✅ 修改 `llama-model-loader.cpp:1498` 强制拷贝数据
2. ✅ 增加 `LLAMA_LAZY_MAX_LAYERS` 默认值到32或更高
3. ✅ 添加数据验证日志

### 第二阶段：优化（2-3小时）
1. ✅ 实现 `release_tensor` 立即释放机制
2. ✅ 添加 reference counting
3. ✅ 区分全局tensors和layer tensors

### 第三阶段：完善（1天）
1. ✅ 添加单元测试
2. ✅ 性能基准测试
3. ✅ 文档更新

## 具体代码修改

### 文件1: `src/llama-model-loader.cpp`

**位置**: 第1498-1502行

**修改前**:
```cpp
if (cur->data == nullptr) {
    cur->data = (uint8_t *)addr;
} else {
    memcpy(cur->data, addr, ggml_nbytes(cur));
}
```

**修改后**:
```cpp
// Lazy loading: always copy data to tensor's buffer
// This ensures data remains valid even if the layer is evicted
size_t tensor_size = ggml_nbytes(cur);

if (cur->data == nullptr) {
    // Allocate buffer for tensor
    cur->data = (uint8_t *)malloc(tensor_size);
    if (cur->data == nullptr) {
        throw std::runtime_error(format(
            "failed to allocate %zu bytes for tensor '%s'",
            tensor_size, ggml_get_name(cur)
        ));
    }
}

// Copy data from lazy-loaded mapping
memcpy(cur->data, addr, tensor_size);

// Optional: release mapping immediately if all tensors in layer are loaded
// This reduces memory pressure but adds complexity
// TODO: Implement reference counting for proper lifecycle management
```

### 文件2: `src/llama-lazy.cpp`

**在 `get_tensor_addr` 函数末尾添加验证**:

**位置**: 第420行之前

```cpp
// 7. 返回实际地址
void* addr = static_cast<char*>(mapping.mapped_addr) + offset_in_mapping;

// **添加数据验证**
if (ctx.params.debug_log) {
    // 验证地址可读
    volatile uint8_t test = ((uint8_t*)addr)[0];
    
    // 计算简单checksum（前1KB）
    uint32_t checksum = 0;
    const uint8_t* data = (const uint8_t*)addr;
    size_t check_size = std::min(size_t(1024), 
                                  mapping.file_offset_end - file_offset);
    for (size_t i = 0; i < check_size; i++) {
        checksum = (checksum * 31) + data[i];
    }
    
    if (mapping.access_count <= 2) {
        fprintf(stderr, "[LAZY] Addr=%p, first_byte=0x%02x, checksum=0x%08x\n",
                addr, test, checksum);
    }
}

return addr;
```

### 文件3: `common/arg.cpp`

**修改默认值**:

**位置**: 搜索 `LLAMA_LAZY_MAX_LAYERS`

**修改前**:
```cpp
size_t max_mapped_layers = 16;  // 或其他小值
```

**修改后**:
```cpp
size_t max_mapped_layers = 64;  // 增加到足够容纳大多数模型
// 对于70B模型（80层），考虑使用100+
```

## 测试验证

### 测试1: 基础功能测试
```bash
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=64
export LLAMA_LAZY_DEBUG=1

./build/bin/llama-cli -m model.gguf \
    -p "Hello! Please respond with a short greeting." \
    -n 30
```

**期望结果**: 输出正常文本，无乱码

### 测试2: 对比正常模式
```bash
# 正常模式
export LLAMA_LAZY_LOADING=0
./build/bin/llama-cli -m model.gguf -p "Test prompt" -n 50 > normal.txt

# Lazy模式
export LLAMA_LAZY_LOADING=1
./build/bin/llama-cli -m model.gguf -p "Test prompt" -n 50 > lazy.txt

# 对比输出
diff normal.txt lazy.txt
```

**期望结果**: 输出完全一致

### 测试3: 内存占用测试
```bash
# 小的max_layers
export LLAMA_LAZY_MAX_LAYERS=4
./build/bin/llama-cli -m model.gguf -p "Test" -n 100 &
PID=$!
sleep 2
pmap $PID | tail -1  # 查看内存占用
```

### 测试4: 长文本生成
```bash
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=32

./build/bin/llama-cli -m model.gguf \
    -p "Write a detailed story about" \
    -n 500
```

**期望结果**: 连续生成500个token，无乱码，无卡顿

## 性能影响评估

### 内存使用
- **方案A（强制拷贝）**: 
  - 峰值内存 = 模型大小（与正常模式相同）
  - 过渡期内存 = 模型大小 + max_layers * layer_size

### 加载时间
- **首次推理**: 
  - 正常模式: ~2-5秒（全部mmap）
  - Lazy模式: ~2-5秒（按需加载，累计时间相似）

### 推理速度
- **影响**: 几乎无影响（数据拷贝只在加载时发生一次）
- **预期**: < 1% 性能差异

## 回退方案

如果修复后仍有问题：

### 选项1: 禁用Lazy Loading
```bash
export LLAMA_LAZY_LOADING=0
```

### 选项2: 映射所有层（无驱逐）
```bash
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=1000  # 远大于层数
```

### 选项3: 仅用于超大模型
```bash
# 仅当模型 > 某个阈值时启用
if [ $(stat -f%z model.gguf) -gt 34359738368 ]; then  # 32GB
    export LLAMA_LAZY_LOADING=1
fi
```

## 长期改进方向

### 1. 智能预取
- 基于推理模式预测下一层
- Transformer通常顺序访问层

### 2. 压缩存储
- 保持mmap状态，使用madvise控制物理内存

### 3. 分层策略
- 前几层：永久映射（访问频繁）
- 中间层：LRU管理
- 后几层：永久映射（输出层）

### 4. 多线程优化
- 异步预映射下一层
- 后台驱逐线程

## 总结

**核心修复**:
1. ✅ 强制数据拷贝到tensor buffer（解决生命周期问题）
2. ✅ 增加max_mapped_layers默认值（减少驱逐频率）
3. ✅ 添加数据验证（确保正确性）

**预期效果**:
- ✅ 生成正常文本，无乱码
- ✅ 内存使用合理
- ✅ 性能影响最小

**实施优先级**:
1. **P0**: 修改 llama-model-loader.cpp（强制拷贝）
2. **P1**: 增加 max_layers 默认值
3. **P2**: 添加验证和日志
4. **P3**: 实现引用计数和智能释放

---
日期: 2026-06-19
状态: 待实施
