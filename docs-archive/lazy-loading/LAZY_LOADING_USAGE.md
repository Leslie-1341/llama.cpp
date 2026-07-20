# Lazy Loading 使用指南

## 简介

Lazy Loading 是 llama.cpp 的一个实验性特性，允许大模型在有限内存的情况下运行。它通过按需加载模型层，只在内存中保留最近使用的层，从而大幅降低内存占用。

## 启用方式

通过环境变量启用：

```bash
export LLAMA_LAZY_LOADING=1           # 启用 lazy loading
export LLAMA_LAZY_MAX_LAYERS=8        # 最多同时映射 8 层
export LLAMA_LAZY_DEBUG=1             # 启用调试输出（可选）
```

## 参数说明

### LLAMA_LAZY_MAX_LAYERS

控制同时在内存中保留的层数。

**推荐值**：
- 8B 模型: 8-12 层
- 13B 模型: 6-10 层  
- 30B+ 模型: 4-8 层

**权衡**：
- 更大值 = 更少的映射/驱逐开销，但内存占用更高
- 更小值 = 更低的内存占用，但更频繁的映射/驱逐

### LLAMA_LAZY_DEBUG

启用详细的调试输出，包括：
- 层的映射和驱逐
- 文件索引和 offset 信息
- LRU 队列状态
- 统计信息

## 使用示例

### 基础使用

```bash
# 启用 lazy loading
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8

# 运行推理
./build/bin/llama-cli \
    -m models/llama-2-7b-chat.gguf \
    -p "Hello, how are you?" \
    -n 100
```

### 调试模式

```bash
# 启用调试输出
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8
export LLAMA_LAZY_DEBUG=1

./build/bin/llama-cli \
    -m models/model.gguf \
    -p "Write a poem" \
    -n 50 \
    2>&1 | tee lazy_debug.log
```

### 极致内存节省

```bash
# 只保留 4 层（适用于非常有限的内存）
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=4

./build/bin/llama-cli -m models/large-model.gguf -p "Hello"
```

## 工作原理

1. **初始化阶段**
   - 分析模型文件，构建层映射表
   - 记录每层的文件位置和大小
   - 不立即加载任何层

2. **推理阶段**
   - 当访问某一层时，检查是否已在内存中
   - 如果不在，使用 mmap 映射该层（按需加载）
   - 如果内存中的层数超过限制，驱逐最久未使用的层（LRU）

3. **LRU 驱逐策略**
   - 维护一个 LRU 队列
   - 每次访问层时，将该层移到队尾
   - 需要驱逐时，从队头取出最久未使用的层

## Split 模型支持

Lazy Loading 完全支持 split 模型（分割成多个文件的大模型）：

```bash
# 自动处理多文件模型
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8

./build/bin/llama-cli \
    -m models/large-model-00001-of-00003.gguf \
    -p "Hello"
```

系统会自动：
- 识别所有分割文件
- 为每个文件维护独立的文件描述符
- 正确映射跨文件的层

## 性能提示

### 1. 选择合适的 MAX_LAYERS

太小会导致频繁的映射/驱逐开销：
```bash
# 不推荐：太小
export LLAMA_LAZY_MAX_LAYERS=2  # ❌ 性能差
```

太大则无法节省内存：
```bash
# 如果能容纳所有层，lazy loading 无意义
export LLAMA_LAZY_MAX_LAYERS=100  # ⚠️ 不如直接用 mmap
```

### 2. 避免过度随机访问

Lazy loading 适合顺序推理，不适合：
- 频繁切换不同层的操作
- 随机访问模型权重

### 3. 监控统计信息

启用调试模式查看统计：
```
Map calls:       1234
  - Hits:        1000 (81.0%)
  - Misses:      234 (19.0%)
Evictions:       190
Avg map time:    45.2 us
Avg evict time:  12.3 us
Peak mapped:     512.3 MB
```

高命中率（>80%）表示 MAX_LAYERS 设置合理。

## 故障排除

### 问题：输出乱码

**原因**：旧版本的 bug（已修复）

**解决**：确保使用最新代码，包含 split 文件支持修复

### 问题：内存占用仍然很高

**可能原因**：
1. MAX_LAYERS 设置太大
2. 模型的全局 tensors（embedding, output）总是加载
3. 其他内存占用（KV cache, context buffer）

**解决**：
- 减小 MAX_LAYERS
- 减小 context size (`-c` 参数)

### 问题：性能很慢

**可能原因**：
1. MAX_LAYERS 太小，频繁映射/驱逐
2. 磁盘 I/O 瓶颈

**解决**：
- 增加 MAX_LAYERS
- 使用 SSD 而非 HDD
- 启用 prefetch（自动优化）

## 限制和注意事项

1. **仅支持推理** - 不支持训练或微调
2. **Linux/Unix 优先** - 使用 POSIX mmap，在 Windows 上可能需要额外适配
3. **额外开销** - 映射/驱逐有轻微性能开销
4. **调试功能** - 仍是实验性特性，可能有未知问题

## 与其他功能的兼容性

✅ **兼容**：
- 量化模型（Q4_0, Q8_0, 等）
- Split 模型
- CPU 推理
- 多线程推理

⚠️ **部分兼容**：
- GPU offload - 仅 CPU 层使用 lazy loading
- NUMA - 需要额外测试

❌ **不兼容**：
- mlock - 会禁用 lazy loading
- 某些后端的特殊内存布局要求

## 参考

- 源代码: `src/llama-lazy.h`, `src/llama-lazy.cpp`
- 修复文档: `LAZY_LOADING_FIX.md`
- 测试脚本: `test_lazy_loading_fix.sh`
