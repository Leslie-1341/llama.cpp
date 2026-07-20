# Lazy Loading 乱码问题修复报告

## 问题描述

Lazy Loading 功能在生成文本时输出乱码字符。调试日志显示层映射机制工作正常（正确的加载/驱逐），但生成的文本是无意义字符。

## 根本原因分析

经过深入分析，发现问题出在 **split 文件支持** 上：

### 原有代码的问题

1. **只使用第一个文件的文件描述符**
   ```cpp
   // llama-model-loader.cpp:1367
   int fd = files[0]->file_id();  // ❌ 只使用 files[0]
   ```

2. **Layer mapping 不包含文件索引**
   ```cpp
   // llama-lazy.h - 原始结构
   struct llama_layer_mapping {
       int layer_id;
       // ❌ 缺少 file_idx
       size_t file_offset_start;
       // ...
   };
   ```

3. **构建 layer map 时忽略文件索引**
   ```cpp
   // llama-model-loader.cpp:1386-1389
   for (const auto & [name, weight] : weights_map) {
       tensor_offsets[name] = weight.offs;
       // ❌ 没有传递 weight.idx (文件索引)
   }
   ```

### 为什么会导致乱码

对于 split 模型（例如分成多个 .gguf 文件）：

- **File 0**: 包含 embedding 和前几层，offset 相对于 file 0
- **File 1**: 包含中间层，offset 相对于 file 1
- **File 2**: 包含后面的层和 output，offset 相对于 file 2

当尝试加载来自 File 1 的 tensor 时：
```
1. tensor.offs = 1234567 (相对于 File 1)
2. 使用 files[0]->file_id() 作为 fd
3. mmap(fd_of_file0, ..., offset=1234567)
4. ❌ 读取到 File 0 在 offset 1234567 处的数据
5. 结果：完全错误的数据 → 乱码输出
```

## 修复方案

### 1. 修改数据结构

**llama-lazy.h**:
- 添加 `file_idx` 到 `llama_layer_mapping` 结构
- 将 `file_fd` 改为 `file_fds` 向量

```cpp
struct llama_layer_mapping {
    int layer_id;
    uint16_t file_idx;  // ✓ 新增：文件索引
    size_t file_offset_start;
    // ...
};

struct llama_lazy_loading_context {
    std::vector<int> file_fds;  // ✓ 改为向量，支持多文件
    // ...
};
```

### 2. 更新函数签名

```cpp
// 初始化时接收所有文件的 fd
bool llama_lazy_loading_init(
    llama_lazy_loading_context & ctx,
    const llama_lazy_loading_params & params,
    int n_layers,
    const std::vector<int> & file_fds  // ✓ 改为向量
);

// 构建 layer map 时接收文件索引信息
bool llama_lazy_loading_build_layer_map(
    llama_lazy_loading_context & ctx,
    const std::unordered_map<std::string, size_t> & tensor_offsets,
    const std::unordered_map<std::string, size_t> & tensor_sizes,
    const std::unordered_map<std::string, uint16_t> & tensor_file_indices  // ✓ 新增
);

// 获取 tensor 地址时指定文件索引
void* llama_lazy_loading_get_tensor_addr(
    llama_lazy_loading_context & ctx,
    const std::string & tensor_name,
    size_t file_offset,
    uint16_t file_idx  // ✓ 新增
);
```

### 3. 修改实现逻辑

**llama-lazy.cpp**:

1. 按 `(file_idx, layer_id)` 分组 tensors
   ```cpp
   std::map<std::pair<uint16_t, int>, std::vector<std::string>> layer_tensors;
   for (const auto & [name, offset] : tensor_offsets) {
       int layer_id = llama_lazy_loading_parse_layer_id(name);
       uint16_t file_idx = tensor_file_indices.at(name);
       layer_tensors[{file_idx, layer_id}].push_back(name);
   }
   ```

2. 映射层时使用正确的文件描述符
   ```cpp
   int file_fd = ctx.file_fds[mapping.file_idx];
   mapping.mapped_addr = mmap(nullptr, mapping.aligned_size, 
                               PROT_READ, MAP_SHARED, 
                               file_fd, mapping.file_offset_start);
   ```

3. 查找 mapping 时匹配文件索引
   ```cpp
   auto it = std::find_if(ctx.layer_mappings.begin(), ctx.layer_mappings.end(),
                          [layer_id, file_idx](const auto & m) {
                              return m.layer_id == layer_id && 
                                     m.file_idx == file_idx;
                          });
   ```

**llama-model-loader.cpp**:

1. 收集所有文件的 fd
   ```cpp
   std::vector<int> file_fds;
   for (const auto & file : files) {
       file_fds.push_back(file->file_id());
   }
   ```

2. 传递文件索引信息
   ```cpp
   std::unordered_map<std::string, uint16_t> tensor_file_indices;
   for (const auto & [name, weight] : weights_map) {
       tensor_file_indices[name] = weight.idx;
   }
   ```

3. 调用时提供文件索引
   ```cpp
   void * addr = llama_lazy_loading_get_tensor_addr(
       lazy_ctx, tensor_name, w.offs, w.idx);
   ```

## 修复的关键点

✅ **支持多文件 split 模型** - 每个文件使用独立的文件描述符  
✅ **正确的文件索引跟踪** - Layer mapping 包含 file_idx  
✅ **按文件分组映射** - 同一层在不同文件中的数据分别映射  
✅ **准确的地址计算** - 使用正确的 fd 和对应的 offset  
✅ **完整的向后兼容** - 单文件模型仍然正常工作  

## 测试验证

运行测试脚本：
```bash
./test_lazy_loading_fix.sh
```

或手动测试：
```bash
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8
export LLAMA_LAZY_DEBUG=1
./build/bin/llama-cli -m <model> -p "Hello" -n 50
```

### 预期结果

- ✓ 输出正常的文本，没有乱码字符
- ✓ 调试日志显示正确的 layer 映射和驱逐
- ✓ 对于 split 模型，显示正确的文件索引

## 性能影响

此修复对性能的影响：

- **内存占用**: 无变化（每个 layer-file 组合独立映射）
- **映射开销**: 轻微增加（需要跟踪更多 mapping 条目）
- **查找开销**: 可忽略（使用 `std::find_if` 时需要匹配两个字段）

总体来说，性能影响极小，而正确性得到了保证。

## 文件变更清单

修改的文件：
- `src/llama-lazy.h` - 数据结构和函数签名
- `src/llama-lazy.cpp` - 实现逻辑
- `src/llama-model-loader.cpp` - 调用代码

新增的文件：
- `test_lazy_loading_fix.sh` - 测试脚本
- `LAZY_LOADING_FIX.md` - 本文档

## 相关 Issue

此修复解决的问题：
- Lazy Loading 生成乱码输出
- Split 模型不支持
- 内存映射地址错误

## 未来改进建议

1. **更智能的预取策略** - 根据访问模式预取相邻层
2. **更细粒度的映射** - 按 tensor 而非 layer 映射（适用于极大模型）
3. **压缩感知的映射** - 对于量化模型优化映射策略
4. **跨文件的 LRU** - 更公平的驱逐策略

## 致谢

感谢报告此问题的用户，以及 llama.cpp 社区的支持。
