# Lazy Loading 实施状态 - 发现内置VM系统

## 日期
2026-06-20

## 重要发现 ⚠️

llama.cpp **已经内置了完整的虚拟内存（VM）管理系统**，支持：
- `vm_dontneed` - madvise DONTNEED
- `vm_sliding_unmap` - 滑动窗口unmap
- `vm_window_layers` - 窗口大小（层数）
- `vm_window_steps` - 窗口步长
- `vm_prefetch_budget_mb` - 预取预算
- `vm_reclaim_budget_mb` - 回收预算

这个系统在 `llama_model_default_params()` 中可配置。

## 当前状态

### ✅ 已完成
1. **第1步：安全基线** - 所有层强制拷贝，确保100%安全
2. **分类系统** - 在 `build_layer_map` 时正确分类层
3. **数据结构** - TensorClass, LayerMemoryConfig等

### 🔴 问题
1. **与内置VM系统冲突** - 我们的lazy loading可能与VM系统竞争
2. **集成点未找到** - 层处理在各个模型的 `build_graph` 中，难以统一钩入
3. **架构复杂** - llama.cpp使用图构建方式，非顺序层循环

## 决策点：下一步方向

### 选项A：停止Lazy Loading，使用内置VM系统 ✅ **推荐**

**理由**：
- llama.cpp已有成熟的VM系统
- 避免重复造轮子和冲突
- VM系统已被测试和优化
- 减少维护负担

**行动**：
1. 测试内置VM系统效果
2. 调优VM参数达到目标
3. 移除我们的lazy loading代码

**测试命令**：
```bash
# 启用VM系统
./build/bin/llama-cli -m model.gguf \
    --vm-dontneed \
    --vm-sliding-unmap \
    --vm-window-layers 8 \
    -p "prompt" -n 100
```

### 选项B：继续Lazy Loading（不推荐）

**问题**：
- 需要深度集成到模型架构
- 与VM系统冲突
- 难以维护
- 可能破坏现有功能

## 推荐方案：测试并文档化内置VM系统

### 第1步：验证内置VM效果

```bash
#!/bin/bash
# test_builtin_vm.sh

MODEL="model.gguf"
PROMPT="Tell me a story"

echo "=== 测试1：无VM ==="
./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n 100 &
PID=$!
sleep 3
RSS_BASELINE=$(ps -o rss= -p $PID | awk '{print $1}')
echo "Baseline RSS: $RSS_BASELINE KB"
wait $PID

echo ""
echo "=== 测试2：VM + DONTNEED ==="
./build/bin/llama-cli -m "$MODEL" \
    --vm-dontneed \
    -p "$PROMPT" -n 100 &
PID=$!
sleep 3
RSS_DONTNEED=$(ps -o rss= -p $PID | awk '{print $1}')
echo "DONTNEED RSS: $RSS_DONTNEED KB"
wait $PID

echo ""
echo "=== 测试3：VM + Sliding Unmap ==="
./build/bin/llama-cli -m "$MODEL" \
    --vm-dontneed \
    --vm-sliding-unmap \
    --vm-window-layers 8 \
    -p "$PROMPT" -n 100 &
PID=$!
sleep 3
RSS_SLIDING=$(ps -o rss= -p $PID | awk '{print $1}')
echo "Sliding RSS: $RSS_SLIDING KB"
wait $PID

echo ""
echo "=== 结果对比 ==="
echo "Baseline:  $RSS_BASELINE KB (100%)"
REDUCTION_DONTNEED=$(echo "scale=2; 100 * (1 - $RSS_DONTNEED / $RSS_BASELINE)" | bc)
REDUCTION_SLIDING=$(echo "scale=2; 100 * (1 - $RSS_SLIDING / $RSS_BASELINE)" | bc)
echo "DONTNEED:  $RSS_DONTNEED KB (-$REDUCTION_DONTNEED%)"
echo "Sliding:   $RSS_SLIDING KB (-$REDUCTION_SLIDING%)"
```

### 第2步：文档化VM参数

创建 `VM_USAGE_GUIDE.md`：

```markdown
# llama.cpp 内置VM系统使用指南

## 基础用法

### 降低内存占用

\`\`\`bash
./llama-cli -m model.gguf \\
    --vm-dontneed \\              # 启用madvise DONTNEED
    --vm-sliding-unmap \\         # 启用滑动窗口unmap
    --vm-window-layers 8 \\       # 窗口大小8层
    -p "prompt"
\`\`\`

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vm-dontneed` | false | 使用后释放物理页面 |
| `--vm-sliding-unmap` | false | 滑动窗口unmap |
| `--vm-window-layers` | 2 | 窗口大小（层数） |
| `--vm-window-steps` | 2 | 窗口步长 |
| `--vm-prefetch-budget-mb` | 512 | 预取内存预算 |

### 推荐配置

#### 低内存优先
\`\`\`bash
--vm-dontneed \\
--vm-sliding-unmap \\
--vm-window-layers 4 \\
--vm-reclaim-budget-mb 128
\`\`\`

#### 平衡模式
\`\`\`bash
--vm-dontneed \\
--vm-window-layers 8 \\
--vm-prefetch-budget-mb 512
\`\`\`

#### 性能优先
\`\`\`bash
--vm-prefetch-budget-mb 1024 \\
--vm-window-layers 16
\`\`\`
```

## 当前Lazy Loading代码的处理

### 保留的部分
- ✅ 分类系统（可能未来有用）
- ✅ 统计和监控基础设施

### 移除的部分
- ❌ 直接mmap使用（安全隐患）
- ❌ 自定义LRU驱逐（与VM冲突）
- ❌ 预取/释放函数（VM已实现）

### 最终状态
- 所有层拷贝到独立buffer（安全基线）
- 分类信息保留但不影响行为
- 用户应使用内置VM系统降低内存

## 结论

**建议**：
1. ✅ 保持当前安全基线（所有层拷贝）
2. ✅ 测试并文档化内置VM系统
3. ✅ 为用户提供VM配置指南
4. ❌ 不继续开发自定义lazy loading

**理由**：
- 内置VM系统功能完整
- 避免重复工作和冲突
- 降低维护成本
- 用户可通过VM参数灵活配置

---
日期: 2026-06-20
状态: 建议停止Lazy Loading，转向VM系统
