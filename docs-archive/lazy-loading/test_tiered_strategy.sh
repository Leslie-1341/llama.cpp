#!/bin/bash
# 测试分层内存策略

echo "=========================================="
echo "分层内存策略测试"
echo "=========================================="
echo ""

MODEL_PATH="${1:-model.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path>"
    exit 1
fi

echo "使用模型: $MODEL_PATH"
echo ""

TEST_PROMPT="Tell me a short story about a robot."
NUM_TOKENS=100

mkdir -p /tmp/tiered-test
cd /tmp/tiered-test

echo "=========================================="
echo "测试 1: Native模式（基准）"
echo "=========================================="
export LLAMA_LAZY_LOADING=0

echo "获取RSS基准..."
/root/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$TEST_PROMPT" \
    -n $NUM_TOKENS \
    --single-turn \
    2>&1 | tee native.log &

PID=$!
sleep 5

# 监控RSS
RSS_NATIVE=$(ps -o rss= -p $PID 2>/dev/null | awk '{print $1}')
echo "Native RSS: $RSS_NATIVE KB ($(echo "scale=2; $RSS_NATIVE/1024" | bc) MB)"

wait $PID
echo ""

echo "=========================================="
echo "测试 2: 分层策略 (max_layers=16)"
echo "=========================================="
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=16
export LLAMA_LAZY_DEBUG=1

echo "运行分层策略..."
/root/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$TEST_PROMPT" \
    -n $NUM_TOKENS \
    --single-turn \
    2>&1 | tee tiered.log &

PID=$!
sleep 5

# 监控RSS
RSS_TIERED=$(ps -o rss= -p $PID 2>/dev/null | awk '{print $1}')
echo "Tiered RSS: $RSS_TIERED KB ($(echo "scale=2; $RSS_TIERED/1024" | bc) MB)"

wait $PID
echo ""

echo "=========================================="
echo "结果分析"
echo "=========================================="

echo ""
echo "--- 内存使用对比 ---"
if [ -n "$RSS_NATIVE" ] && [ -n "$RSS_TIERED" ]; then
    REDUCTION=$(echo "scale=2; 100 * (1 - $RSS_TIERED / $RSS_NATIVE)" | bc)
    echo "Native:  $RSS_NATIVE KB"
    echo "Tiered:  $RSS_TIERED KB"
    echo "减少:    $REDUCTION%"
else
    echo "无法获取RSS数据"
fi

echo ""
echo "--- Tensor分类统计 ---"
grep "LAZY-CRITICAL\|LAZY-HOT\|LAZY-WARM\|LAZY-COLD" tiered.log | \
    awk '{print $1}' | sort | uniq -c

echo ""
echo "--- 内存操作统计 ---"
grep "PREFETCH\|RELEASE\|EVICT" tiered.log | wc -l | \
    awk '{print "Prefetch/Release/Evict操作: " $1 " 次"}'

echo ""
echo "--- madvise DONTNEED统计 ---"
grep "advised DONTNEED" tiered.log | head -5

echo ""
echo "--- 正确性检查 ---"
# 提取生成的文本
grep -v "^\[" tiered.log | grep -v "^llama" | grep -v "^build" | \
    tail -20 > tiered_text.txt

if [ -s tiered_text.txt ]; then
    # 检查是否包含合理的英文
    if grep -qiE "\b(the|and|is|was|were|robot|story)\b" tiered_text.txt; then
        echo "✅ 输出正常（包含合理英文文本）"
    else
        echo "⚠️  输出可疑，请人工检查"
        head -5 tiered_text.txt
    fi
else
    echo "❌ 无输出"
fi

echo ""
echo "=========================================="
echo "详细日志已保存到: /tmp/tiered-test/"
echo "=========================================="
