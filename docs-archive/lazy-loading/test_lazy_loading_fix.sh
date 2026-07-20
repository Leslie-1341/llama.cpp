#!/bin/bash
# Lazy Loading 修复验证测试脚本

set -e

echo "=========================================="
echo "Lazy Loading 修复验证测试"
echo "=========================================="
echo ""

# 检查模型文件
MODEL_PATH="${1:-/root/llama.cpp/model.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path>"
    exit 1
fi

echo "使用模型: $MODEL_PATH"
echo ""

# 测试提示词
TEST_PROMPT="Hello! Please respond with a short greeting."
NUM_TOKENS=30

# 创建输出目录
mkdir -p /tmp/lazy_test
cd /tmp/lazy_test

echo "=========================================="
echo "测试 1: 正常模式（无 Lazy Loading）"
echo "=========================================="
export LLAMA_LAZY_LOADING=0
unset LLAMA_LAZY_MAX_LAYERS
unset LLAMA_LAZY_DEBUG

echo "运行命令..."
/root/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$TEST_PROMPT" \
    -n $NUM_TOKENS \
    --no-display-prompt \
    2>&1 | tee normal_output.txt

echo ""
echo "提取生成的文本..."
grep -v "^\[" normal_output.txt | grep -v "^llama" | grep -v "^build:" | grep -v "^system" | tail -5 > normal_text.txt
echo "正常模式输出:"
cat normal_text.txt
echo ""

echo "=========================================="
echo "测试 2: Lazy Loading 模式（使用默认配置）"
echo "=========================================="
export LLAMA_LAZY_LOADING=1
unset LLAMA_LAZY_MAX_LAYERS
export LLAMA_LAZY_DEBUG=1

echo "运行命令..."
/root/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$TEST_PROMPT" \
    -n $NUM_TOKENS \
    --no-display-prompt \
    2>&1 | tee lazy_default_output.txt

echo ""
echo "提取生成的文本..."
grep -v "^\[" lazy_default_output.txt | grep -v "^llama" | grep -v "^build:" | grep -v "^system" | tail -5 > lazy_default_text.txt
echo "Lazy Loading (默认) 输出:"
cat lazy_default_text.txt
echo ""

echo "=========================================="
echo "测试 3: Lazy Loading 模式（max_layers=8）"
echo "=========================================="
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8
export LLAMA_LAZY_DEBUG=1

echo "运行命令..."
/root/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$TEST_PROMPT" \
    -n $NUM_TOKENS \
    --no-display-prompt \
    2>&1 | tee lazy_8_output.txt

echo ""
echo "提取生成的文本..."
grep -v "^\[" lazy_8_output.txt | grep -v "^llama" | grep -v "^build:" | grep -v "^system" | tail -5 > lazy_8_text.txt
echo "Lazy Loading (max_layers=8) 输出:"
cat lazy_8_text.txt
echo ""

echo "=========================================="
echo "结果对比"
echo "=========================================="

check_garbage() {
    local file=$1
    local mode=$2

    if grep -q "illoorečetotchayan" "$file" 2>/dev/null; then
        echo "❌ $mode: 检测到已知乱码模式"
        return 1
    fi

    if grep -qE "\b(Hello|Hi|Good|Nice|meet|you)\b" "$file" 2>/dev/null; then
        echo "✅ $mode: 输出包含合理的英文文本"
        return 0
    else
        echo "⚠️  $mode: 未检测到预期的问候语"
        return 1
    fi
}

echo ""
echo "正常模式:"
check_garbage normal_text.txt "正常模式"
NORMAL_OK=$?

echo ""
echo "Lazy Loading (默认 max_layers=64):"
check_garbage lazy_default_text.txt "Lazy (默认)"
LAZY_DEFAULT_OK=$?

echo ""
echo "Lazy Loading (max_layers=8):"
check_garbage lazy_8_text.txt "Lazy (max=8)"
LAZY_8_OK=$?

echo ""
echo "=========================================="
echo "统计信息"
echo "=========================================="
echo ""
echo "--- Lazy Loading (默认) 统计 ---"
grep -E "\[LAZY\]|\[INFO\] Lazy" lazy_default_output.txt | head -15

echo ""
echo "=========================================="
echo "最终结论"
echo "=========================================="
echo ""

if [ $NORMAL_OK -eq 0 ] && [ $LAZY_DEFAULT_OK -eq 0 ] && [ $LAZY_8_OK -eq 0 ]; then
    echo "🎉 所有测试通过！Lazy Loading 修复成功！"
    echo ""
    echo "✅ 正常模式: 工作正常"
    echo "✅ Lazy Loading (默认): 工作正常"
    echo "✅ Lazy Loading (max=8): 工作正常"
    exit 0
else
    echo "❌ 部分测试失败"
    echo ""
    [ $NORMAL_OK -ne 0 ] && echo "❌ 正常模式: 失败"
    [ $LAZY_DEFAULT_OK -ne 0 ] && echo "❌ Lazy Loading (默认): 失败"
    [ $LAZY_8_OK -ne 0 ] && echo "❌ Lazy Loading (max=8): 失败"
    echo ""
    echo "测试输出已保存到: /tmp/lazy_test/"
    exit 1
fi
