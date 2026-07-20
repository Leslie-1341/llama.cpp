#!/bin/bash
# Lazy Loading 修复验证测试 v2

echo "=========================================="
echo "Lazy Loading 修复验证测试 v2"
echo "=========================================="
echo ""

# 检查模型文件
MODEL_PATH="${1:-model.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path>"
    exit 1
fi

echo "使用模型: $MODEL_PATH"
echo ""

TEST_PROMPT="Hello! Please respond with a short greeting."
NUM_TOKENS=20

mkdir -p /tmp/lazy-test-v2
cd /tmp/lazy-test-v2

run_test() {
    local mode=$1
    local output_file=$2
    
    echo "运行测试: $mode"
    timeout 30s /root/llama.cpp/build/bin/llama-cli \
        -m "$MODEL_PATH" \
        -p "$TEST_PROMPT" \
        -n $NUM_TOKENS \
        --no-display-prompt \
        --single-turn \
        2>&1 | tee "$output_file" || true
    
    # 提取实际生成的文本（排除日志）
    grep -v "^\[" "$output_file" | \
        grep -v "^llama" | \
        grep -v "^build:" | \
        grep -v "^system" | \
        grep -v "^sampling" | \
        grep -v "^generate:" | \
        tail -10 > "${output_file}.text"
    
    echo "提取的文本内容:"
    cat "${output_file}.text"
    echo ""
}

echo "=========================================="
echo "测试 1: 正常模式"
echo "=========================================="
export LLAMA_LAZY_LOADING=0
unset LLAMA_LAZY_MAX_LAYERS
unset LLAMA_LAZY_DEBUG
run_test "normal" "normal.log"

echo "=========================================="
echo "测试 2: Lazy Loading (默认 max=64)"
echo "=========================================="
export LLAMA_LAZY_LOADING=1
unset LLAMA_LAZY_MAX_LAYERS
export LLAMA_LAZY_DEBUG=1
run_test "lazy-default" "lazy_default.log"

echo "=========================================="
echo "测试 3: Lazy Loading (max=8, 压力测试)"
echo "=========================================="
export LLAMA_LAZY_LOADING=1
export LLAMA_LAZY_MAX_LAYERS=8
export LLAMA_LAZY_DEBUG=1
run_test "lazy-8" "lazy_8.log"

echo "=========================================="
echo "结果分析"
echo "=========================================="

check_output() {
    local file=$1
    local mode=$2
    
    # 检查是否为空
    if [ ! -s "$file" ]; then
        echo "❌ $mode: 无输出"
        return 1
    fi
    
    # 检查已知乱码模式
    if grep -qE "illoorečet|otchayan|eggies" "$file" 2>/dev/null; then
        echo "❌ $mode: 检测到乱码模式"
        cat "$file" | head -3
        return 1
    fi
    
    # 检查是否包含大量非ASCII字符
    local non_ascii=$(grep -o '[^\x00-\x7F]' "$file" | wc -l)
    local total_chars=$(wc -c < "$file")
    if [ $total_chars -gt 10 ] && [ $non_ascii -gt $((total_chars / 3)) ]; then
        echo "❌ $mode: 包含过多非ASCII字符 ($non_ascii/$total_chars)"
        return 1
    fi
    
    # 检查是否包含合理的英文单词（但排除prompt中的词）
    # 寻找回复中的常见词
    if grep -qiE "\b(nice|good|great|well|thank|pleasure|glad)\b" "$file" 2>/dev/null; then
        echo "✅ $mode: 输出正常（包含有意义的英文回复）"
        return 0
    elif grep -qiE "\b(meet|you|too|here)\b" "$file" 2>/dev/null; then
        echo "✅ $mode: 输出正常（包含合理回复）"
        return 0
    else
        echo "⚠️  $mode: 输出存在但未检测到明确的回复词汇"
        echo "内容预览:"
        head -3 "$file"
        return 2
    fi
}

echo ""
check_output "normal.log.text" "正常模式"
NORMAL=$?

echo ""
check_output "lazy_default.log.text" "Lazy默认"
LAZY_DEFAULT=$?

echo ""
check_output "lazy_8.log.text" "Lazy max=8"
LAZY_8=$?

echo ""
echo "=========================================="
echo "Lazy Loading 统计信息"
echo "=========================================="
echo "--- 默认配置 (max=64) ---"
grep -E "\[LAZY-LOAD\]|\[INFO\] Lazy|Mapped layer" lazy_default.log | head -20 || echo "(无统计信息)"

echo ""
echo "--- 压力配置 (max=8) ---"
grep -E "\[LAZY-LOAD\]|\[INFO\] Lazy|Mapped layer|Evicted" lazy_8.log | head -20 || echo "(无统计信息)"

echo ""
echo "=========================================="
echo "最终结论"
echo "=========================================="

if [ $NORMAL -eq 0 ] && [ $LAZY_DEFAULT -eq 0 ]; then
    echo "✅ 核心测试通过！Lazy Loading 修复成功"
    [ $LAZY_8 -eq 0 ] && echo "✅ 压力测试也通过"
    exit 0
else
    echo "❌ 测试失败"
    [ $NORMAL -ne 0 ] && echo "  - 正常模式问题（可能是模型问题）"
    [ $LAZY_DEFAULT -ne 0 ] && echo "  - Lazy Loading 默认配置失败"
    [ $LAZY_8 -ne 0 ] && echo "  - Lazy Loading 压力配置失败"
    echo ""
    echo "详细日志保存在: /tmp/lazy-test-v2/"
    exit 1
fi
