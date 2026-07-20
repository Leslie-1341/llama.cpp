#!/bin/bash
# 测试llama.cpp内置VM系统

echo "=========================================="
echo "内置VM系统测试"
echo "=========================================="
echo ""

MODEL_PATH="${1:-model.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path>"
    exit 1
fi

TEST_PROMPT="Tell me a short story about a robot."
NUM_TOKENS=100

mkdir -p /tmp/vm-test
cd /tmp/vm-test

# 函数：运行测试并监控RSS
run_test() {
    local name=$1
    shift
    local args="$@"
    
    echo "运行: $name"
    echo "参数: $args"
    
    /root/llama.cpp/build/bin/llama-cli \
        -m "$MODEL_PATH" \
        -p "$TEST_PROMPT" \
        -n $NUM_TOKENS \
        --single-turn \
        $args \
        > "${name}.log" 2>&1 &
    
    local pid=$!
    
    # 等待进程启动
    sleep 3
    
    # 持续监控RSS
    local max_rss=0
    local sum_rss=0
    local count=0
    
    while kill -0 $pid 2>/dev/null; do
        local rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{print $1}')
        if [ -n "$rss" ] && [ "$rss" -gt 0 ]; then
            [ $rss -gt $max_rss ] && max_rss=$rss
            sum_rss=$((sum_rss + rss))
            count=$((count + 1))
        fi
        sleep 0.5
    done
    
    wait $pid
    local exit_code=$?
    
    if [ $count -gt 0 ]; then
        local avg_rss=$((sum_rss / count))
        echo "  峰值RSS: $max_rss KB ($(echo "scale=2; $max_rss/1024" | bc) MB)"
        echo "  平均RSS: $avg_rss KB ($(echo "scale=2; $avg_rss/1024" | bc) MB)"
        echo "  退出码: $exit_code"
    else
        echo "  ⚠️  无法获取RSS数据"
    fi
    
    # 检查输出正确性
    if grep -qiE "\b(robot|story|the|and)\b" "${name}.log" 2>/dev/null; then
        echo "  ✅ 输出正常"
    else
        echo "  ❌ 输出异常"
    fi
    
    echo ""
    
    # 返回RSS值供对比
    echo "$max_rss $avg_rss"
}

echo "=========================================="
echo "测试1：Baseline（无VM）"
echo "=========================================="
result1=$(run_test "baseline" "")
read -r peak1 avg1 <<< "$result1"

echo "=========================================="
echo "测试2：VM DONTNEED"
echo "=========================================="
result2=$(run_test "vm_dontneed" "--vm-dontneed")
read -r peak2 avg2 <<< "$result2"

echo "=========================================="
echo "测试3：VM Sliding Unmap (window=4)"
echo "=========================================="
result3=$(run_test "vm_sliding_4" "--vm-dontneed --vm-sliding-unmap --vm-window-layers 4")
read -r peak3 avg3 <<< "$result3"

echo "=========================================="
echo "测试4：VM Sliding Unmap (window=8)"
echo "=========================================="
result4=$(run_test "vm_sliding_8" "--vm-dontneed --vm-sliding-unmap --vm-window-layers 8")
read -r peak4 avg4 <<< "$result4"

echo "=========================================="
echo "结果汇总"
echo "=========================================="
echo ""
echo "峰值RSS对比:"
echo "  Baseline:       $peak1 KB (100%)"

if [ "$peak1" -gt 0 ]; then
    reduction2=$(echo "scale=2; 100 * (1 - $peak2 / $peak1)" | bc)
    reduction3=$(echo "scale=2; 100 * (1 - $peak3 / $peak1)" | bc)
    reduction4=$(echo "scale=2; 100 * (1 - $peak4 / $peak1)" | bc)
    
    echo "  VM DONTNEED:    $peak2 KB ($reduction2%)"
    echo "  VM Sliding 4:   $peak3 KB ($reduction3%)"
    echo "  VM Sliding 8:   $peak4 KB ($reduction4%)"
else
    echo "  无法计算减少百分比"
fi

echo ""
echo "平均RSS对比:"
echo "  Baseline:       $avg1 KB (100%)"

if [ "$avg1" -gt 0 ]; then
    avg_reduction2=$(echo "scale=2; 100 * (1 - $avg2 / $avg1)" | bc)
    avg_reduction3=$(echo "scale=2; 100 * (1 - $avg3 / $avg1)" | bc)
    avg_reduction4=$(echo "scale=2; 100 * (1 - $avg4 / $avg1)" | bc)
    
    echo "  VM DONTNEED:    $avg2 KB ($avg_reduction2%)"
    echo "  VM Sliding 4:   $avg3 KB ($avg_reduction3%)"
    echo "  VM Sliding 8:   $avg4 KB ($avg_reduction4%)"
else
    echo "  无法计算减少百分比"
fi

echo ""
echo "=========================================="
echo "详细日志保存在: /tmp/vm-test/"
echo "=========================================="
