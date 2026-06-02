#!/usr/bin/env bash
#
# run_kv_baseline.sh — KV cache baseline 实验脚本
#
# 目的:为 "runtime KV swap demo" 固化一组 CPU-only baseline 数据,
#       覆盖多个 context length，记录 RSS / prompt eval / decode eval /
#       tokens-per-second / 总耗时，供后续与 swap demo 对比。
#
# 本脚本【只采集 baseline】:
#   - 不修改任何源码;
#   - 不实现 runtime swap;
#   - 只调用已编译好的 build/bin/llama-completion。
#
# 所有关键变量均可通过环境变量覆盖，便于先跑短烟雾测试。
# 示例(短烟雾测试,只跑一组、各 1 次):
#   CTX_LENGTHS="512" REPEATS=1 N_PREDICT=16 PROMPT="Hello" \
#       bash scripts/run_kv_baseline.sh
#
set -euo pipefail

# ----------------------------------------------------------------------------
# 0. 路径与可覆盖变量
# ----------------------------------------------------------------------------
# 脚本所在目录的上一级 = 仓库根目录(scripts/ 的父目录)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 可被环境变量覆盖的实验参数(默认值见右侧)。
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-${REPO_ROOT}/build/bin/llama-completion}"
THREADS="${THREADS:-12}"                       # -t 线程数
NGL="${NGL:-0}"                                # -ngl 0 = CPU-only
N_PREDICT="${N_PREDICT:-128}"                  # -n 生成 token 数
REPEATS="${REPEATS:-3}"                        # 每组重复次数
CTX_LENGTHS="${CTX_LENGTHS:-512 1024 2048}"    # -c context length 列表(空格分隔)
SEED="${SEED:-42}"                             # 固定随机种子,保证可复现
# 固定 prompt(保持各组一致,避免 prompt 差异污染对比)。
PROMPT="${PROMPT:-Explain the concept of virtual memory in an operating system, including paging, demand loading, and page replacement, in a few clear paragraphs.}"

# 结果输出目录。
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/kv_baseline}"

# /usr/bin/time -v 用于采集 Maximum resident set size(峰值 RSS)。
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

# ----------------------------------------------------------------------------
# 1. 前置检查(任一关键路径缺失即退出,不创建任何文件)
# ----------------------------------------------------------------------------
echo "==============================================================="
echo " KV baseline 实验脚本"
echo "==============================================================="
echo " REPO_ROOT   = ${REPO_ROOT}"
echo " MODEL       = ${MODEL}"
echo " BIN         = ${BIN}"
echo " THREADS     = ${THREADS}"
echo " NGL         = ${NGL} (0 = CPU-only)"
echo " N_PREDICT   = ${N_PREDICT}"
echo " REPEATS     = ${REPEATS}"
echo " CTX_LENGTHS = ${CTX_LENGTHS}"
echo " SEED        = ${SEED}"
echo " RESULTS_DIR = ${RESULTS_DIR}"
echo "---------------------------------------------------------------"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ -x "${BIN}" ]       || fail "llama-completion 不存在或不可执行: ${BIN}"
[ -f "${MODEL}" ]     || fail "模型文件不存在: ${MODEL}"
[ -x "${TIME_BIN}" ]  || fail "/usr/bin/time 不存在: ${TIME_BIN} (请安装 GNU time)"

# ----------------------------------------------------------------------------
# 2. 准备输出目录
# ----------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"

# 记录本次运行的元信息,便于复盘。
META_FILE="${RESULTS_DIR}/run_meta.txt"
{
    echo "model=${MODEL}"
    echo "bin=${BIN}"
    echo "threads=${THREADS}"
    echo "ngl=${NGL}"
    echo "n_predict=${N_PREDICT}"
    echo "repeats=${REPEATS}"
    echo "ctx_lengths=${CTX_LENGTHS}"
    echo "seed=${SEED}"
    echo "prompt=${PROMPT}"
} > "${META_FILE}"
echo "已写入运行元信息: ${META_FILE}"
echo "---------------------------------------------------------------"

# ----------------------------------------------------------------------------
# 3. 实验主循环
# ----------------------------------------------------------------------------
# 命名约定:每组实验输出两个文件
#   ctx<CTX>_run<R>.log   —— llama-completion 原始输出(stdout+stderr,含 perf 行)
#   ctx<CTX>_run<R>.time  —— /usr/bin/time -v 的输出(含 Maximum resident set size)
total_runs=0
for ctx in ${CTX_LENGTHS}; do
    for r in $(seq 1 "${REPEATS}"); do
        total_runs=$((total_runs + 1))
    done
done

echo "计划共运行 ${total_runs} 次实验。"
echo "==============================================================="

run_idx=0
for ctx in ${CTX_LENGTHS}; do
    for r in $(seq 1 "${REPEATS}"); do
        run_idx=$((run_idx + 1))
        log_file="${RESULTS_DIR}/ctx${ctx}_run${r}.log"
        time_file="${RESULTS_DIR}/ctx${ctx}_run${r}.time"

        echo ""
        echo ">>> [${run_idx}/${total_runs}] ctx=${ctx} run=${r}"
        echo "    log : ${log_file}"
        echo "    time: ${time_file}"

        # -no-cnv: 关闭对话模式,走单轮 prompt->生成,输出稳定可解析。
        # 标准输出与标准错误都重定向到 log_file(perf 行在 stderr)。
        # /usr/bin/time -v 的统计写入 time_file。
        "${TIME_BIN}" -v -o "${time_file}" \
            "${BIN}" \
                -m "${MODEL}" \
                -t "${THREADS}" \
                -ngl "${NGL}" \
                -c "${ctx}" \
                -n "${N_PREDICT}" \
                -s "${SEED}" \
                -no-cnv \
                -p "${PROMPT}" \
            > "${log_file}" 2>&1 \
            || fail "实验失败 (ctx=${ctx} run=${r}),详见 ${log_file}"

        echo "    完成。"
    done
done

echo ""
echo "==============================================================="
echo " 全部 ${total_runs} 次 baseline 实验完成。"
echo " 原始日志位于: ${RESULTS_DIR}"
echo " 下一步运行: python3 scripts/parse_kv_baseline.py"
echo "==============================================================="
