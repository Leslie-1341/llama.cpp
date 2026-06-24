# KV Paged Read - Stage 10-B WikiText-2 Perplexity Smoke 结果记录

## 1. 阶段目标

Stage 10-B 是进入真实语料验证的第一个小阶段。

本阶段目标不是验证真实 server workload，也不是验证多会话 paused/resume 调度，而是先确认：

1. 官方 WikiText-2 语料可以在当前服务器上获取、解压并使用；
2. `llama-perplexity` 可以在真实文本语料路径下正常运行；
3. 在 tail/lazy KV reclaim 配置下，连续文本推理路径没有出现明显运行异常。

本阶段属于 **真实语料连续文本路径 smoke**，不是最终的真实 serving 验证。

---

## 2. 语料检查

本阶段使用仓库推荐的 WikiText-2 raw 语料。

确认存在的语料文件为：

```text
wikitext-2-raw/wiki.test.raw
```

检查结果：

```text
official_wikitext_exists=1
-rw-rw---- 1 root root 1.3M Aug 15  2016 wikitext-2-raw/wiki.test.raw
```

下载得到的压缩包也经过完整性检查：

```text
zip_test_exit=0
Archive:  wikitext-2-raw-v1.zip
    testing: wikitext-2-raw/                 OK
    testing: wikitext-2-raw/wiki.test.raw    OK
    testing: wikitext-2-raw/wiki.valid.raw   OK
    testing: wikitext-2-raw/wiki.train.raw   OK
No errors detected in compressed data of wikitext-2-raw-v1.zip.
```

压缩包 `wikitext-2-raw-v1.zip` 仅用于下载和解压，检查完成后已删除，不纳入 git 跟踪。

清理后：

```text
git status --short
```

输出为空，说明没有遗留未跟踪文件或源码改动。

---

## 3. Smoke 配置

本阶段使用 `llama-perplexity` 运行 WikiText-2 test corpus。

模型路径：

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

语料路径：

```text
wikitext-2-raw/wiki.test.raw
```

关键参数：

```text
--ctx-size 2048
--batch-size 128
--ubatch-size 128
--cache-type-k f32
--cache-type-v f32
--kv-unified
--chunks 1
```

关键环境变量：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
```

---

## 4. Smoke 结果

运行结果：

```text
exit=0
```

stdout 末尾输出：

```text
0.52 minutes
[1]8.1375,
```

stderr 异常检查结果为空：

```text
tail err abnormal:
```

未发现以下异常关键词对应的真实异常：

```text
warn
error
failed
failure
nan
backend_fail
violation
assert
abort
segmentation
SIGSEGV
segv
```

---

## 5. 结果解释

Stage 10-B 已验证：

1. 官方 WikiText-2 语料在当前环境中可用；
2. `llama-perplexity` 能够在真实文本连续语料路径下运行；
3. tail/lazy KV reclaim 配置没有在该 smoke 中触发 crash、NaN、assert、backend failure 或 segmentation fault；
4. 本阶段没有引入源码改动；
5. 清理下载 zip 后，git 工作区保持干净。

可以严谨表述为：

```text
Stage 10-B 通过：WikiText-2 真实语料路径和 llama-perplexity 连续文本推理路径已 smoke 验证通过。
```

---

## 6. 边界与限制

本阶段不能被解释为真实 server workload 验证。

原因是 `llama-perplexity` 属于连续文本流评测路径，不天然产生以下生命周期：

1. 多 session；
2. paused idle；
3. resume pending；
4. idle-owned KV swap-out；
5. resume 前 prefetch；
6. defer swap-out；
7. request queue；
8. server slot；
9. HTTP / OpenAI API 请求；
10. continuous batching。

因此，本阶段只能说明：

```text
真实语料连续文本路径可跑；
tail/lazy 配置在该路径下未出现明显异常。
```

不能说明：

```text
idle swap / prefetch / defer 已经在真实多会话 server workload 中验证通过。
```

---

## 7. 阶段结论

Stage 10-B 完成了从人工 workload 向真实语料 workload 过渡的第一个 smoke。

核心结论：

```text
WikiText-2 官方语料可用；
llama-perplexity 连续文本路径可跑；
tail/lazy 配置下未发现明显运行异常；
当前结果不等价于真实 server workload 验证。
```

---

## 8. 下一步方向

下一步进入 Stage 10-C：corpus-backed semi-real multi-session workload。

目标是：

```text
把 Stage 9 semi-real multi-session driver 中的固定 prompt 替换为真实语料片段；
保留多 session、paused idle、resume pending、prefetch、defer 等生命周期；
在真实文本驱动的多会话暂停/恢复 workload 下重新验证 RSS / latency trade-off。
```

建议 Stage 10-C 先做设计审计，再改 driver：

```text
Stage 10-C-A：corpus-backed semi-real workload 设计审计；
Stage 10-C-B：只修改 example driver，支持从 corpus 文件读取 session prompt；
Stage 10-C-C：跑 S0/S2/S4/S5 或 S0-S5 3-run median；
Stage 10-C-D：整理结果文档。
```

当前暂不直接进入 `llama-server` / ShareGPT / k6 benchmark。
