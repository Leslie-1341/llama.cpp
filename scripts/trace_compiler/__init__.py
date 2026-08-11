# -*- coding: utf-8 -*-
"""Alibaba/Qwen-Bailian trace workload compiler package.

Streaming compiler that turns the anonymized production trace into frozen
canonical workload manifests for the V3 KV-offload work. See common.py for
the pinned dataset and the runtime/ground-truth field contract.

This package is intentionally NOT a second benchmark runner/parser. It only
emits versioned manifests; the canonical runner lives at
scripts/run-kv-offload-benchmark.py (owned by the V3-1A work) and is not
modified here.
"""
