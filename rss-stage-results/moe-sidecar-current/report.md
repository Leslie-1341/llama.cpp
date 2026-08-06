# MoE TG Profile

- model: `/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf`
- bench: `/root/llama.cpp-merge-test/build/bin/llama-bench`
- output: `rss-stage-results/moe-sidecar-current`
- columns: `summary.tsv` includes timing counters and reload-within-1/4-token thrash indicators.

```tsv
tag       rc  budget_mb  workers  peak_mb  pp_tps  tg_tps  streams  hits  evictions  resident_mib  bytes_read_mib  sidecar_read_mib  sidecar_read_count  moe_total_us  cache_lookup_us  victim_select_us  sidecar_submit_us  sidecar_wait_us  sidecar_read_us  q2_unpack_us  cache_hit  cache_miss  prefetch_hit  prefetch_late  prefetch_unused  evict_clean  evict_active_window  reload_1tok  reload_4tok  thrash_4tok_per_evict
b512_w1   0   512        1        633      4.04    5.19    3573     6592  3084       3.1           3723.0          3723.038          3573                2692311       0                460143            98828              0                2166585          0             4209       1184        2383          0              6                3084         0                    963          1827         0.592412
b512_w4   0   512        4        626      4.47    5.53    3576     6591  3087       3.1           3726.2          3726.164          3576                2355011       11887            457835            100835             11878            1859911          0             4209       1185        2382          0              9                3087         0                    966          1830         0.592809
b768_w1   0   768        1        881      4.73    7.08    2118     7070  1383       3.1           2206.9          2206.939          2118                1438676       0                253991            97777              0                1117757          0             5658       706         1412          0              0                1383         0                    333          666          0.481562
b768_w4   0   768        4        881      4.90    7.32    2118     7070  1383       3.1           2206.9          2206.939          2118                1361308       0                252660            97956              0                1047572          0             5658       706         1412          0              0                1383         0                    333          666          0.481562
b1536_w1  0   1536       1        1328     4.78    8.96    1167     7387  0          3.1           1216.0          1216.005          1167                752704        5                0                 46280              0                691233           0             6609       389         778           0              0                0            0                    0            0            0
b1536_w4  0   1536       4        1319     4.79    9.52    1167     7387  0          3.1           1216.0          1216.005          1167                645544        0                0                 46833              0                590850           0             6609       389         778           0              0                0            0                    0            0            0
b2432_w1  0   2432       1        1315     4.65    9.42    1167     7387  0          3.1           1216.0          1216.005          1167                647751        0                0                 46681              0                591636           0             6609       389         778           0              0                0            0                    0            0            0
b2432_w4  0   2432       4        1319     4.89    9.53    1167     7387  0          3.1           1216.0          1216.005          1167                661550        3                0                 47376              0                596109           0             6609       389         778           0              0                0            0                    0            0            0
```

- b512_w1: reload_4tok/evict_clean = 0.592, clear cache thrashing signal.
- b512_w4: reload_4tok/evict_clean = 0.593, clear cache thrashing signal.
- b768_w1: reload_4tok/evict_clean = 0.482, clear cache thrashing signal.
- b768_w4: reload_4tok/evict_clean = 0.482, clear cache thrashing signal.
