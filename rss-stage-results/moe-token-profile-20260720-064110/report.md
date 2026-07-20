# MoE TG Profile

- model: `/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf`
- bench: `/root/llama.cpp/build/bin/llama-bench`
- output: `/root/llama.cpp/rss-stage-results/moe-token-profile-20260720-064110`
- columns: `summary.tsv` includes timing counters and reload-within-1/4-token thrash indicators.

```tsv
tag       rc  budget_mb  workers  peak_mb  pp_tps  tg_tps  streams  hits  evictions  resident_mib  bytes_read_mib  sidecar_read_mib  sidecar_read_count  moe_total_us  cache_lookup_us  victim_select_us  sidecar_submit_us  sidecar_wait_us  sidecar_read_us  q2_unpack_us  cache_hit  cache_miss  prefetch_hit  prefetch_late  prefetch_unused  evict_clean  evict_active_window  reload_1tok  reload_4tok  thrash_4tok_per_evict
b512_w1   0   512        1        817      2.13    2.65    9027     3690  8745       510.8         0.0             16581.984         9027                8868105       213787           910424            21939              213715           18435023         0             2826       4086        864           88             4029             8745         435                  4419         6501         0.743396
b512_w4   0   512        4        638      1.66    2.48    11565    3919  11289      508.1         0.0             21229.828         11565               9368115       216299           1357198           46471              216201           29620534         0             2748       3857        1171          199            6531             11289        1707                 6348         9045         0.801222
b2432_w1  0   2432       1        2533     3.33    15.57   1317     7247  0          2416.0        0.0             2416.047          1317                1079098       51217            0                 1581               51162            2751951          0             6600       529         647           28             0                0            0                    0            0            0
b2432_w4  0   2432       4        2537     4.74    16.78   1317     7577  0          2416.0        0.0             2416.047          1317                604327        134833           0                 8309               134759           3561477          0             6600       199         977           138            0                0            0                    0            0            0
```

- b512_w1: reload_4tok/evict_clean = 0.743, clear cache thrashing signal.
- b512_w4: reload_4tok/evict_clean = 0.801, clear cache thrashing signal.
