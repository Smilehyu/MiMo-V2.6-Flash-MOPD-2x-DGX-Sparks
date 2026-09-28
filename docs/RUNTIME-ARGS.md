# Runtime arguments — what this deployment actually starts SGLang with

Source and provenance: the boot of **2026-09-28 09:52:04Z** on the head, read back at
**2026-09-28T10:11:44Z** (UTC) from the running engine. The engine's own copy of the launch
line is `GET /server_info` → `launch_command`; the container writes the same array to
`state/launch.json`. **`state/launch.json` is gitignored and is not published** — this file is
the excerpt, with every site-specific value replaced by a placeholder.

Profile in force: `.env` as copied from `.env.example` at that boot, i.e. MOPD checkpoint,
`CHUNKED_PREFILL_SIZE=8192`, `MEM_GUARD_GIB=2`, `MEM_FRACTION_STATIC=0.93`, `SPEC_ALGO=EAGLE`.
Measured consequences (pool 2,546,496 tokens, 82.9 GB weights + 17.2 GB KV per rank,
284 s cold start) are in [`TEST-RESULTS.md`](TEST-RESULTS.md) §2 and §3.

## The launch line, as recorded

`<HEAD_IP>` / `<WORKER_IP>` are the two nodes' fabric addresses, `<HEAD_IP>:20000` is the
`--dist-init-addr` value taken from `DIST_PORT=20000`.

```text
python3 -u boot.py run
  --model-path /models/MiMo-V2.6-Flash-MOPD
  --served-model-name MiMo-v2.6-Flash
  --tp-size 2 --ep-size 2 --nnodes 2 --node-rank <0|1>
  --dist-init-addr <HEAD_IP>:20000
  --context-length 1048576
  --mem-fraction-static 0.93
  --max-total-tokens 2743808
  --max-running-requests 8
  --chunked-prefill-size 8192 --max-prefill-tokens 16384
  --page-size 64
  --kv-cache-dtype fp8_e4m3
  --swa-full-tokens-ratio 0.03
  --attention-backend triton
  --mm-attention-backend triton_attn
  --moe-runner-backend flashinfer_mxfp4
  --moe-a2a-backend none
  --moe-dense-tp-size 1
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --enable-multi-layer-eagle
  --enable-multimodal
  --enable-metrics
  --enable-cache-report
  --weight-loader-drop-cache-after-load
  --disable-cuda-graph
  --disable-prefill-cuda-graph
  --sleep-on-idle
  --watchdog-timeout 1800
  --reasoning-parser mimo
  --tool-call-parser mimo
```

## Flag by flag, with where each value comes from

| flag | value | source / notes |
|---|---|---|
| `--model-path` | `/models/MiMo-V2.6-Flash-MOPD` | must equal `Dockerfile`'s `ENV MODEL_PATH` and `start.sh`'s `MODEL_MOUNT`; the checkpoint is bind-mounted read-only from `$MODEL_DIR` on the head and over NFSv4 on the worker |
| `--served-model-name` | `MiMo-v2.6-Flash` | `SERVED_MODEL_NAME`; the id `/v1/models` reports |
| `--tp-size` / `--ep-size` / `--nnodes` | 2 / 2 / 2 | one GPU per Spark; `DP_SIZE=1`, so no `--enable-dp-attention` / `--enable-dp-lm-head` |
| `--dist-init-addr` | `<HEAD_IP>:20000` | `HEAD_IP` + `DIST_PORT`; rank 0 is the rendezvous. Site-specific — this is the value that is redacted everywhere it appears |
| `--context-length` | 1,048,576 | the model's maximum (`max_req_input_len` came back 1,048,570) |
| `--mem-fraction-static` | 0.93 | `MEM_FRACTION_STATIC`; the deviation and its measurement are in README "Deviations from upstream" |
| `--max-total-tokens` | 2,743,808 requested | `MAX_TOTAL_TOKENS`, a ceiling. **The engine allocated 2,546,496** and logged `max_total_tokens=2743808 is larger than the profiled value 2546496. Use the profiled value instead.` Quote the allocation, not the request |
| **KV pool as allocated** | **2,546,496 tokens** | `/v1/loads` → `max_total_num_tokens` and `memory.token_capacity`, matching the boot log's `Use sliding window memory pool` line (SWA side pool 76,352) |
| `--max-running-requests` | 8 | `MAX_RUNNING_REQUESTS` |
| `--chunked-prefill-size` / `--max-prefill-tokens` | 8192 / 16384 | **deviation from upstream** (upstream 2048, and its Defaults table warns against raising it at 0.93). Measured ladder in TEST-RESULTS §7 |
| `--page-size` | 64 | `PAGE_SIZE` (upstream recipe uses 1; this is a fragmentation knob) |
| `--kv-cache-dtype` | `fp8_e4m3` | `KV_CACHE_DTYPE` |
| `--swa-full-tokens-ratio` | 0.03 | `SWA_FULL_TOKENS_RATIO`; drives the SWA pool cost — see README "KV geometry" |
| `--attention-backend` / `--mm-attention-backend` | `triton` / `triton_attn` | GB10; `fa4` is not an SM121 path |
| `--moe-runner-backend` | `flashinfer_mxfp4` | consumes the checkpoint's packed MXFP4 experts |
| `--moe-a2a-backend` / `--moe-dense-tp-size` | `none` / 1 | plain expert parallel; `deepep` needs NVSHMEM across the fabric and is untested on two Sparks |
| `--speculative-algorithm` | `EAGLE` | the shipped default; per-category acceptance in TEST-RESULTS §8 |
| `--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens` / `--enable-multi-layer-eagle` | 3 / 1 / 4 / on | `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT_TOKENS`; EAGLE-only |
| `--enable-multimodal` | always on | the MOPD checkpoint is omni (text/image/video/audio); the image installs `ffmpeg` + `torchcodec` and refuses to build without them |
| `--enable-metrics` / `--enable-cache-report` | on | `ENABLE_METRICS` / `ENABLE_CACHE_REPORT`; what `bench/` reads |
| `--weight-loader-drop-cache-after-load` | on | `WEIGHT_LOADER_DROP_CACHE=1`; on unified memory the page cache and the KV budget are one pool |
| `--disable-cuda-graph` / `--disable-prefill-cuda-graph` | on | `CUDA_GRAPH_MAX_BS_DECODE=0` and `CUDA_GRAPH_PREFILL=0`. Decode graph capture **crashes** under EAGLE + `--enable-multi-layer-eagle` (README "The EAGLE MTP failure") — this is not a slow-down knob |
| `--sleep-on-idle` | on | `SLEEP_ON_IDLE=1`; the first request after a quiet period pays a wake-up cost |
| `--watchdog-timeout` | 1800 | long enough for a cold start whose worker loads weights over NFS for ~230 s |
| `--reasoning-parser` / `--tool-call-parser` | `mimo` | `REASONING_PARSER` / `TOOL_CALL_PARSER` |

## Reading the truth out of a running server

```bash
curl -s http://127.0.0.1:8888/server_info | python3 -m json.tool   # launch_command, startup_time
curl -s http://127.0.0.1:8888/v1/loads   | python3 -m json.tool    # the allocated pool, weights, KV
docker logs mimo26-tp2-head 2>&1 | grep -E 'max_total_num_tokens=|Use sliding window'
```

`boot.py`'s own `preflight:` line over-states the budget (it is computed before the engine
profiles the pool); the `max_total_num_tokens=` line above is the authority.
