# MiMo-V2.6-Flash-MOPD on 2× DGX Spark (TP2/EP2)

Serve [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
— 309B total, 15B active, MXFP4 experts, 1M context, text / image / video / audio — with
SGLang on **two** DGX Sparks (GB10, SM121, 121.7 GiB unified memory each) over ConnectX-7
RoCE, behind one OpenAI-compatible endpoint on the head. The head keeps the only copy of the
checkpoint and exports it over NFSv4; the worker reads it from there.

This is Smilehyu's deployment kit for the MOPD checkpoint: a two-node SGLang profile, a
ModelScope downloader with a strict integrity gate, the engine patches the TP2/EP2 layout
needs, and the measured results of this exact configuration.

It is a **CLI kit**. There is no dashboard, no proxy and no second control plane: two
containers are started and stopped by `./start.sh` / `./stop.sh`, and everything observable
comes from the engine's own HTTP endpoints and `docker logs`.

## Status

Honest state of this deployment. Every number below was measured on **this** configuration —
the MOPD checkpoint, `MEM_FRACTION_STATIC=0.93`, `CHUNKED_PREFILL_SIZE=8192`,
`MEM_GUARD_GIB=2`, `SPEC_ALGO=EAGLE` — and every one can be traced to a command in
[`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md), which records the raw output and the UTC
collection time. Anything unmeasured says so.

| | |
|---|---|
| Boots and serves | yes, TP=2/EP=2 over RoCE, `/health` 200, text + image + video + audio |
| Cold start | **284 s** container start → `fired up` (2026-09-28 09:52:04Z → 09:56:49Z) |
| Weight load | head **115.5 s** / 82.92 GB resident; worker **229.3 s** / 83.42 GB read over NFS from the head |
| Checkpoint | 155 blobs, 177,765,031,619 B (165.56 GiB), **128** expert shards, index 6,921,726 B; `verify-weights.py` PASS (0 missing, 0 size mismatches, 29 files sha256-verified, 128/128 shards) |
| **KV pool** | **2,546,496 tokens** (`max_total_num_tokens`) = 2.43× a full 1M context; SWA side pool 76,352; `available_gpu_mem` 9.15 GB, host available 6,186 MiB |
| Prefill (cold, this fleet) | peak **2,311 tok/s** at 16k, 2,189 at 8k, 1,750 at 64k, 1,273 at 128k; TTFT 3.66 s → 100.56 s. Prefix-cache re-visit of the same 8k prompt: **TTFT 0.34 s** (`cached=8000`, 10.8×) |
| Decode (EAGLE MTP, this fleet) | aggregate 31.8–46.8 tok/s at 1 stream → 98.3–226.0 at 8, by category; per-stream 38.6 → 15.0 tok/s (prose) and 63.4 → 40.0 (counting) |
| Speculative decoding | **EAGLE MTP by default** (draft 4, steps 3, topk 1, multi-layer). Median accepted length **2.5–4.0 of 4**: ~4.0 on counting/math, 2.5–2.75 on prose |
| Context | 1,048,576 configured and the pool holds it — one full-context request fits |
| CUDA graphs | decode capture is **off** and must stay off under EAGLE multi-layer (it crashes the boot, see "The EAGLE MTP failure"); prefill capture off |
| Multimodal | image path executes and is asserted by `scripts/check-serving.sh` (left/right colour split, `image_tokens` 256); a real screenshot was identified from its own contents (200 image tokens) and a 1.5 MB real photograph used 8,100. **Object recognition on a real photograph was wrong** — see §6 of the test results for the exact prompts and the honest result |
| Not measured | long-context (>128k) prefill, concurrency above 8, decode beyond 256 tokens/stream, the DFlash drafter on this checkpoint, needle/retrieval quality |
| Earlier observations | this fleet previously served the same architecture at a different checkpoint and memory fraction; those numbers (a **2,878,144**-token pool, ~19–28 tok/s uncaptured decode, a ~95 s/rank load) belong to that configuration and are **not** reproduced here. The pool value quoted for this profile is 2,546,496 |

Getting here took four node reboots and two OOM kills. "Why the first boots crashed", "The
EAGLE MTP failure" and "Deviations from upstream" are the useful parts of this README if you
are porting something else to a Spark.

## Requirements

- 2× DGX Spark (GB10), one GPU each, linked over ConnectX-7. A single Spark cannot serve this
  model at all — ~166 GiB of weights on a 121.7 GiB box.
- ~170 GiB of free disk on the head for the checkpoint, plus Docker.
- Passwordless ssh from the head to the worker.
- `BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5` or newer: MiMo-V2.6's MXFP4
  experts need [sgl-project/sglang#40448](https://github.com/sgl-project/sglang/pull/40448).
  Older engines expand the experts to FP8 and take the node down.

The checkpoint is 177,765,031,619 B (165.56 GiB) over 155 files, **128 expert shards**
(`model_pp0_ep0` … `ep127`), 256 routed experts, 15B activated, 1M context, text / image /
video / audio, 3-layer MTP. The routed experts are stored **MXFP4** (U8 nibble pairs plus E8M0
block-32 scales, 149.8 GiB for all 256); attention and the dense MLP are FP8 e4m3 with
block-128 `weight_scale_inv`; embeddings, `o_proj`, gates and the encoders are BF16.

| Layout | Resident weights per GPU | Fits 121.7 GiB? |
|---|---:|---|
| **2 ranks (TP=2)** | **~82.9 GiB** (measured 82.92 head / 83.42 worker, EAGLE MTP drafter) | **yes**, ~9 GiB left outside the KV pool at `MEM_FRACTION_STATIC=0.93` |

Those numbers only hold while the engine keeps the experts in their stored 4-bit layout.
`boot.py` measures them from the safetensors headers and prints them as `preflight: …` before
SGLang starts.

## Divisibility

`config.json` has 64 query heads, 4 global KV heads, 8 SWA KV heads and 256 experts. The
safetensors index metadata records `"tp_size": 4` for the fused `qkv_proj`. The pinned engine
reshards that onto TP=2 itself: it merges two checkpoint shards per rank, then de-interleaves
`[q0 k0 v0 q1 k1 v1]` into `[q0 q1 | k0 k1 | v0 v1]` with a dequant/requant of the block
scales (`Resolved deferred qkv scale_inv … de-interleaved QKV` in the log). The MTP draft
loader (`mimo_v2_nextn.py`) does not use that path yet, so `patches/sitecustomize.py` adds it
for the draft through an import hook; without it every rank fails right after the 82 GiB
target load with

```text
qkv_proj scale_inv model.mtp_block.self_attn.qkv_proj.weight_scale_inv: shape mismatch (116, 32) vs (58, 32)
```

See [`patches/README.md`](patches/README.md). The pin is what makes `patches/` load-bearing:
the patch becomes a no-op once the engine passes the deferred dict itself, and it is the only
place this kit touches the engine.

## Topology

```text
head    <HEAD_IP>    rank 0   API :8888, NFSv4 export of the checkpoint
worker  <WORKER_IP>  rank 1   mounts the head over ConnectX and reads the weights from there
```

Workers do **not** keep a copy of the checkpoint. `share` exports the head's tree and mounts
it on the worker; `serve` runs that check first and will not start a rank until the worker can
read the head's `model.safetensors.index.json` (same byte size) and all 128 expert shards.

Inside both containers the checkpoint is bind-mounted read-only at
`/models/MiMo-V2.6-Flash-MOPD`. Three places must name that path and they must agree —
`Dockerfile`'s `ENV MODEL_PATH`, `boot.py`'s `MODEL` default and `start.sh`'s `MODEL_MOUNT` —
because a mismatch makes the engine read an empty directory instead of failing loudly.
`HF_REPO` / `HF_REVISION` from the profile are reported as the model's metadata and nothing
more; the checkpoint on disk is whatever `MS_REVISION` downloaded.

The worker's own check line is the strongest evidence that the export works:

```text
[+] NFS volume mimo26-tp2-weights on <WORKER_IP> → nfs://<HEAD_IP>:/
ok index=6921726 shards=128
```

## Quick start

```bash
git clone https://github.com/Smilehyu/MiMo-V2.6-Flash-MOPD-2x-DGX-Sparks.git mimo26 && cd mimo26
cp .env.example .env            # then edit the cluster / fabric / NFS block
./start.sh doctor               # image label, engine + memory preflight, ssh/IB, shard count
./start.sh download             # ~166 GiB, once, on the head, from ModelScope
./start.sh build                # FROM $BASE_IMAGE, then docker save | load to the worker
./start.sh share                # NFSv4 export on the head + volume on the worker
./start.sh serve                # start both ranks
./scripts/check-serving.sh      # read-only: status, mounts, text smoke, asserting image smoke
```

`./mimo26.sh start` is the same thing wrapped with the GB10 pre-launch cache ritual
(below). `./start.sh` with no argument is `doctor → image → share → serve`.

### Get the checkpoint

`./start.sh download` fetches the pinned revision to `$MODEL_DIR`
(`$HOME/NewModels/MiMo-V2.6-Flash-MOPD` by default) on **the head only** — workers read it
over NFS and never keep a copy.

```bash
df -h "$HOME/NewModels"         # need ~170 GiB free
./start.sh download             # 30-90 min on a fast link; resumable
```

The download goes to **ModelScope**, not Hugging Face: it uses `MS_REPO`, `MS_REVISION` and
`MS_PYTHON` from `.env`, so no VPN and no HF token are needed.

| key | value shipped | why |
|---|---|---|
| `MS_REPO` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | the checkpoint repository |
| `MS_REVISION` | `24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1` | the revision this deployment downloaded and verified file by file |
| `MS_PYTHON` | `$HOME/ai-stack/venvs/ms/bin/python` | any interpreter with `modelscope` installed — change it to yours |
| `HF_REPO` / `HF_REVISION` | same repo / `2479e2d0…` | **informational only.** These are reported as the model's `revision` metadata by `/v1/models`; nothing is downloaded from Hugging Face, and the HF revision was never verified by a download here |
| `EXPECTED_SHARDS` | `128` | the MOPD layout splits the experts into 128 files (an earlier RL checkpoint of the same architecture had 64) |

`download-ms.sh` is the standalone, strictly-checked version of the same step and can also be
used to re-verify an existing tree:

```bash
./download-ms.sh                # uses .env when present
```

What lands, and why all of it matters:

| | |
|---|---|
| `model_pp0_ep0 … ep127` | 128 expert shards, ~150 GiB, the MXFP4 routed experts |
| `model_mtp.safetensors` | 1.2 GiB, the 3-layer MTP head — the **EAGLE MTP** drafter |
| `dflash/` | 2.9 GiB separate Qwen3 draft model — the **DFlash** drafter |
| `audio_tokenizer/`, `preprocessor_config.json` | audio and vision preprocessing |
| `config.json` | must contain `vision_config` **and** `audio_config` |

**A partial download is the failure mode to watch for.** It leaves valid-looking JSON beside
missing weights, and the engine then dies mid-load. Both `download-ms.sh` and `start.sh
download` therefore end with the same gate: the five files `boot.py` requires, every shard the
index references, and at least 128 expert shard files. Failure prints the count and the first
few names rather than a confusing load error.

`scripts/verify-weights.py` goes further when you want the whole checkpoint checked against
the published metadata — every upstream file's byte size, sha256 for everything under 256 MiB
plus a sample of the large shards, and the shard count:

```bash
./scripts/verify-weights.py     # read-only; safe to run while serving
```

```text
upstream files: 155
expected total: 177765031619 bytes (165.56 GiB)
missing files       : 0 []
size mismatches     : 0 []
sha256 verified     : 29 files, mismatches: 0 []
expert shards       : upstream 128 / on disk 128
VERIFY_RESULT: PASS
```

Re-running `download` after an interruption resumes rather than restarting, and the revision
is pinned (`MS_REVISION`) so a re-run cannot silently pull a different checkpoint — if the
pinned revision fails, `download-ms.sh` says so loudly **before** falling back to `master`.
Already have the checkpoint elsewhere? Point `MODEL_DIR` at it in `.env` and skip this step —
`share` and `serve` validate whatever is there.

### Share the checkpoint over NFS

```bash
./start.sh share                # NFSv4 export of $MODEL_DIR on the head + worker volume
```

`files/nfs-share.sh` builds a small exporter image (`files/nfs-server/`), exports the tree
read-only to the addresses in `NFS_CLIENTS`, and mounts it on the worker. The gate is the
worker-side check quoted in "Topology": same index byte count, 128 shards. If you switch
checkpoints, run `./stop.sh --unmount` first — a live exporter that still holds the old path
is the one failure this step cannot recover from on its own.

### Serve

```bash
./start.sh serve                # or ./mimo26.sh start (adds the cache ritual)
./start.sh status
./start.sh logs worker
```

`./start.sh serve` runs the download and share checks first and refuses to launch anything if
they fail. First boot on a cold machine takes ~5 minutes (see "Status"); the head finishes its
load in 115 s and the worker's NFS read is what fills the rest.

### Talk to it

```bash
curl http://<HEAD_IP>:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-v2.6-Flash","messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":32,"temperature":1.0,"top_p":0.95,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

The shipped sampling is temperature 1.0 and top_p 0.95. Thinking is on unless
`enable_thinking` is false. Parsers are `mimo` for tools and reasoning. No API key is
configured by default; set `API_KEY` in the profile to require one.

## The GB10 cache ritual

On GB10, host RAM *is* GPU memory, so a launch that follows big file reads sees a large page
cache, and SGLang sizes its KV budget from `MemAvailable`. `./mimo26.sh start|restart` runs
`sync` + `drop_caches` + `swappiness=0` on **both** nodes first:

```bash
MIMO26_SUDO_PW=... ./mimo26.sh start        # start | restart | stop | status | logs | smoke | doctor | download
```

The password is read from `MIMO26_SUDO_PW` and piped into `sudo -S` on stdin — it is never
written into a command line locally or over ssh. With passwordless sudo on both nodes you can
use `./start.sh` directly; note that `start.sh`'s own `sudo -n drop_caches` **silently skips**
when a password is required, which is exactly what this wrapper exists for. The worker address
comes from `WORKER_USER` + the first entry of `WORKER_IPS` in `.env`.

## Defaults

Everything below ships in `.env.example`, which `start.sh` copies to `.env` on first run.
These are the values this kit was measured at, not upstream's — the divergences and their
reasons are in "Deviations from upstream" below.

| | default | why |
|---|---|---|
| `TP_SIZE` / `EP_SIZE` / `NNODES` | 2 / 2 / 2 | one GPU per Spark; a single Spark cannot hold the model |
| `DP_SIZE` | 1 | one GPU per node, so the DP flags stay off |
| `CONTEXT_LENGTH` | 1,048,576 | the model maximum, and the pool holds it |
| `MAX_TOTAL_TOKENS` | 2,743,808 | a ceiling above the budget, so the budget binds. The engine allocated **2,546,496** measured. Drop to ~2,299,968 for DFlash, whose drafter needs its own pool |
| `MEM_FRACTION_STATIC` | 0.93 | aggressive on purpose; the boot ends with `available_gpu_mem=9.15 GB` |
| `MEM_GUARD_GIB` | 2 | **deviation** — 4 killed both ranks during the FlashInfer autotune phase in this profile |
| `SWA_FULL_TOKENS_RATIO` | 0.03 | the upstream Flash value. 0.3 (V2.5-era) cost 10x more for the sliding-window pool |
| `MAX_RUNNING_REQUESTS` | 8 | also the decode graph tiers, so every reachable batch replays a graph |
| `CHUNKED_PREFILL_SIZE` | 8192 | **deviation** — upstream ships 2048 and warns against raising it at 0.93. This profile is measured at 8192 |
| `PAGE_SIZE` | 64 | upstream uses 1; this is a fragmentation knob, not a capacity one |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | nothing smaller exists |
| `CUDA_GRAPH_MAX_BS_DECODE` | 0 | **required** by the default EAGLE MTP drafter — capture crashes under it. Set to 8 with DFlash, where it works |
| `CUDA_GRAPH_PREFILL` | 0 | prefill capture off, as upstream also does |
| `SPEC_ALGO` / `SPEC_DRAFT_TOKENS` | `EAGLE` / 4 | the arm measured here; DFlash is not benchmarked in this repository |
| `SPEC_STEPS` / `SPEC_TOPK` / `ENABLE_MTP` | 3 / 1 / 1 | EAGLE only; DFlash forces its own and ignores these |
| `ATTENTION_BACKEND` / `MM_ATTENTION_BACKEND` | `triton` / `triton_attn` | `fa4` is not an SM121 path |
| `MOE_RUNNER_BACKEND` | `flashinfer_mxfp4` | consumes the packed MXFP4 experts; `auto` would fail on them |
| `MOE_A2A_BACKEND` / `MOE_DENSE_TP_SIZE` | `none` / 1 | `deepep` needs NVSHMEM across the fabric, untested on two Sparks |
| `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` | 0 | byte-identical greedy output, ~0.3 ms/step |
| `WEIGHT_LOADER_DROP_CACHE` | 1 | page cache and the KV budget are the same pool here |
| `ENABLE_METRICS` / `ENABLE_CACHE_REPORT` | 1 / 1 | what `bench/` reads |
| `OFFLOAD_MODE` | `off` | `ssd` spills HiCache L3; `ram`/`cpu` are refused |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | `True` produced NaN logits on this GB10 stack |
| `NCCL_MAX_NCHANNELS` | 16 | **deviation** — 2 RoCE rails × 8 channels; see below |

The cluster half of the profile (`HEAD_IP`, `WORKER_IPS`, `WORKER_HOSTS`, `WORKER_USER`, the
`NCCL_*` and `NFS_*` block, `MODEL_DIR`) is site-specific — edit it for your fleet. The
examples shipped in `.env.example` are documentation examples, not this deployment's
addresses. The container, volume and worker-directory names still carry a `tp2` suffix from
when this kit had two profiles; they are only identifiers.

## Deviations from upstream

This kit is derived from an earlier Spark kit for the same architecture (see the
"License & attribution" note at the end), which documents it at a different checkpoint. Five
differences are deliberate, and each is backed by
a measurement in [`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md):

1. **`CHUNKED_PREFILL_SIZE=8192`** (upstream 2048, and upstream's Defaults table says *do not
   raise this at 0.93*). The whole point of the larger chunk on this box is the long-prompt
   case: 128k becomes 16 chunks instead of 64. Measured with this profile: the cold-prefill
   ladder 8k → 128k ran 5/5 with no `MEMORY GUARD` line, and the profile has cold-booted
   repeatedly (test results §7). **If you are conservative, or if a boot ends with
   `MEMORY GUARD: MemAvailable … killing sglang`, put 2048 back** — the upstream warning is
   about the prefill transient living in the slack the memory fraction keeps outside the pool,
   and a larger `CHUNKED_PREFILL_SIZE` spends that slack.
2. **`MEM_GUARD_GIB=2`** (upstream 4). At 0.93 + 8192 this profile's boot ends with
   `available_gpu_mem=9.15 GB` and 6,186 MiB host-available, so a 4 GiB userspace OOM trigger
   fires during FlashInfer autotune and kills both ranks — the only symptom is
   `Server failed to become healthy`. 2 leaves air above the trigger. Raise it back to 4 if
   you lower `MEM_FRACTION_STATIC`.
3. **`NCCL_MAX_NCHANNELS=16` and multi-rail RoCE** (upstream 8, single rail). This fleet links
   the two Sparks with two NICs whose ports sit on different subnets, so `cmd_serve` in
   `start.sh` collects **every** HCA pair that shares a subnet (upstream kept only the first
   one) and merges them into a multi-rail `NCCL_IB_HCA` list, with `NCCL_PROTO=^LL128` and the
   trimmed `NCCL_BUFFSIZE`/`NCCL_LL128_BUFFSIZE` values. 16 = 2 rails × 8 channels. Both
   guards upstream relies on are unchanged: it still dies with
   `no shared RoCE subnet …` when the nodes share none, and with
   `RoCE port pair was not selected` when a pair was found but could not be pinned. If your
   nodes are linked by a single NIC, set this back to 8.
4. **ModelScope instead of Hugging Face** for the download (`MS_REPO`, `MS_REVISION`,
   `MS_PYTHON`, and `EXPECTED_SHARDS=128`). Upstream's HF three-step fallback chain is not
   used at all; `HF_REPO`/`HF_REVISION` survive only as the metadata `/v1/models` reports.
5. **Measured results belong to this checkpoint and this harness.** The decode and prefill
   tables in this README come from `bench/` runs recorded in
   [`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md), on the MOPD checkpoint, issued from the head
   against the loopback endpoint. Numbers that this repository cannot reproduce — the DFlash
   arm, uncaptured-decode baselines, and the pool size of an earlier profile — are either
   labelled as earlier observations or left out.

## Launch flags

The reference recipe is SGLang's
[MiMo-V2.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.6)
(Flash on 4× B300, single node, from
[sgl-project/sglang#40448](https://github.com/sgl-project/sglang/pull/40448)): `--tp 4 --ep 4
--dp 1 --pp-size 1 --moe-runner-backend deep_gemm --moe-a2a-backend deepep --deepep-mode auto
--moe-dense-tp-size 1 --enable-dp-lm-head --attention-backend fa4 --mm-attention-backend fa4
--mm-enable-dp-encoder --context-length 1048576 --mem-fraction-static 0.6
--swa-full-tokens-ratio 0.03 --chunked-prefill-size 49152 --max-prefill-tokens 65536
--page-size 1 --speculative-algorithm DFLASH --speculative-num-draft-tokens 8
--reasoning-parser mimo --tool-call-parser mimo --enable-cache-report`, with running requests
and the decode CUDA-graph batch size both at 64 and **prefill CUDA graphs disabled**.

**This kit was originally built against the V2.5 recipe, which the V2.6 cookbook explicitly
tells you not to carry over** ("Do not add `--enable-dp-attention` or copy EAGLE-specific
settings from the V2.5 cookbook"). Two settings came from there and were wrong for V2.6: the
EAGLE drafter, and `--swa-full-tokens-ratio 0.3` where Flash wants **0.03** — a 10×
over-allocation of the expensive sliding-window pool (see "KV geometry"). Both were corrected
on 2026-09-22.

What still differs here, and why. Each Spark is one GPU, so `DP_SIZE=1` and the DP flags stay
off, and `--ep` equals `--tp` at 2. `fa4` is not an SM121 path, so attention stays Triton.
`deep_gemm`/`deepep` for the MoE need NVSHMEM across the fabric and are untested on two
Sparks, so the runner is `flashinfer_mxfp4` with `--moe-a2a-backend none`.
`--mem-fraction-static 0.93` against the cookbook's 0.6 is forced by unified memory: host RAM
*is* GPU memory here, so a large reserve outside the pool starves KV rather than protecting
it. `--page-size 64` against 1 is a memory-bound choice on a 121.7 GiB box and has not been
laddered. `--chunked-prefill-size` is 8192 rather than the cookbook's 49152 or upstream's
2048 — see the deviation note above.

MoE on this fleet:

- `MOE_RUNNER_BACKEND=flashinfer_mxfp4`: the experts stay packed and run on FlashInfer's
  SM120 cutlass MXFP8×MXFP4 kernel. `marlin` (W4A16) is the alternative. `auto` keeps the
  memory but hands packed tensors to the FP8 Triton kernels.
- `MOE_A2A_BACKEND=none`: plain expert parallel with all-reduce. The cookbook's `deepep` needs
  NVSHMEM across the fabric and is untested on two Sparks.
- `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`: byte-identical greedy output, ~0.3 ms per step.
- `WEIGHT_LOADER_DROP_CACHE=1` → `--weight-loader-drop-cache-after-load`: each shard's page
  cache is released once its tensors are on the GPU. On unified memory that cache and the KV
  budget come from the same pool.

The full argument list of the boot this README documents (with the site-specific addresses
replaced by placeholders) is in [`docs/RUNTIME-ARGS.md`](docs/RUNTIME-ARGS.md).

## Weight loading

The three loader paths, measured on this fleet's head (GB10, NVMe, one 2.3 GiB expert shard
of 1128 tensors) for the same architecture:

| path | read | copies to GPU | per shard | 82 GiB rank |
|---|---:|---:|---:|---:|
| mmap, copy straight from the view (SGLang default) | 0.4 s | 17.7 s | ~18 s | 525 s (measured) |
| `--weight-loader-disable-mmap` (`read()` + parse) | 3.1 s | 1.35 s | ~4.5 s | 131 s (measured), crashed both nodes |
| mmap, `clone()` one tensor, then copy (the patch) | 0.4 s | 1.15 s + 0.26 s | ~1.9 s | 96 s (measured), 82.7 GB resident |

For the MOPD checkpoint specifically, the boot of 2026-09-28 measured **115.5 s / 82.92 GB on
the head** and **229.3 s / 83.42 GB on the worker over NFS** — the worker is slower because it
reads the same shards across the fabric instead of from local NVMe. Do not compare a head
figure with a worker figure without saying which is which.

The CUDA pageable copy faults the mmap'd file pages 4 KiB at a time, about 15 ms per 4 MB
tensor. A `clone()` reads the same tensor through the kernel's readahead at 2 GiB/s and the
copy then runs at 9 GiB/s. `read()` gets the speed too, but every in-flight shard becomes
anonymous memory twice over (the bytes plus the parsed copies): 25 GiB for `ep0`, and ~7 GiB
per rank stayed resident after the load. `patches/sitecustomize.py` therefore keeps the
default mmap loader and clones one tensor at a time on the way to the GPU, skipping routed
experts that belong to the other EP rank (the model skips those without touching the data, so
each rank reads its ~83 GiB and not the full 166). The same hook makes each EAGLE draft pass
read only `model_mtp.safetensors` instead of every shard. `WEIGHT_LOADER_DROP_CACHE=1`
releases each shard's page cache once it is consumed. `fastsafetensors` is not in the image; a
saved per-rank `sharded_state` would be faster still but needs 83 GiB of free disk on every
worker.

`MEM_GUARD_GIB` is the userspace OOM guard: `boot.py` kills SGLang when `MemAvailable` drops
under that many GiB, which ends in an exited container instead of a rebooted Spark — see the
deviation note for why this profile uses 2.

## CUDA graphs

Capture is the big decode lever: with speculation the engine runs several model passes per
step across 48 layers, and uncaptured that is thousands of eager launches per token — which is
why an earlier EAGLE measurement on this fleet was only **19–28 tok/s** with
`cuda graph: False` on every `Decode batch` line. **That was a different checkpoint and is not
the number this README publishes**; the measured MOPD decode ladder is in
[`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md) §8.

`CUDA_GRAPH_MAX_BS_DECODE=8` **failed under EAGLE + `--enable-multi-layer-eagle`** and is
being retried under DFlash, which uses a different draft runner (`dflash_worker_v2`) and which
the upstream cookbook does run decode graphs with. The EAGLE MTP failure below is kept because
it is a real incompatibility, not a memory problem, and because the diagnosis generalises.

`CUDA_GRAPH_PREFILL` stays 0 either way — that also matches upstream, which disables prefill
CUDA graphs.

### The EAGLE MTP failure

The **target verify** graph captures fine:

```text
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4,
  bs=[1, 2, 3, 4, 5, 6, 7, 8], avail mem=19.67 GB
Capture target verify CUDA graph end. elapsed=4.35 s, mem usage=0.76 GB, avail mem=18.90 GB
```

The **EAGLE draft extend** graph then dies, with 18.10 GB still free — this is not a memory
failure:

```text
SGLANG_ENABLE_SINGLE_CG_DRAFT is on but TritonAttnBackend cannot refresh its
  draft-extend metadata for a combined graph; falling back to per-step draft graphs
Capture draft extend CUDA graph begin. step=2, avail mem=18.10 GB
  → multi_layer_eagle_draft_extend_cuda_graph_runner.capture_one_shape
  → maybe_flashinfer_autotune_speculative_draft → mimo_v2_nextn.forward
  → triton.compiler.errors.CompilationError at offs_kv_loc = tl.load(kv_indices + ...)
    AttributeError("'NoneType' object has no attribute 'type'")
```

`ATTENTION_BACKEND=triton` and multi-layer-EAGLE draft-extend capture do not compose in this
engine: the Triton extend kernel is compiled against a dummy batch whose `kv_indices` is
`None`. Two untested candidates, in order of preference, both one flag:

1. `--speculative-draft-attention-backend flashinfer` — takes the draft off the Triton extend
   kernel and leaves the target and the MoE autotune alone (`flashinfer`, `fa3`, `fa4`,
   `triton` are the draft choices).
2. `--disable-flashinfer-autotune` — skips exactly the call that raised
   (`flashinfer_autotune.py` returns early on this flag), at the cost of MoE autotuning
   everywhere.

Do not re-enable capture without one of them, and expect to need a boot per candidate.

**What the attempt did establish.** Capture is cheap (0.76 GB for 8 tiers) and comes out of
the slack kept *outside* the pool, not out of the KV pool — the same reserve the prefill
transient uses, which is why a smaller chunk size helps it. And the KV pin is silently
clamped, as suspected: asking for more tokens than the budget allows does not raise the pool,
it just produces a `max_total_tokens=… is larger than the profiled value …` line and the
profiled allocation. Read `Capture cuda graph end … mem usage=` and the pool line from the
boot log before raising either.

Two things to check on the first graph-enabled boot:

- **Acceptance.** Per-position acceptance pinned at exactly 1.00 at pos0 is a known
  CUDA-graph bug, not a win; healthy is a monotone decay. Compare `accept len` against the
  2.5–4.0 medians in [`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md) §8.
- **The KV pin.** The boot log's `max_total_num_tokens=` line is the authority, not the
  profile value.

The server always starts with `--enable-multimodal`: this is the full omni checkpoint. The
image installs `ffmpeg` and refuses to build unless `torchcodec` and `torchaudio` import,
because audio goes through torchcodec and video needs `ffprobe`. Workers must see
`preprocessor_config.json`, `audio_tokenizer/model.safetensors`, and `model_mtp.safetensors`
on the head export, not only the language shards.

Vision attention on GB10 is `triton_attn` (`MM_ATTENTION_BACKEND`).

## KV geometry

Why the context limit is not automatically the context you get. `config.json`'s
`hybrid_layer_pattern` gives **9 full-attention layers and 39 sliding-window layers**
(window 128) out of 48. Per token per rank at TP=2 with fp8 KV:

| | layers | KV heads/rank | bytes/token |
|---|---:|---:|---:|
| full attention | 9 | 2 | 5,760 |
| sliding window | 39 | 4 | **49,920** |

An SWA token costs **8.7×** a full-attention token, because it is 39 layers at 4 KV heads
against 9 at 2. `MAX_TOTAL_TOKENS` is the pin in *full-attention* tokens and the SWA pool is
derived as `SWA_FULL_TOKENS_RATIO ×` that, so the ratio dominates the bill:

| `SWA_FULL_TOKENS_RATIO` | bytes per pin token | KV for a full 1M context |
|---|---:|---:|
| 0.30 (this kit until 2026-09-22) | 20,736 | 20.25 GiB |
| **0.03** (upstream Flash recipe) | **7,258** | **7.09 GiB** |

At 0.3 the pool clamped well under a 1,048,576 context, so a full-context request could not be
served. 0.03 is what the cookbook ships and brings a genuine 1M context inside the measured
budget.

### The pool, as measured

For **this** profile (MOPD checkpoint, `MEM_FRACTION_STATIC=0.93`, EAGLE MTP, 8192-token
chunks), from the boot log of 2026-09-28 09:56:45Z and confirmed live by `/v1/loads`:

```text
Use sliding window memory pool. full_layer_tokens=2546496, swa_layer_tokens=76352
max_total_num_tokens=2546496, chunked_prefill_size=8192, max_prefill_tokens=16384,
max_running_requests=8, context_len=1048576, available_gpu_mem=9.15 GB
```

`/v1/loads` reports the live split and needs no extra flags:

```text
weight_gb 82.918   kv_cache_gb 17.214   graph_gb 0.0   token_capacity 2546496
```

**2,546,496 tokens is 2.43× a full 1M context**, so several 1M-context requests can be
resident at once, limited by `MAX_RUNNING_REQUESTS=8`, not by KV.

Earlier boots of this fleet reported other pool sizes (2,533,952; 2,592,384; 2,878,144). They
belong to configurations before this one — a different checkpoint, `chunked_prefill_size=2048`
and/or a different memory fraction — and must not be quoted for this profile. **2,546,496** is
the value with a boot log and a live `/v1/loads` behind it.

**The pin is not the gate.** SGLang clamps `MAX_TOTAL_TOKENS` to whatever is left after the
weights inside `MEM_FRACTION_STATIC`, silently. Read `max_total_num_tokens=` from the boot
log; the env value is a ceiling, not a promise. `boot.py`'s `preflight:` line consistently
over-states the budget by roughly 2× against what the engine then allocates, so trust the
engine's line.

The drafter changes the accounting: EAGLE's three MTP draft KV pools are ~0.1 GB each, while
DFlash keeps a second pool sized to the *same* `max_total_num_tokens` (7.94 GB at a
1,653,056-token pool, measured on this fleet in an earlier profile). At a fixed memory
fraction that is roughly half the pool EAGLE would give you — which is why the shipped
`MAX_TOTAL_TOKENS` and the DFlash recipe in "Booting each" differ.

## Speculative decoding

The checkpoint ships **two** drafters — an **EAGLE MTP** head and a **DFlash draft model** —
and `SPEC_ALGO` picks between them. They are different things: EAGLE MTP drafts from the
target's own hidden states through the bundled 3-layer MTP head, while DFlash is a separate
5-layer Qwen3-architecture model that drafts a whole block at once. **Which is faster depends
on what the model is writing**, because the mechanism is acceptance, not raw speed: a drafter
proposes `SPEC_DRAFT_TOKENS` tokens per step and the target verifies them, and rejected drafts
are wasted compute.

| | drafts through | weights | verify window | measured here |
|---|---|---|---|---|
| `EAGLE`<br>*EAGLE MTP* (**default**) | the target's hidden states (3-layer MTP head) | `model_mtp.safetensors`, 1.2 GiB | 4 | median accepted length **4.00** on counting/math, 3.2–3.9 on structured/json/code, **2.50–2.75** on prose. Requires `CUDA_GRAPH_MAX_BS_DECODE=0` |
| `DFLASH`<br>*DFlash draft model* | a separate 5-layer Qwen3 model, whole block at once | `dflash/`, 2.9 GiB | 8 (its trained `block_size`) | **not benchmarked in this repository**; it keeps a second KV pool sized to the pin, so expect a much smaller pool at the same memory fraction |
| `off` | — | — | — | a clean baseline |

The category split is the point: a narrower window that lands beats a wider one that does not,
and a blended single number hides which drafter suits a workload. That is why
`bench/decode.py` reports acceptance per category — see
[`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md) §8 for the full 6 × 4 matrices.

There is **no per-request switch** and no dual-drafter mode: the draft model is loaded once at
startup (`SamplingParams` carries nothing speculative), so picking one is a boot-time
decision. Switching is one line and one boot.

### Booting each

**EAGLE MTP** — the shipped default (prose, chat, long-form).

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=EAGLE/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=4/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=0/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2743808/' .env
./start.sh restart
```

> ### ⚠ `CUDA_GRAPH_MAX_BS_DECODE=0` is NOT optional under EAGLE MTP
>
> It is the **third** variable in the command above, and leaving it at 8 does not degrade
> performance — **it fails the boot outright**. Decode graph capture dies in
> `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`: the Triton extend-attention kernel
> cannot compile against a dummy batch whose `kv_indices` is `None`, the scheduler raises, and
> the server never becomes healthy. It is not a memory failure — 18.10 GB was free when it
> happened. Full diagnosis and the two candidate fixes are in
> [The EAGLE MTP failure](#the-eagle-mtp-failure). **Switching to EAGLE means giving up decode
> CUDA graphs**, which partly offsets its advantage on prose.

**DFlash draft model** — code, structured output, agent work. Decode graphs work here.

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=DFLASH/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=8/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=8/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2299968/' .env
./start.sh restart
```

**No speculation**, for a clean baseline:

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=off/' .env && ./start.sh restart
```

Expect `cuda graph: False` on every `Decode batch` line under EAGLE — that is correct for
EAGLE, not a fault. `accept len` is out of 4 under EAGLE and out of 8 under DFlash. Confirm
which one came up, and how well it is drafting, from the engine log:

```bash
grep -o "'speculative_algorithm': '[A-Z]*'" logs/mimo26.log | tail -1
grep -E "Decode batch" logs/mimo26.log | tail -3   # accept len: N of the window
```

DFlash-specific notes:

- SGLang carries DFLASH natively (`srt/models/dflash.py`,
  `srt/speculative/dflash_worker_v2.py`) and `mimo_v2.py` already implements the
  aux-hidden-state capture the drafter needs (`AuxHiddenStatePacker`, `layers_to_capture`), so
  no target patch is required. The drafter's `dflash_config.target_layer_ids` is
  `[0, 11, 23, 35, 47]` against `num_target_layers: 48`, which matches this checkpoint.
- DFLASH costs **+1.37 GiB per rank** over EAGLE (82.79 vs 81.42 GiB resident, measured on an
  earlier profile); `boot.py` counts whichever drafter `SPEC_ALGO` will actually load, so the
  preflight is not under-estimating the rank by the difference.
- DFLASH forces `speculative_num_steps` and `speculative_eagle_topk` to 1 itself and warns if
  they are set otherwise, so `boot.py` emits neither under DFLASH; `SPEC_STEPS` / `SPEC_TOPK` /
  `ENABLE_MTP` are read only by EAGLE.
- `SPEC_DRAFT_TOKENS` becomes `--speculative-num-draft-tokens`, which SGLang documents
  `--speculative-dflash-block-size` as an alias of for DFLASH.
- The published `dflash/config.json` has a trailing comma before its closing brace, so it is
  not valid JSON and transformers refuses it. The checkpoint is mounted read-only and shared
  to the worker over NFS, so `boot.py` does not edit it: under `SPEC_ALGO=DFLASH` it writes a
  re-serialized `config.json` into `/state/dflash` and symlinks the weights beside it, then
  points `--speculative-draft-model-path` there. If the trailing-comma repair does not yield
  valid JSON it refuses rather than guessing.

## Performance — prefill (measured here)

Cold-prefill ladder, 2026-09-28 10:08:22Z → 10:11:07Z, exit 0, 5/5 rungs. Speculation does
not touch prefill, so these hold for either drafter.

```bash
BENCH_BASE_URL=http://127.0.0.1:8888 python3 bench/cold_prefill.py --out-name mopd-cold-prefill
```

| context | prompt tokens | TTFT | prefill tok/s |
|---|---:|---:|---:|
| 8k | 8,010 | 3.66 s | 2,189 |
| 16k | 16,010 | 6.93 s | **2,311** |
| 32k | 32,010 | 14.85 s | 2,155 |
| 64k | 64,010 | 36.57 s | 1,750 |
| 128k | 128,010 | 100.56 s | 1,273 |

Plus a prefix-cache follow-up on the first rung: same 8k prompt, `cached=8000`, **TTFT
0.34 s** — 10.8× faster than its cold twin, which is the radix cache working.

The curve has two regimes, and both are expected:

- **Short prompts amortize badly.** Fixed per-request cost — tokenize, template, prefix-cache
  hash, queue, first-chunk ramp — is a larger share of a 8k request than of a 16k one, which
  is why the peak is at 16k rather than at the bottom of the ladder.
- **Past 16k it decays with length**, to 55% of peak at 128k. Each 8192-token chunk attends
  over the whole prefix so far, so per-chunk cost grows as the prompt does. A 128k prompt is
  16 chunks, each more expensive than the last.

This ladder is also the evidence for the `CHUNKED_PREFILL_SIZE=8192` deviation: it ran under
that setting, with no `MEMORY GUARD` line, on the profile in `.env.example`.

Reference point, *not* a controlled comparison: a 2×-Spark vLLM recipe for the same checkpoint
family reports 1,697.9 tok/s at 7.9K and 656.4 at 248K. Different engine, different MoE runner
and a 300K max-model-len against 1M — but the same model on the same two boxes.

```bash
# image
curl http://<HEAD_IP>:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-v2.6-Flash","max_tokens":128,"temperature":1.0,"top_p":0.95,
       "messages":[{"role":"user","content":[
         {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}},
         {"type":"text","text":"What is in this image?"}]}]}'

# video (ffprobe) and audio (torchcodec) use the same content list
# with type video_url / audio_url
```

## Performance — decode (measured here)

Decode ladder, 2026-09-28 10:12:10Z → 10:16:19Z, 24/24 cells OK, EAGLE MTP, 256 pinned tokens
per stream.

```bash
BENCH_BASE_URL=http://127.0.0.1:8888 HEAD_CTN=mimo26-tp2-head \
  python3 bench/decode.py --out-name mopd-decode
```

Aggregate throughput (tokens/s across all streams in the cell):

| category | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| counting | 46.8 | 77.7 | 120.0 | **226.0** |
| structured | 37.8 | 61.3 | 90.3 | 148.9 |
| json | 42.4 | 62.8 | 80.9 | 122.1 |
| code | 35.3 | 53.4 | 87.1 | 128.6 |
| math | 44.1 | 67.8 | 118.0 | 201.8 |
| prose | 31.8 | 45.3 | 64.2 | 98.3 |

Median per-stream decode rate (tokens/s, TTFT excluded): prose 38.6 → 15.0, counting 63.4 →
40.0 as concurrency goes 1 → 8. Median TTFT stays at 1.4–3.3 s in every cell (the prompts are
short). Median accepted draft tokens per step (of a possible 4): counting 4.00, math 3.50–4.00,
json/code/structured 3.17–3.95, prose 2.50–2.75.

Conclusions:

- Decode scales **4.8× at C8 for counting** and **3.1× for prose**; per-stream rate falls as
  batch grows, as expected on a memory-bound MoE.
- EAGLE MTP is doing real work, and the category split is exactly what the harness exists to
  expose. `bench/decode.py` exits non-zero on some runs because of a substring heuristic in
  `bench/common.py` (`nan_free` tests for the literal letters "nan", which ordinary prose can
  contain); the receipt still shows 0 failed cells and 0 length mismatches. See
  [`docs/TEST-RESULTS.md`](docs/TEST-RESULTS.md) §8 for the reproduced trigger.
- The DFlash arm is **not** measured here. Run the same ladder once per drafter with the same
  category set before choosing.

## SSD offload

In `.env`:

```bash
OFFLOAD_MODE=ssd
SSD_DIR=$HOME/mimo26-ssd-tp2          # head NVMe
WORKER_SSD_DIR=$HOME/mimo26-tp2/ssd   # worker NVMe
HICACHE_SIZE_GB=4                     # unified-memory staging, per rank
```

Each rank gets its own subdirectory so file-backend buckets do not collide. This does not move
expert weights off the GPU. It only spills hierarchical KV to disk. Leave it `off` until a
long-context run actually needs the extra tier. `OFFLOAD_MODE=ram` and `cpu` are refused: host
RAM is GPU memory on a Spark.

## Kernels

GB10 defaults: Triton attention, Triton multimodal attention, FlashInfer MXFP4 MoE, FP8 E4M3
KV. Override with `ATTENTION_BACKEND`, `MM_ATTENTION_BACKEND`, `MOE_RUNNER_BACKEND`,
`KV_CACHE_DTYPE`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is set because
expandable segments produced NaN logits on this same GB10 stack with another SGLang model.

## Benchmarks

Ported from an earlier Spark kit's cold-prefill harness and adapted for SGLang. **Run them
from a worker, never the head** — the head runs the HTTP server, tokenizer, detokenizer and
NFS export, and is the rank the other waits for. (The results in this README were collected
from the head against `127.0.0.1:8888`, the only unproxied endpoint here; treat those absolute
numbers as head-local.)

```bash
BENCH_BASE_URL=http://<HEAD_IP>:8888 python3 bench/cold_prefill.py
BENCH_BASE_URL=http://<HEAD_IP>:8888 python3 bench/decode.py --concurrency 1,2,4,8
```

Receipts land in `bench/results/` (gitignored). Both refuse to run against a busy server
(`/v1/loads`), and both take `--out-name` so an A/B writes two labelled receipts.

What the protocol buys, and why these details are not optional:

- **`cold_prefill.py`** puts a unique salt on every request, so the radix cache cannot serve
  it. Without that a "cold" ladder silently measures prefix-cache hits. Prompt length is
  verified through `/tokenize` to within 0.5% rather than assumed, and TTFT is the first
  *content* token. It also runs one prefix-cache follow-up row, which should be far faster
  than its cold twin.
- **`decode.py`** pins output length with `min_tokens` + `ignore_eos`, because a run that
  stops early is not comparable to one that does not. The decode rate excludes TTFT —
  `(emitted - 1) / (last - first content token)`. Counting SSE chunks would under-report
  badly, since one chunk can carry a whole accepted speculative block.
- **Acceptance is per category, and that is the point.** This checkpoint ships two drafters
  that fail differently — acceptance is high on structured and code output, low on prose — so
  a blended number hides which drafter suits a workload. Acceptance is not on the OpenAI path,
  so it is read from the engine's `Decode batch` lines for each arm's window.

`ENABLE_METRICS=1` and `ENABLE_CACHE_REPORT=1` (both default on) give `/metrics` and
`prompt_tokens_details.cached_tokens`; the harnesses degrade to "unknown" rather than failing
when they are off.

## What is in the box

```text
start.sh              doctor / pull / build / download / share / serve / restart / stop / status / logs / smoke
stop.sh               teardown; waits for the driver to release the GPU on every node
boot.py               in-container entrypoint: engine + memory preflight, arg planner,
                      DFlash config repair, userspace OOM guard, smoke
Dockerfile            FROM $BASE_IMAGE + ffmpeg, torchcodec/torchaudio import gate
.env.example          the profile; copied to .env on first run
download-ms.sh        ModelScope download + strict integrity gate
mimo26.sh             operational wrapper: the GB10 pre-launch cache ritual + start/stop/logs
head-pull-base.sh     pull the pinned base image through domestic mirrors
DEPLOY-NOTES.md       this deployment's record, in Chinese
docs/
  TEST-RESULTS.md     every measured number, with the command and the UTC time
  RUNTIME-ARGS.md     the launch arguments of the boot these results come from
patches/
  sitecustomize.py    import hooks: per-tensor weight materialization and
                      MTP fused-qkv resharding
files/                NFSv4 export helpers (incl. the exporter image), NCCL setup
scripts/
  remote.py           ssh helper for the worker
  check-serving.sh    read-only post-boot verification; the image smoke asserts the
                      left/right colour split and exits non-zero when it fails
  verify-weights.py   checkpoint vs ModelScope metadata: sizes, sha256, shard count
tests/                arg planner + memory preflight, stop/restart guard rails,
                      in-image patch checks
```

## Tests

```bash
python3 -m unittest tests.test_boot_args -v     # arg planner + memory preflight, no torch needed
bash tests/test_stop_restart.sh                 # stop/restart guard rails; safe against a live fleet
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5 tests/test_engine_patches.py   # MTP qkv patch, CPU
```

Measured loads at TP=2 on this fleet (`Load weight end` lines):

| loader | target model | resident | each EAGLE draft |
|---|---:|---:|---:|
| default mmap (first boot, worker) | 525 s | 82.74 GB | crashed on the fused qkv |
| `--weight-loader-disable-mmap` (second boot) | 131 s | 89.79 GB | crashed the nodes re-reading 65 shards |
| mmap + per-tensor `clone()` patch (head, MOPD checkpoint) | 115.5 s | 82.92 GB | 0.55–0.87 s, 1 file per draft |

## License & attribution

This repository is a derivative work of
[MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks)
(AGPL-3.0), adapted for the MiMo-V2.6-Flash-MOPD checkpoint; modified files carry a notice
stating the change and its date. Code: **AGPL-3.0** (see [LICENSE](LICENSE)) — if you run a
modified version as a network service, the AGPL requires you to offer its source to the users
of that service. Model weights: **MIT**, Xiaomi's, not redistributed here — `download-ms.sh`
fetches them from ModelScope.

Upstream work Copyright (C) 2026 Mia'a AI Lab, licensed AGPL-3.0. This derivative work is
Copyright (C) 2026 its author, modified 2026-09-28.
