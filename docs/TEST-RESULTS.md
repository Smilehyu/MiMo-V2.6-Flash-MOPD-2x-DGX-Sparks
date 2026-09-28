# Test results — MiMo-V2.6-Flash-MOPD on 2× DGX Spark (TP2/EP2, SGLang)

Reproducible results for **this** deployment, measured on the MOPD checkpoint. Every row carries
**the command**, **the raw output**, and **the UTC collection time**, and is labelled either:

- **[measured]** — executed for this report against the live service, or read back from the
  running engine's own API/logs during this window;
- **[quoted]** — read back from an existing log of *this same boot* (nothing was restarted,
  re-served or reconfigured to produce these numbers).

| | |
|---|---|
| Model | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1` (ModelScope) |
| Served name | `MiMo-v2.6-Flash` |
| Topology | 2× NVIDIA GB10 (DGX Spark), `--tp 2 --ep 2 --nnodes 2`, weights shared head→worker over NFSv4 |
| Engine | SGLang `0.0.0.dev1+g0f6761b54`, image `mimo26-spark:local` (base `lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5`) |
| Endpoint under test | `http://127.0.0.1:8888` **on the head** (loopback API of rank 0). No proxy or second control plane was involved. |
| Collection window | 2026-09-28 10:06Z – 10:25Z (UTC) |
| Container age at collection | started 2026-09-28 09:52:04Z, i.e. the MOPD boot; no restart happened during collection |
| Profile under test | `CHUNKED_PREFILL_SIZE=8192`, `MEM_GUARD_GIB=2`, `MEM_FRACTION_STATIC=0.93`, `SPEC_ALGO=EAGLE` (see `.env.example` and README "Deviations from upstream") |

Placeholders used below: `<HEAD_HOST>` / `<WORKER_HOST>` are the two nodes, `<HEAD_IP>` /
`<WORKER_IP>` their fabric addresses, `$KIT` is this repository's directory on the head
(`$HOME/…`, wherever you cloned it), and `$SCRATCH` is any scratch directory on the head
(the recorded one-off runners lived in one; nothing they wrote is published). The raw captures
behind these tables are **not** published (the publishing rules forbid shipping logs, state and
receipts); every command below runs against your own deployment. Two rows are exceptions and are
labelled where they occur: §4's worker-side check, which was read from this boot's own log, and
§6's real-image rows, which were produced by a one-off runner rather than by a script this
repository ships.

## Test method (what a reader needs to reproduce or distrust)

- **Idle precondition.** `bench/cold_prefill.py` and `bench/decode.py` both refuse to run
  unless `/v1/loads` reports `num_running_reqs=0, num_waiting_reqs=0`. The checks passed in
  both runs (output quoted below), so no other traffic is inside these numbers.
- **Cache cannot serve the "cold" rungs.** The cold ladder puts a fresh UUID salt +
  48 hex chars of padding in every request, and the decode ladder uses a per-stream nonce,
  so the radix cache cannot answer a rung from a previous one.
- **Pinned decode length.** `min_tokens == max_tokens == 256` with `ignore_eos`, so every
  stream emits exactly 256 tokens; a run that stopped early would not be comparable.
- **Decode rate excludes TTFT**: `(emitted - 1) / (last - first content token)`. Counting SSE
  chunks instead under-reports by the accepted speculative block.
- **Warm-up.** `decode.py` does its own throwaway warm-up; for the cold ladder a single
  throwaway chat completion was issued first, because this deployment sets
  `--sleep-on-idle`, and rung 1 must not pay the engine's wake-up.
- **Runs were issued from the head against the loopback endpoint.** The bench docstrings
  recommend issuing from a worker (the head also hosts the HTTP front end, the
  tokenizer/detokenizer and the NFS export). The loopback endpoint is the only unproxied
  one here, so that is what was measured — treat the absolute numbers as head-local.
- **Read-only.** No service start/stop/restart, no `.env` or kit edits, no container or NFS
  changes. Only inference calls, `docker logs`/`inspect`, `stat`, `curl` GETs and `sha256sum`.

---

## 1. Checkpoint integrity — full listing + sampled sha256 **[measured]**

Collected 2026-09-28T10:07:37Z → 10:08:02Z (whole-checkpoint run, wall clock 25.42 s) and
10:21:53Z → 10:23:24Z (per-file sha256 table).

```bash
# whole-checkpoint check: every upstream file byte-exact, sha256 for every file
# < 256 MiB plus the sampled shards, and the expert-shard count
ssh <HEAD_HOST> 'MODEL_DIR=$HOME/NewModels/MiMo-V2.6-Flash-MOPD python3 $KIT/scripts/verify-weights.py'

# per-file sha256 table for the nine files named below (a one-off wrapper, not shipped
# with this repository, that calls the ModelScope repo/files API and compares one file
# at a time)
ssh <HEAD_HOST> 'python3 $SCRATCH/mopd-bench/verify-mopd-files.py'
```

Raw output, `verify-weights.py` (2026-09-28T10:07:37Z, `wall_clock=25.42s`, exit 0):

```text
upstream files: 155
expected total: 177765031619 bytes (165.56 GiB)
missing files       : 0 []
size mismatches     : 0 []
sha256 verified     : 29 files, mismatches: 0 []
expert shards       : upstream 128 / on disk 128
VERIFY_RESULT: PASS
```

Raw output, per-file table (2026-09-28T10:21:53Z → 10:22:27Z, exit 0) — every row `MATCH`
means local bytes == upstream bytes **and** local sha256 == ModelScope sha256:

| file | bytes | local sha256 == ModelScope sha256 |
|---|---:|---|
| `config.json` | 8,068 | `61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e62` |
| `model.safetensors.index.json` | **6,921,726** | `e16ec5b736c59c8032af31004f3b99e31560eee70bd24c368c843a4d63f93b67` |
| `model_mtp.safetensors` | 1,189,405,960 | `f8f78a78a79e4ec9e42e02ff94790d2a0183995ae278ff521416baa25ae2f38c` |
| `dflash/dflash_draft_model.safetensors` | 2,936,121,080 | `94d9c02c17e0b469f88699e98424825ebe13a17f07eb3b86727b5e0c8143d287` |
| `audio_tokenizer/model.safetensors` | 1,872,618,384 | `077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265` |
| `model_pp0_ep0_shard0.safetensors` | 12,130,902,720 | `9a6083ea6ccf25854158fb6de082c4990642d62ef70e6871903b7be34fbd17b3` |
| `model_pp0_ep1_shard0.safetensors` | 1,256,787,552 | `65bb19cf7cf2f768e8ce29230c8c5d26efba9670bb32104287e96e6dac30a828` |
| `model_pp0_ep64_shard0.safetensors` | 1,256,788,680 | `945a9cf9d74d4f0dd899074c2241f7a49a484c1405ccc46a2331b8ffe09b2a4b` |
| `model_pp0_ep127_shard0.safetensors` | 1,256,788,680 | `fb574517f63c2232593385f1bd2c75e88cc948c56e5fb8b2cadc7578f848f2b9` |

Tail of the same output:

```text
upstream listing: 158 files @ revision 24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1
files checked: 9, not MATCH: 0
RESULT: PASS
```

**Conclusions**

- `model.safetensors.index.json` is 6,921,726 bytes — the exact byte count the worker prints
  at boot when it checks the checkpoint over NFS (`ok index=6921726 shards=128`, §4).
- No file is missing and no size differs from the pinned upstream revision.
- 29 files were sha256-verified, including the four requested shard samples
  (`ep0`, `ep1`, `ep64`, `ep127`), the MTP head, the DFlash drafter and the audio tokenizer.
- Listing-count note: the raw ModelScope listing returns **158** entries;
  `scripts/verify-weights.py` filters to **155** blobs (`Type == "blob"` or a truthy size),
  which is why the two counts differ. Neither is a mismatch.

---

## 2. Served configuration and memory state **[measured]**

Collected 2026-09-28T10:11:44Z.

```bash
ssh <HEAD_HOST> 'curl -s http://127.0.0.1:8888/v1/models'
ssh <HEAD_HOST> 'curl -s http://127.0.0.1:8888/v1/loads'
ssh <HEAD_HOST> 'curl -s http://127.0.0.1:8888/server_info'   # -> launch_command, startup_time
```

`GET /v1/models`:

```json
{"object":"list","data":[{"id":"MiMo-v2.6-Flash","object":"model","created":1790590304,
 "owned_by":"sglang","root":"MiMo-v2.6-Flash","parent":null,"max_model_len":1048576}]}
```

`GET /v1/loads` (fields that matter; the full line was captured in this run's raw log):

```json
"accelerator": "NVIDIA GB10", "num_accelerators": 2,
"loads": [{ "num_running_reqs": 0, "num_waiting_reqs": 0,
  "max_total_num_tokens": 2546496, "max_running_requests": 8,
  "memory": {"weight_gb": 82.918, "kv_cache_gb": 17.214, "graph_gb": 0.0,
             "token_capacity": 2546496},
  "speculative": {"accept_length": 1.275, "accept_rate": 0.0} }]
```

Host/GPU memory on the head at the same moment:

```text
$ nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
name, memory.used [MiB], memory.total [MiB]
NVIDIA GB10, [N/A], [N/A]          # GB10 is unified memory: nvidia-smi cannot split it

$ free -m | head -2
               total        used        free      shared  buff/cache   available
Mem:          124610      118423        1615         414        6067        6186
```

Engine's own pool accounting, from the boot log **[quoted, 2026-09-28 09:56:45Z]**:

```text
Use sliding window memory pool. full_layer_tokens=2546496, swa_layer_tokens=76352
max_total_num_tokens=2546496, chunked_prefill_size=8192, max_prefill_tokens=16384,
max_running_requests=8, context_len=1048576, available_gpu_mem=9.15 GB
```

Serving parameters, from `/server_info.launch_command` (identical to `state/launch.json`,
which is gitignored and not published; the same list is written out in `docs/RUNTIME-ARGS.md`):

| parameter | value |
|---|---|
| `--context-length` | 1,048,576 (`max_req_input_len` 1,048,570) |
| `--mem-fraction-static` | 0.93 |
| `--max-total-tokens` | 2743808 requested → **2546496 allocated** (profiled value) |
| `--max-running-requests` | 8 |
| `--chunked-prefill-size` / `max_prefill_tokens` | 8192 / 16384 |
| `--page-size` / `--swa-full-tokens-ratio` | 64 / 0.03 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--speculative-algorithm` | EAGLE, `--speculative-num-steps 3`, `--speculative-eagle-topk 1`, `--speculative-num-draft-tokens 4`, `--enable-multi-layer-eagle` |
| attention / MoE backends | `--attention-backend triton`, `--mm-attention-backend triton_attn`, `--moe-runner-backend flashinfer_mxfp4`, `--moe-a2a-backend none`, `--moe-dense-tp-size 1` |
| other | `--disable-cuda-graph`, `--sleep-on-idle`, `--enable-metrics`, `--enable-cache-report`, `--weight-loader-drop-cache-after-load`, `--watchdog-timeout 1800` |

**Conclusions**

- The KV/context configuration is exactly the intended one, and the engine reports the pool
  as **2,546,496 tokens** — 2.43× the 1M context, i.e. the full context fits, with the SWA
  side pool at 76,352 tokens (SWA costs 8.7× a full-attention token, see the README's
  "KV geometry").
- Memory left outside the pool at steady state: **`available_gpu_mem=9.15 GB`** (engine
  accounting) and **6,186 MiB** host-available (kernel view) — both after weights (82.9 GB)
  and KV (17.2 GB) are resident.
- `--max-total-tokens 2743808` was not honoured verbatim; the engine logged why (§3) and
  allocated its profiled value instead. Quote both numbers if this is ever compared.
- **Pool-size cross-check for anyone quoting an older note:** `2,533,952` (and `2,592,384`)
  appear in earlier boot logs of this fleet — one at `logs/relaunch.log:339`, dated
  **2026-09-27 12:03:53Z**, i.e. before the MOPD switch and before `chunked_prefill_size`
  became 8192. The value that belongs to the configuration described here, and the one to
  publish, is **2,546,496** (boot of 2026-09-28 09:56:45Z, confirmed live by `/v1/loads`).

---

## 3. Cold start, weight load and the MTP drafters **[measured + quoted]**

Collected 2026-09-28T10:11:44Z from the running container (no restart performed).

```bash
ssh <HEAD_HOST> "docker inspect -f 'created={{.Created}} started={{.State.StartedAt}}' mimo26-tp2-head"
ssh <HEAD_HOST> "docker logs mimo26-tp2-head 2>&1 | grep -E 'Load weight (begin|end)|Use sliding window|max_total_num_tokens=|fired up'"
ssh <HEAD_HOST> "ssh <WORKER_HOST> \"docker logs mimo26-tp2-worker 2>&1 | grep -E 'Load weight (begin|end)'\""
```

```text
created=2026-09-28T09:52:04.37281706Z   started=2026-09-28T09:52:04.555677003Z   image=mimo26-spark:local

# head (rank 0, TP0 EP0)
[2026-09-28 09:52:45] Load weight begin. avail mem=113.08 GB
[2026-09-28 09:54:40] Load weight end. elapsed=115.51 s, type=MiMoV2ForCausalLM, quant=fp8, fmt=e4m3, avail mem=30.16 GB, mem usage=82.92 GB.
[2026-09-28 09:56:35] Load weight end. elapsed=0.55 s, type=MiMoV2MTP, ... mem usage=1.96 GB.
[2026-09-28 09:56:36] Load weight end. elapsed=0.62 s, type=MiMoV2MTP, ... mem usage=1.47 GB.
[2026-09-28 09:56:37] Load weight end. elapsed=0.87 s, type=MiMoV2MTP, ... mem usage=1.50 GB.
[2026-09-28 09:56:49] The server is fired up and ready to roll!

# worker (rank 1, TP1 EP1), read over SSH from the head
[2026-09-28 09:52:45] Load weight begin. avail mem=114.00 GB
[2026-09-28 09:56:34] Load weight end. elapsed=229.34 s, type=MiMoV2ForCausalLM, quant=fp8, fmt=e4m3, avail mem=30.58 GB, mem usage=83.42 GB.
[2026-09-28 09:56:35] Load weight end. elapsed=0.68 s, type=MiMoV2MTP, ... mem usage=1.95 GB.
[2026-09-28 09:56:36] Load weight end. elapsed=0.62 s, type=MiMoV2MTP, ... mem usage=1.51 GB.
[2026-09-28 09:56:37] Load weight end. elapsed=1.00 s, type=MiMoV2MTP, ... mem usage=1.48 GB.
```

Engine-side timers from `/server_info.startup_time` (seconds):

```text
load_weight          117.55      # = 115.51 + 0.55 + 0.62 + 0.87: main weights plus the three MTP heads
kv_cache_allocation    2.62
scheduler_e2e        245.37
tokenizer_e2e        250.49
cuda_graph         {prefill: 0.0, decode: 0.0, target_verify: 0.0, draft_*: 0.0}   # --disable-cuda-graph
```

`--max-total-tokens` being overridden (same boot log) **[quoted]**:

```text
[2026-09-28 09:56:37] max_total_tokens=2743808 is larger than the profiled value 2546496. Use the profiled value instead.
```

**Conclusions**

- **Cold start (container start → serving): 284 s** — `StartedAt 09:52:04.6Z` → `fired up
  09:56:49Z`, i.e. 4 m 44 s. The rank-0 weight load is 115.5 s (82.92 GB resident).
- Rank 1 loads the same 83.42 GB **over NFS from the head** and takes **229.34 s** — about
  2× rank 0, and it finishes at 09:56:34, so the worker's load, not the head's, fills most of
  the boot. `startup_time.load_weight` (117.55 s) is the head's own sum
  (115.51 + 0.55 + 0.62 + 0.87), not the cluster-wide time — do not quote it as such.
- The three EAGLE MTP drafters cost 0.55 / 0.62 / 0.87 s (head) and add 4.93 GB total.
- Engine-reported `scheduler_e2e` 245.4 s and `tokenizer_e2e` 250.5 s are **not** the same
  clock as the container-level 284 s (they are measured inside the engine process and exclude
  parts of the boot the container clock includes). Quote whichever metric you need, but do not
  mix them in one sentence.
- Earlier notes of "~95 s per rank" for this checkpoint come from a different boot and
  configuration; they are **not** reproduced here. The numbers above are the ones this
  repository stands behind.

---

## 4. Expert shard count, index bytes and worker-side visibility **[measured + quoted]**

Collected 2026-09-28T10:11:44Z (head) and 10:12:06Z (worker, over SSH from the head).

```bash
ssh <HEAD_HOST> 'stat -c "%s %n" "$MODEL_DIR/model.safetensors.index.json"'
ssh <HEAD_HOST> 'ls "$MODEL_DIR" | grep -c "^model_pp0_ep.*_shard0\.safetensors$"'
ssh <HEAD_HOST> 'du -sh "$MODEL_DIR"'
```

```text
index bytes: 6921726 …/NewModels/MiMo-V2.6-Flash-MOPD/model.safetensors.index.json
expert shard files (model_pp0_ep*_shard0.safetensors): 128
total dir size: 166G
```

Worker-side check as printed by the kit during the same boot **[quoted, `logs/rename-restart.log:34`]**:

```text
[+] NFS volume mimo26-tp2-weights on <WORKER_IP> → nfs://<HEAD_IP>:/
ok index=6921726 shards=128
[+] <WORKER_IP> loads head weights via nfs://<HEAD_IP>:/ (index 6921726 bytes)
```

**Conclusions**

- 128 expert shards and a 6,921,726-byte index agree on the head (filesystem), in the
  checkpoint metadata (§1) and on the worker over NFS.
- The worker's own mount check is the strongest single line here: it is the rank that
  actually reads those shards, and it saw all 128.
- The worker keeps **no** copy of the checkpoint: the whole 166 GB lives on the head and is
  read over NFSv4 on ConnectX.

---

## 5. Text smoke **[measured]** and **[quoted]**

```bash
ssh <HEAD_HOST> '$KIT/start.sh smoke'          # text + image, as shipped
ssh <HEAD_HOST> 'python3 $SCRATCH/mopd-bench/smoke.py'   # the one-off runner behind this row
```

Measured 2026-09-28T10:17:21Z:

```text
== text smoke ==
text 19+23   http=200 0.4s prompt=25 completion=3 image_tokens=None finish=stop
    Q: What is 19 + 23? Answer with the number only.
    A: 42
```

Boot-time smoke of the same boot, from the kit's own log **[quoted, `logs/rename-restart.log`]**:

```text
[2026-09-28 09:56:55] POST /v1/chat/completions HTTP/1.1 200 OK
smoke: '42'
Ready: API on port 8888 (rank 0), no API key.
```

**Conclusion:** the served MOPD checkpoint answers the kit's arithmetic smoke with `42`,
both at boot (09:56:55Z) and on demand (10:17:21Z). No API key is configured for this
deployment.

---

## 6. Vision smoke — colour split, real photograph, real screenshot **[measured]**

The **colour-split row is reproducible from this repository**: `scripts/check-serving.sh`
builds a left-red/right-blue PNG in-process, asks the question printed below, asserts that the
answer names red (and that the request carried image tokens), and exits non-zero if it does not.
The real-image rows were recorded with a one-off runner that is **not** shipped; the image files
are OS-shipped and every prompt is printed in full below, so the same requests can be repeated by
hand or through `--real-image`.

```bash
ssh <HEAD_HOST> '$KIT/scripts/check-serving.sh'                    # split smoke + assertion
# optional, and entirely your own input: any JPEG/PNG that already exists on the node
ssh <HEAD_HOST> '$KIT/scripts/check-serving.sh --real-image <path to any JPEG/PNG on the node>'
```

`--real-image` is a **user-supplied** step: point it at an image you already have on the node
(its answer is informational — see the conclusions below). The recorded rows quoted next used
OS-shipped sample images, whose file names are given; no path from that machine is reproduced
here.

Three multimodal requests were recorded at 2026-09-28T10:17:21Z → 10:17:38Z, plus a second pass
over the same photograph at 2026-09-28T10:25:02Z → 10:25:16Z. The synthetic image is generated
inside the script; the two real images are files shipped with the base OS, so the test is
reproducible on any node with the same distro.

```text
== vision smoke: synthetic left/right colour split ==
split L=red R=blue   http=200 0.8s prompt=287 completion=15 image_tokens=256 finish=stop
    Q: Which colour is on the left half of this image, red or blue? Answer with one word.
    A: Based on the image provided, the left half is **red**.

== vision smoke: real images shipped with the OS (no private asset) ==
real image: keys-d.jpg (1533852 B)              http=200 13.5s prompt=8118 completion=45 image_tokens=8100 finish=stop
    Q: Describe this photograph in one sentence.
    A: The photograph captures a close-up view of a textured, woven fabric featuring a
       repeating pattern of diagonal stripes in alternating shades of dark grey and light grey.

real image: synaptic-packagelist.png (26374 B)  http=200  2.5s prompt=224 completion=49 image_tokens=200 finish=stop
    Q: What application's window is shown? Answer in one short sentence.
    A: Based on the columns "Package", "Installed Version", "Latest Version", and
       "Description", along with the package names (e.g., `apmd`, `aptitude`), this is the
       **Synaptic Package Manager** window.

== vision re-probe: real photograph ==
free-form one sentence   image_tokens=8100 prompt=8118 completion=39 -> ...a close-up, abstract texture
    of a dark, woven fabric, likely denim or a similar heavy textile, characterized by a diagonal twill pattern.
forced object naming     image_tokens=8100 prompt=8126 completion=48 -> ...a close-up of a **woven fabric**
    or **textile**. ...  **Woven fabric**
```

**Ground truth for the two real images** (established by opening the exact files this report
used — the files themselves are OS-shipped, so the claim can be re-checked without the cluster):

- `keys-d.jpg` is a close-up photograph of a **piano keyboard** (coloured keys receding
  diagonally into the dark) — not fabric.
- `synaptic-packagelist.png` is the Spanish-language package list of **Synaptic**, columns
  `Paquete / Versión instalada / Última versión / Tamaño / Descripción`, rows `apmd`,
  `aptitude`, `aptconf`… — exactly what the model named.

**Conclusions — read this before quoting a vision number**

- **The multimodal path executes and is genuinely image-conditioned.** `image_tokens` is
  256 (synthetic), 200 (screenshot) and 8,100 (1.5 MB photograph), never zero; the answer
  changes with the image, and the same image gives a stable answer across two different
  prompts.
- **Left/right colour attribution is correct** and not mirrored: red is on the left, and the
  model says red. `scripts/check-serving.sh` asserts exactly this on every run, so the row is
  reproducible rather than merely recorded.
- **The screenshot is identified correctly**, from its own contents (package names and column
  headers) rather than from the question.
- **The free-form description of the real photograph is wrong about the object**: the model
  repeatedly calls the piano keyboard a "woven fabric / denim" with a "diagonal twill
  pattern". The diagonal repetition and the dark/light tone banding are real, but the object
  is misidentified, and it does so again when asked directly for two words. So: image
  conditioning is proven, **object recognition accuracy on real photographs is not proven by
  this smoke** — do not publish a vision-quality claim from it.
- Everything above is a single 4096×4096 JPEG and a single 793×262 PNG; resolution and
  subject matter are not sampled. A real quality statement needs a small labelled set.

---

## 7. Cold-prefill ladder **[measured]**

Run 2026-09-28T10:08:22Z → 10:11:07Z (2 m 45 s), exit code 0, 5/5 rungs OK.

```bash
ssh <HEAD_HOST> "BENCH_BASE_URL=http://127.0.0.1:8888 \
  python3 \$KIT/bench/cold_prefill.py --out-name mopd-cold-prefill"
# defaults: one cold request per rung at 8k/16k/32k/64k/128k prompt tokens,
# temperature 0, thinking off, 8 generated tokens, fresh salt per rung.
```

Raw result lines (trimmed; the receipt was written to `bench/results/`, which is gitignored):

```text
model=MiMo-v2.6-Flash max_model_len=1048576
idle check: {"num_running_reqs": 0, "num_waiting_reqs": 0, ...}

=== cold 8,000 tokens ===   calibrated: tokenize=8,001 off_by=0.013%
  prompt_tokens=8,010    TTFT=3.66s   prefill=2189 tok/s  cached=None
=== prefix-cache follow-up on the first rung ===
  prompt_tokens=8,025    TTFT=0.34s   cached=8000
=== cold 16,000 tokens ===  calibrated: tokenize=16,001 off_by=0.006%
  prompt_tokens=16,010   TTFT=6.93s   prefill=2311 tok/s  cached=None
=== cold 32,000 tokens ===  calibrated: tokenize=32,001 off_by=0.003%
  prompt_tokens=32,010   TTFT=14.85s  prefill=2155 tok/s  cached=None
=== cold 64,000 tokens ===  calibrated: tokenize=64,001 off_by=0.002%
  prompt_tokens=64,010   TTFT=36.57s  prefill=1750 tok/s  cached=None
=== cold 128,000 tokens === calibrated: tokenize=128,001 off_by=0.001%
  prompt_tokens=128,010  TTFT=100.56s prefill=1273 tok/s  cached=None

| prompt tokens | TTFT (s) | prefill tok/s |
|---|---:|---:|
| 8,010 | 3.66 | 2189 |
| 16,010 | 6.93 | 2311 |
| 32,010 | 14.85 | 2155 |
| 64,010 | 36.57 | 1750 |
| 128,010 | 100.56 | 1273 |
receipt: bench/results/mopd-cold-prefill-20260928T101107Z.json
exit_code=0
```

**Conclusions**

- Cold prefill peaks at **2,311 tok/s (16k)** and degrades to **1,273 tok/s (128k)**; TTFT
  grows from 3.66 s to 100.56 s. Every rung's calibrated prompt is within 0.013% of target.
- `cached=None` on all cold rungs confirms the salts did their job; the follow-up request on
  the same 8k prompt returns `cached=8000` with **TTFT 0.34 s** (10.8× faster), so the radix
  prefix cache demonstrably works on this build.
- `chunked_prefill_size=8192` means the 128k rung is 16 chunks; the falling rate is the
  expected shape of this configuration, not a fault. This ladder is also the evidence behind
  the README's "Deviations from upstream" note on `CHUNKED_PREFILL_SIZE`.
- The 2,189 tok/s at 8k here is a single-stream loopback measurement on the head; it is not
  comparable with a worker-side figure from another deployment without saying so.

---

## 8. Decode ladder with speculative acceptance **[measured]**

Run 2026-09-28T10:12:10Z → 10:16:19Z (4 m 9 s), 24/24 cells OK — see the exit-code note.

```bash
ssh <HEAD_HOST> "BENCH_BASE_URL=http://127.0.0.1:8888 HEAD_CTN=mimo26-tp2-head \
  python3 \$KIT/bench/decode.py --out-name mopd-decode"
# defaults: 6 categories × concurrency 1/2/4/8, 256 pinned tokens per stream,
# one throwaway warm-up first, acceptance read from the engine's 'Decode batch' lines.
```

Aggregate throughput (tokens/s across all streams in the cell):

| category | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| counting | 46.8 | 77.7 | 120.0 | 226.0 |
| structured | 37.8 | 61.3 | 90.3 | 148.9 |
| json | 42.4 | 62.8 | 80.9 | 122.1 |
| code | 35.3 | 53.4 | 87.1 | 128.6 |
| math | 44.1 | 67.8 | 118.0 | 201.8 |
| prose | 31.8 | 45.3 | 64.2 | 98.3 |

Median per-stream decode rate (tokens/s, TTFT excluded):

| category | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| counting | 63.4 | 54.0 | 47.1 | 40.0 |
| structured | 47.9 | 39.4 | 30.2 | 23.2 |
| json | 56.2 | 42.7 | 27.0 | 20.0 |
| code | 44.5 | 33.8 | 27.3 | 20.8 |
| math | 59.8 | 47.6 | 40.7 | 34.4 |
| prose | 38.6 | 27.6 | 21.5 | 15.0 |

Median TTFT (s):

| category | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| counting | 1.44 | 1.87 | 3.12 | 2.68 |
| structured | 1.44 | 1.85 | 2.25 | 2.53 |
| json | 1.49 | 2.06 | 2.79 | 3.34 |
| code | 1.52 | 2.00 | 2.35 | 3.01 |
| math | 1.54 | 2.16 | 2.34 | 2.68 |
| prose | 1.46 | 2.04 | 2.63 | 3.23 |

Median accepted draft tokens per step (EAGLE MTP, max 4) — the drafter's real score:

| category | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| counting | 4.00 | 4.00 | 3.95 | 4.00 |
| structured | 3.48 | 3.36 | 3.89 | 3.45 |
| json | 3.95 | 3.70 | 3.44 | 3.49 |
| code | 3.17 | 3.74 | 3.42 | 3.37 |
| math | 3.50 | 3.96 | 4.00 | 3.93 |
| prose | 2.62 | 2.75 | 2.63 | 2.50 |

Engine log line the acceptance column is parsed from (real sample,
`docker logs mimo26-tp2-head | grep 'Decode batch'`, 2026-09-28 10:21:06Z):

```text
[2026-09-28 10:21:06 TP0 EP0] Decode batch, #running-req: 1, #full token: 192, full token usage: 0.00,
#swa token: 192, swa token usage: 0.00, accept len: 2.77, accept rate: 0.59, cuda graph: False,
gen throughput (token/s): 33.95, #queue-req: 0
```

### Why `decode.py` exits non-zero (no failures)

```text
receipt: bench/results/mopd-decode-20260928T101619Z.json
exit_code=1
finished_utc=2026-09-28T10:16:19Z
```

The receipt has **no `failed` entries and no `length_mismatches`** in any cell (all 24 cells
emitted exactly 256 tokens, `ok == concurrency` in every row). The non-zero exit comes from
the harness's `nan_free` heuristic: `bench/common.py` sets
`nan_free = "nan" not in body_text.lower()`, a plain substring test that ordinary prose trips.
Replayed 5× with the same settings (2026-09-28T10:20:27Z → 10:21:10Z):

```text
sample 0: completion_tokens=256 finish=length literal 'nan' occurrences=0
sample 1: completion_tokens=256 finish=length literal 'nan' occurrences=1
    ...the autoregressive generation phase—is predominantly memory-bound...
sample 2: completion_tokens=256 finish=length literal 'nan' occurrences=0
sample 3: completion_tokens=256 finish=length literal 'nan' occurrences=0
sample 4: completion_tokens=256 finish=length literal 'nan' occurrences=1
    ...the autoregressive generation phase—is predominantly memory-bound...
samples containing the literal substring 'nan': 2/5
```

**Conclusions**

- Decode scales **4.8× at C8 for counting** (46.8 → 226.0 tok/s) and **3.1× for prose**
  (31.8 → 98.3 tok/s); per-stream rate falls as batch grows, as expected on a memory-bound
  MoE.
- EAGLE MTP is doing real work: median accepted length 2.5–4.0 of a possible 4, high on
  structured/numeric output (counting, math ≈ 4.0) and lowest on free prose (2.5–2.75) —
  i.e. the category split this harness exists to expose, and the reason a blended single
  number would mislead.
- TTFT stays at 1.4–3.3 s in every cell, including C8, because the prompts are short.
- The `exit_code=1` is **the harness's `nan_free` substring heuristic, not a model fault**
  (no NaN is observable on the OpenAI path: normal `usage`, `finish=length`, engine log lines
  parse cleanly). The receipt's `text_head` keeps only the first 160 characters, which is why
  the flag cannot be replayed from the receipt itself; the probe above reproduces the
  trigger from the model's own output.
- **DFlash is not benchmarked in this repository.** The EAGLE arm is the shipped default and
  the only arm measured here; upstream reported DFlash ahead on code and behind on prose.
  Run `bench/decode.py` once per drafter with the same category set before choosing.

---

## 9. Reproduction checklist

```bash
# 0. expected constants for this deployment
#     128 expert shards, index 6,921,726 bytes, max_model_len 1,048,576,
#     pool 2,546,496 tokens, weight 82.9 GB + KV 17.2 GB resident per rank

# 1. checkpoint (read-only; safe while serving)
ssh <HEAD_HOST> "MODEL_DIR=\$HOME/NewModels/MiMo-V2.6-Flash-MOPD python3 \$KIT/scripts/verify-weights.py"
#    expect VERIFY_RESULT: PASS

# 2. live config / memory
ssh <HEAD_HOST> 'curl -s http://127.0.0.1:8888/v1/loads' | python3 -m json.tool
ssh <HEAD_HOST> 'curl -s http://127.0.0.1:8888/server_info' | python3 -m json.tool

# 3. smokes (also proves the multimodal path)
ssh <HEAD_HOST> '$KIT/scripts/check-serving.sh'

# 4. benchmarks — each refuses to run unless the engine is idle
ssh <HEAD_HOST> "BENCH_BASE_URL=http://127.0.0.1:8888 python3 \$KIT/bench/cold_prefill.py --out-name mopd-cold-prefill"
ssh <HEAD_HOST> "BENCH_BASE_URL=http://127.0.0.1:8888 HEAD_CTN=mimo26-tp2-head python3 \$KIT/bench/decode.py --out-name mopd-decode"
```

## 10. Caveats and limits of this evidence

1. **Head-local measurements.** All inference calls were issued on the head against
   `127.0.0.1:8888` (the bench docstrings prefer a worker). The head also hosts the
   HTTP front end, tokenizer/detokenizer and the NFS export, so these numbers are not
   directly comparable with worker-side figures from other deployments.
2. **Single deployment, single run.** Each ladder was run once. Rung-to-rung differences of a
   few percent are inside run-to-run noise; nothing here is an error bar.
3. **Not measured here:** long-context (>128k) prefill, concurrency above 8, decode beyond
   256 tokens/stream, the DFlash drafter, needle/retrieval quality.
4. **Engine log cadence.** Acceptance is parsed from `Decode batch` lines, logged every
   `decode_log_interval=40` decode steps, so a 256-token cell contributes only 1–3 samples;
   the medians are therefore coarse by construction.
5. **`nvidia-smi` cannot report GB10 memory** (unified memory) — it prints `[N/A]`. Use
   `/v1/loads` and `free -m` as done in §2.
6. **Quoted rows** (§3 boot lines, §4 worker check, §5 boot smoke) come from the logs of the
   running container and the kit's own boot log for this same boot; they were not regenerated
   in this session and the service was never restarted to regenerate them.
7. **Vision ground truth (§6) was established by opening the exact image files**, not by
   trusting the model. Both files are OS-shipped, so the check can be repeated anywhere.
8. **Scratch paths.** The one-off runner scripts used for this report lived under
   `$SCRATCH/mopd-bench/` on the head and wrote their logs and bench receipts under `$SCRATCH`
   and `bench/results/` (gitignored). Nothing was written into the kit, and none of it is
   published here.

## 11. The collection did not disturb the service **[measured]**

Checked at 2026-09-28T10:26:52Z, after every measurement above:

```bash
ssh <HEAD_HOST> "date -u +%FT%TZ; curl -s -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8888/health"
ssh <HEAD_HOST> "docker inspect -f 'started={{.State.StartedAt}} running={{.State.Running}}' mimo26-tp2-head"
ssh <HEAD_HOST> "stat -c '%y %n' \$KIT/start.sh \$KIT/.env \$KIT/state/launch.json"
```

```text
2026-09-28T10:26:52Z
health=200
started=2026-09-28T09:52:04.555677003Z running=true

2026-09-28 17:49:48 +0800 $KIT/start.sh
2026-09-28 17:49:50 +0800 $KIT/.env
2026-09-28 17:52:29 +0800 $KIT/state/launch.json
```

**Conclusion:** the engine still reports the same `StartedAt` as at the start of collection
(no restart), `/health` is 200, and `start.sh`, `.env` and `state/launch.json` still carry
their pre-collection modification times (before the collection session began). No service
state, kit file, container or NFS export was modified — everything in this report is read-only
collection plus inference calls.
