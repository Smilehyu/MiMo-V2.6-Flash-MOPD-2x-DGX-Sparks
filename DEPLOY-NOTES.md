# MiMo-V2.6-Flash-MOPD × 2×DGX Spark (TP=2/EP=2) 部署记录

> 2026-09-27 / 2026-09-28 · head `$KIT`（本文用 `<HEAD_HOST>`、`<HEAD_IP>` 指代 head，用
> `<WORKER_HOST>`、`<WORKER_IP>` 指代 worker；`$KIT` 是本仓库在 head 上的路径）
> 权重来源：**ModelScope** `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1`，全程未走 Hugging Face / VPN。
> 定位：**本方案的实施记录**（中文）。英文主文档是根目录 `README.md`；可复现的测试数据、
> 命令与采集时间在 `docs/TEST-RESULTS.md`。本文的数字只覆盖本方案这一次部署，不与上游
> 对 RL checkpoint 的测量混用。
>
> 历史：2026-09-27 首装为 `MiMo-V2.6-Flash-RL`；2026-09-28 升级为 MOPD（128 专家分片）并
> 删除 RL 权重。迁移过程见文末两则附录。
>
> **运维形态：只有两个容器和一套 CLI。** 两个 rank 由 `./start.sh` 与 `./stop.sh` 直接
> 编排，健康状态用 `./start.sh status`、`docker logs` 和引擎自己的 HTTP 端点检查；
> 本仓库不含任何旁路服务或中间层。

## 一、结果

| | |
|---|---|
| 服务 | `MiMo-v2.6-Flash`，TP=2 / EP=2，**1M 上下文**，文本/图像/视频/音频（omni） |
| 端口 | 直连 head 的 `http://<HEAD_IP>:8888`（本机回环 `http://127.0.0.1:8888`），无 API key |
| 权重加载 | head 115.51 s / 82.92 GB；worker 经 NFS 229.34 s / 83.42 GB（同一次 boot） |
| 冷启动 | 容器 StartedAt 09:52:04.6Z → `fired up` 09:56:49Z = **284 s** |
| KV 池 | **max_total_num_tokens=2,546,496**（2.43× 全 1M 上下文），SWA 侧 76,352；`available_gpu_mem=9.15 GB`，host available 6,186 MiB |
| 投机解码 | **EAGLE MTP**（默认，draft 4 / steps 3 / topk 1，multi-layer），decode CUDA graph 关闭（EAGLE 下强制） |
| 文本冒烟 | `19+23→42` ✓（boot 时 09:56:55Z，按需 10:17:21Z 各一次） |
| 视觉冒烟 | 左右分色图左侧判为 red ✓；真实截图（Synaptic 包列表，西语）识别正确 ✓；**真实照片（钢琴键盘特写）被稳定误判为"织物"**，见 `docs/TEST-RESULTS.md` §6 的诚实结论 |
| 基准 | cold prefill 8k–128k 五档 5/5 成功（峰值 2,311 tok/s @16k，128k 1,273 tok/s）；decode 24/24 cell 全部 ok，聚合 C1 31.8–46.8 → C8 98.3–226.0 tok/s。全部命令见 `docs/TEST-RESULTS.md` |

参数面（`.env`）里与本方案强相关的偏离，见 `README.md` 的 **Deviations from upstream**：
`CHUNKED_PREFILL_SIZE=8192`（上游 2048 并警告在 0.93 下不要提高）、`MEM_GUARD_GIB=2`
（上游 4）、`NCCL_MAX_NCHANNELS=16`（上游 8，配合双网卡多轨道）。

## 二、部署形态

```
./start.sh serve ──> 容器 mimo26-tp2-head   (rank 0, TP0/EP0, API :8888, NFSv4 导出权重)
                 └─> 容器 mimo26-tp2-worker (rank 1, TP1/EP1, 经 ConnectX 挂载 head 的权重)
```

| 文件（head 的 `$KIT`） | 作用 |
|---|---|
| `start.sh` / `stop.sh` / `boot.py` | 仓库套件；`cmd_download` 走 **ModelScope**（`MS_PYTHON`），`EXPECTED_SHARDS=128` |
| `.env` | 本集群配置（拓扑 / 网卡 / RoCE / 内存参数）；由 `.env.example` 复制后编辑 |
| `mimo26.sh` | 运维入口：`start\|restart\|stop\|status\|logs\|smoke\|doctor\|download`，含双节点清缓存仪式 |
| `download-ms.sh` | ModelScope 下载 + 严格完整性自检（5 必需文件 + index 引用分片 + 128 专家分片 + `WEIGHTS_OK`） |
| `head-pull-base.sh` | Docker Hub 直连不通时经国内 mirror 拉基础镜像并 retag |
| `scripts/verify-weights.py` | 逐文件字节数 + sha256（<256 MiB 全量 + 大分片抽样）+ 分片计数，输出 `VERIFY_RESULT` |
| `scripts/check-serving.sh` | 启动后只读核验：容器状态、引擎里程碑、权重挂载、文本冒烟、**图像冒烟**、index 字节数 |

`mimo26.sh` 的清缓存仪式需要 sudo：口令只从环境变量 `MIMO26_SUDO_PW` 读取，并经 stdin
喂给 `sudo -S`（不出现在命令行 / 进程表）。若两节点已配置免密 sudo，直接用 `./start.sh`
即可——注意 `start.sh` 自带的 `sudo -n drop_caches` 在需要口令的机器上会**静默跳过**，
这正是 `mimo26.sh` 仪式存在的理由。

## 三、日常操作

```bash
# head 上（$KIT 为仓库路径）
cd "$KIT"
./start.sh status              # 容器状态 + 挂载
./start.sh serve               # 启动两个 rank（内部会先跑 share 的 NFS 校验）
./start.sh restart             # stop -> 等 GPU 释放 -> serve
./stop.sh                      # 拆除；等每个节点驱动释放 GPU
./stop.sh --unmount            # 同时卸掉 worker 的 NFS volume
./start.sh smoke               # 文本 + 图像冒烟
./start.sh logs worker         # worker rank 日志
./scripts/check-serving.sh     # 只读核验（含多模态冒烟）
```

```bash
# 对话（默认采样：temperature 1.0 + top_p 0.95；思考默认开，可关）
curl http://<HEAD_IP>:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-v2.6-Flash","messages":[{"role":"user","content":"你好"}],
       "max_tokens":128,"temperature":1.0,"top_p":0.95,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

- 重启机器后**模型不会自动加载**：手动 `./mimo26.sh start`（或 `./start.sh serve`）。
- 无 API key；需要鉴权时在 `.env` 设 `API_KEY`。

## 四、切换投机解码 drafter（boot-time 决定，改 `.env` 后 restart）

```bash
cd "$KIT"
# EAGLE MTP（默认）：散文/聊天好，KV 池大。必须 CUDA_GRAPH_MAX_BS_DECODE=0
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=EAGLE/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=4/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=0/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2743808/' .env
# DFLASH：代码/结构化输出；decode CUDA graph 可开，但 drafter 会再占一份 KV 池，
# 因此 MAX_TOTAL_TOKENS 要降到 ~2299968（本方案未在本仓库复现该臂）
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=DFLASH/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=8/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=8/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2299968/' .env
./mimo26.sh restart
```

本方案发布时的实测臂是 EAGLE MTP：每类别的接受长度中位数 2.5–4.0（counting/math ≈4.0，
prose 2.50–2.75），数据与命令见 `docs/TEST-RESULTS.md` §8。

## 五、踩坑记录（本机实测）

1. **`MEM_GUARD_GIB` 不能沿用 4（配 `MEM_FRACTION_STATIC=0.93`）**。用 4 时：权重加载、
   KV 池分配全部成功，随后 FlashInfer autotune 阶段 `MemAvailable` 降到 4.0 GiB 附近，
   `boot.py` 的用户态 OOM 守卫直接杀掉 sglang（两个 rank 同时），表面症状只有
   `Server failed to become healthy`，日志里才看到 `MEMORY GUARD` 行。改成 2 后本 profile
   的 boot 结束在 `available_gpu_mem=9.15 GB`、host available 6,186 MiB。这是本仓库
   `.env.example` 把默认值写成 2 的原因（上游默认 4）。
2. Docker Hub 直连不通：基础镜像 `lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5`
   经 `docker.1ms.run` 拉取后 retag（`head-pull-base.sh` 把这套尝试固化成脚本）；alpine
   同理（NFS exporter 镜像用）。worker 侧 daemon 配了 registry-mirrors，可作 save|load 后备源。
3. `CHUNKED_PREFILL_SIZE` 提到 8192 后（上游默认 2048、且上游在 0.93 下警告不要提高），
   本 profile 反复冷启动与 8k–128k prefill 阶梯均正常（5/5，见 `docs/TEST-RESULTS.md` §7）。
   若你的 boot 出现 `MEMORY GUARD` 行，先回退到 2048。

## 六、下载 / 构建记录

| 步骤 | 用时 / 结果 |
|---|---|
| ModelScope 权重（**155 个 blob / 165.56 GiB / 128 个专家分片**） | 下载后 `verify-weights.py` 全量校验 PASS：0 缺失、0 尺寸不符、29 个文件 sha256 一致、128/128 分片（25.42 s） |
| SGLang 基础镜像（33.1 GB，经 docker.1ms.run） | head / worker 双路并行拉取 |
| `mimo26-spark:local` 构建（ffmpeg + torchcodec 0.16.0） | 一次通过，`docker save \| load` 分发到 worker |
| 冷启动（权重 → KV 池 → autotune → ready） | **284 s**（容器 StartedAt → `fired up`） |
| worker 经 NFS 读同一份权重 | `ok index=6921726 shards=128`，加载 229.34 s |

## 七、健康检查 / 排障入口

```bash
curl -s http://127.0.0.1:8888/health                          # 200 = 服务在
curl -s http://127.0.0.1:8888/v1/loads | python3 -m json.tool # 权重/KV/并发实况
curl -s http://127.0.0.1:8888/server_info | python3 -m json.tool  # launch_command / startup_time
docker logs -f mimo26-tp2-head                                # 引擎日志（accept len 等）
./mimo26.sh logs worker                                       # worker rank 日志
./scripts/check-serving.sh                                    # 一次性只读核验
```

关注行：`max_total_num_tokens=`（真实池）、`Decode batch ... accept len`（drafter 表现）、
`MEMORY GUARD`（内存守卫触发）、`Load weight end`（加载耗时与常驻）。

## 2026-09-28 — 权重升级为 MiMo-V2.6-Flash-MOPD

- ModelScope `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1`
  （2026-09-28 02:25 CST 发布）。
- `.env` 切换：`MODEL_DIR` / `COMMON_MODEL` / `MS_REPO` / `MS_REVISION` / `HF_REPO`，以及
  **`EXPECTED_SHARDS=128`**（MOPD 把专家分成 128 个分片文件，RL 是 64）；容器内挂载名改为
  `/models/MiMo-V2.6-Flash-MOPD`。
- 服务前校验：155 个上游文件逐字节一致，29 个文件 sha256（含 `ep0/ep1/ep64/ep127`、MTP 头、
  DFlash drafter、audio tokenizer），128/128 分片（`scripts/verify-weights.py`）。
- 切换时踩到两个运维陷阱：
  1. `nfs_ensure_server()` 在旧 exporter 仍挂着**旧 checkpoint 路径**时会 die —— 旧
     NFS 容器无法删除（PID1 已死 / containerd-shim 卡住，`docker rm -f` 报
     "did not receive an exit event"）。先 kill 卡住的 shim，再 `docker rm -f` 才恢复。
     通用教训：换权重路径前先 `./stop.sh --unmount` 把旧导出拆干净。
  2. 冷启动期间（>120 s）任何"等待容器出现"的外部分支都可能反复重启，把冷启动打断。
     本方案因此不再依赖任何常驻的旁路进程 —— 只用 `start.sh` / `stop.sh` / `mimo26.sh`。
- 启动后抽查：worker 的 NFS 校验 `ok index=6921726 shards=128`；文本冒烟 `42`；左右分色图
  判对；真实截图识别正确（真实照片的误判见 `docs/TEST-RESULTS.md` §6）。
- RL 权重已按要求删除（无回滚副本）。回退方式：改回 `MODEL_DIR` / `EXPECTED_SHARDS=64`
  并重新下载 RL checkpoint。

## 2026-09-28 — 全 kit 命名统一为 MiMo-V2.6-Flash-MOPD

- 去除全 kit 的 RL 命名：`start.sh`（头注释、`MODEL_DIR` / `COMMON_MODEL` 默认值、
  `MS_REPO` / `MS_REVISION` / `HF_REPO` / `HF_REVISION` / `EXPECTED_SHARDS=128`、
  **`MODEL_MOUNT=/models/MiMo-V2.6-Flash-MOPD`**、doctor 横幅）、`boot.py`（docstring /
  `REPO` / `REVISION` / `MODEL` 默认路径）、`download-ms.sh`、`README.md`、
  `.env` / `.env.example`、`.github/ISSUE_TEMPLATE/1_bug.md`。
- 三方一致性（容器内权重路径必须同值，否则静默读空）：`Dockerfile` 的 `ENV MODEL_PATH`、
  `boot.py` 的 `MODEL` 默认值、`start.sh` 的 `MODEL_MOUNT` —— 三者均为
  `/models/MiMo-V2.6-Flash-MOPD`。
- 验证：`bash -n` + `py_compile` 通过；冷启动后 head 挂载
  `$HOME/NewModels/MiMo-V2.6-Flash-MOPD -> /models/MiMo-V2.6-Flash-MOPD`，worker 校验
  `ok index=6921726 shards=128`，文本 `42`、左右分色图识别正确。
