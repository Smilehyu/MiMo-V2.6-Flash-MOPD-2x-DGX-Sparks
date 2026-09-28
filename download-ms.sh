#!/usr/bin/env bash
# download-ms.sh — fetch the MiMo-V2.6-Flash-MOPD checkpoint from ModelScope.
#
# Run it on the head only; workers read the checkpoint over NFS and never keep
# a copy. It is the "no VPN / no Hugging Face" download path: `./start.sh
# download` uses the same ModelScope keys, and this script is the standalone,
# strictly-checked version of it (it can also be used to re-verify an existing
# tree).
#
# Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: paths come from the profile
# (or from $HOME), never from a hardcoded account, and the shard gate is 128.
# Derived from MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see LICENSE
# and the "License & attribution" section of README.md.
#
# usage: ./download-ms.sh            # uses .env when present
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the profile when there is one, so MS_* and MODEL_DIR stay in one place.
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

PY="${MS_PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-$HOME/NewModels/MiMo-V2.6-Flash-MOPD}"
REPO="${MS_REPO:-XiaomiMiMo/MiMo-V2.6-Flash-MOPD}"
REV="${MS_REVISION:-24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-128}"

mkdir -p "$MODEL_DIR"
echo "[download-ms] $(date '+%F %T') start repo=$REPO rev=$REV dir=$MODEL_DIR"
echo "[download-ms] interpreter: $PY"

REPO="$REPO" REV="$REV" MODEL_DIR="$MODEL_DIR" "$PY" - <<'PYEOF'
import os

from modelscope import snapshot_download

repo = os.environ["REPO"]
rev = os.environ["REV"]
dest = os.environ["MODEL_DIR"]
try:
    p = snapshot_download(repo, revision=rev, local_dir=dest, max_workers=8)
except Exception as e:  # noqa: BLE001 - the fallback is deliberate, but loud
    print("pinned revision %s failed (%s)" % (rev, e), flush=True)
    print("!!! FALLING BACK TO master - the result is NOT the pinned revision and"
          " must be re-checked before serving", flush=True)
    p = snapshot_download(repo, revision="master", local_dir=dest, max_workers=8)
print("SNAPSHOT_DONE", p)
PYEOF
rc=$?
echo "[download-ms] snapshot rc=$rc - verifying"

EXPECTED_SHARDS="$EXPECTED_SHARDS" "$PY" - "$MODEL_DIR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expect = int(os.environ.get("EXPECTED_SHARDS", "128"))
need = ["config.json", "model.safetensors.index.json", "preprocessor_config.json",
        "audio_tokenizer/model.safetensors", "model_mtp.safetensors"]
missing = [n for n in need if not (root / n).is_file() or (root / n).stat().st_size == 0]
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted({f for f in set(index["weight_map"].values()) if not (root / f).is_file()})
n_local = len(list(root.glob("model_pp0_ep*_shard0.safetensors")))
print("required missing:", missing or "none")
print("index-referenced missing:", len(shards), shards[:5])
print("expert shards on disk:", n_local, "(MOPD layout expects %d)" % expect)
print("dflash present:", (root / "dflash").is_dir(),
      " audio_tokenizer:", (root / "audio_tokenizer").is_dir())
if missing or shards or n_local < expect:
    sys.exit(1)
print("WEIGHTS_OK")
PYEOF
rc=$?
echo "[download-ms] $(date '+%F %T') end rc=$rc"
exit $rc
