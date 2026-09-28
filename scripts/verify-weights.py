#!/usr/bin/env python3
"""Verify a MiMo-V2.6-Flash-MOPD checkpoint against the ModelScope metadata.

- every file in the upstream master listing must exist locally with the exact byte size
- sha256 is recomputed for every file < 256 MiB, plus a sample of the big shards
- the expert-shard count on disk must equal the count upstream publishes (128 for
  the MOPD layout)
Prints PASS/FAIL and exits non-zero on failure.

Run it on the head, where the checkpoint lives. Nothing is written: this is a
read-only check, so it is safe while the cluster is serving.

usage:
    ./scripts/verify-weights.py [MODEL_DIR]
Environment:
    MODEL_DIR    checkpoint directory (default $HOME/NewModels/MiMo-V2.6-Flash-MOPD)
    MS_REPO      ModelScope repository (default XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
    MS_REVISION  pinned revision (default 24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1)

Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: the checkpoint path, repository and
revision come from arguments/environment instead of hardcoded values. Derived
from MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see LICENSE and the
"License & attribution" section of README.md.
"""
import hashlib
import json
import os
import ssl
import sys
import urllib.request

REPO = os.environ.get("MS_REPO", "XiaomiMiMo/MiMo-V2.6-Flash-MOPD")
REV = os.environ.get("MS_REVISION", "24c7ffbf6a1dd64d8d201b075ff2a19fc68f6bf1")
ROOT = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("MODEL_DIR", os.path.expanduser("~/NewModels/MiMo-V2.6-Flash-MOPD")))
SMALL_LIMIT = 256 * 1024 * 1024
SAMPLE = ["model_pp0_ep0_shard0.safetensors", "model_pp0_ep1_shard0.safetensors",
          "model_pp0_ep64_shard0.safetensors", "model_pp0_ep127_shard0.safetensors",
          "model_mtp.safetensors", "dflash/dflash_draft_model.safetensors",
          "audio_tokenizer/model.safetensors"]


def fetch(url):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as r:
        return json.loads(r.read().decode("utf-8"))


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


if not os.path.isdir(ROOT):
    print("no checkpoint directory: %s" % ROOT)
    print("set MODEL_DIR, pass it as argv[1], or download it with ./download-ms.sh")
    sys.exit(2)

url = "https://modelscope.cn/api/v1/models/%s/repo/files?Revision=%s&Recursive=true" % (REPO, REV)
files = {f["Path"]: f for f in fetch(url)["Data"]["Files"] if f.get("Type") == "blob" or f.get("Size")}
print("upstream files: %d" % len(files))

missing, wrong_size, bad_hash, hashed = [], [], [], 0
for path, meta in sorted(files.items()):
    want = meta.get("Size") or 0
    local = os.path.join(ROOT, path)
    if not os.path.isfile(local):
        missing.append(path)
        continue
    got = os.path.getsize(local)
    if got != want:
        wrong_size.append((path, want, got))
        continue
    if want < SMALL_LIMIT or os.path.basename(path) in SAMPLE:
        if meta.get("Sha256"):
            hashed += 1
            if sha256(local) != meta["Sha256"]:
                bad_hash.append(path)

total = sum((m.get("Size") or 0) for m in files.values())
print("expected total: %d bytes (%.2f GiB)" % (total, total / 1024 ** 3))
print("missing files       : %d %s" % (len(missing), missing[:5]))
print("size mismatches     : %d %s" % (len(wrong_size), wrong_size[:3]))
print("sha256 verified     : %d files, mismatches: %d %s" % (hashed, len(bad_hash), bad_hash[:3]))
shards = [p for p in files if p.startswith("model_pp0_ep")]
on_disk = len([p for p in os.listdir(ROOT)
               if p.startswith("model_pp0_ep") and p.endswith("_shard0.safetensors")])
print("expert shards       : upstream %d / on disk %d" % (len(shards), on_disk))

ok = not missing and not wrong_size and not bad_hash and on_disk == len(shards)
print("VERIFY_RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
