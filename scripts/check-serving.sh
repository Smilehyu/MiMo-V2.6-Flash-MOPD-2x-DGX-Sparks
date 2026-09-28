#!/usr/bin/env bash
# check-serving.sh — post-boot verification of a served MiMo-V2.6-Flash-MOPD
# deployment: container state, engine milestones, the container's weight mount,
# a text smoke, a multimodal (image) smoke with a real assertion, and the
# checkpoint's index size.
#
# Run it ON THE HEAD, from the repository root (or anywhere: it finds the
# repository itself). It only reads — no start, no stop, no config change.
#
# The image smoke is the reproducible part of docs/TEST-RESULTS.md §6: it builds
# a left-red/right-blue PNG in-process, asks which colour is on the left half,
# and FAILS (non-zero exit) if the answer does not name red. Pass
# `--real-image <path>` to also send a real image file of your own (any JPEG or
# PNG on the node) and print what the model says about it — that part is
# informational, because object recognition on real photographs was NOT
# established by the recorded run (see §6).
#
# Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: paths come from the profile
# instead of a hardcoded account, container names come from .env, the section
# that queried a second, external endpoint was removed — this repository has no
# other control plane — and the colour-split smoke gained an assertion plus an
# optional real-image step. Derived from
# MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see
# LICENSE and the "License & attribution" section of README.md.
#
# usage: ./scripts/check-serving.sh [--real-image PATH] [--port N]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

REAL_IMAGE=""
PORT_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --real-image)
      shift
      REAL_IMAGE="${1:-}"
      if [ -z "$REAL_IMAGE" ] || [ ! -f "$REAL_IMAGE" ]; then
        echo "--real-image needs an existing file" >&2
        exit 2
      fi
      ;;
    --port)
      shift
      PORT_OVERRIDE="${1:-}"
      ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

HEAD_CTN="${HEAD_CTN:-mimo26-tp2-head}"
PORT="${PORT_OVERRIDE:-${PORT:-8888}}"
MODEL_DIR="${MODEL_DIR:-$HOME/NewModels/MiMo-V2.6-Flash-MOPD}"
API_KEY_FILE="$ROOT/state/api-key"

echo "== ./start.sh status =="
./start.sh status || true
echo
echo "== head container mounts =="
docker inspect "$HEAD_CTN" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' 2>/dev/null | head -6
echo "== engine milestones =="
docker logs "$HEAD_CTN" 2>&1 | grep -E 'Ready: API|Total time|Load weight|Capturing' | tail -6
echo "== smoke (text) =="
./start.sh smoke || true
echo
echo "== smoke (image): synthetic left/right colour split =="
PORT="$PORT" API_KEY_FILE="$API_KEY_FILE" REAL_IMAGE="$REAL_IMAGE" python3 - <<'PY'
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

port = os.environ.get("PORT", "8888")
base = "http://127.0.0.1:%s" % port
real_image = os.environ.get("REAL_IMAGE", "") or ""

# state/api-key is optional: this deployment serves without an API key.
try:
    key = open(os.environ["API_KEY_FILE"]).read().strip()
except (OSError, KeyError):
    key = ""
hdr = {"Content-Type": "application/json"}
if key:
    hdr["Authorization"] = "Bearer " + key

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif"}


def png(w, h, left, right):
    """A w x h PNG whose left half is one RGB colour and right half another."""
    half = w // 2
    row = b"\x00" + bytes(left) * half + bytes(right) * (w - half)
    raw = row * h

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def data_url(path, blob):
    ext = os.path.splitext(path)[1].lower()
    return "data:%s;base64,%s" % (MIME.get(ext, "image/jpeg"), base64.b64encode(blob).decode())


def ask(question, url, max_tokens=48, timeout=300):
    body = {
        "model": "x", "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    }
    try:
        model = json.load(urllib.request.urlopen(
            urllib.request.Request(base + "/v1/models", headers=hdr), timeout=30))
        body["model"] = model["data"][0]["id"]
    except Exception:
        pass
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=hdr, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    usage = d.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    msg = d["choices"][0]["message"]
    return {
        "elapsed": time.time() - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": d["choices"][0].get("finish_reason"),
        "image_tokens": details.get("image_tokens"),
        "text": (msg.get("content") or msg.get("reasoning_content") or "").strip(),
    }


ok = True

question = "Which colour is on the left half of this image, red or blue? Answer with one word."
try:
    res = ask(question, data_url("split.png", png(128, 64, (200, 30, 30), (30, 30, 200))))
    first_line = res["text"].splitlines()[0] if res["text"] else "(empty)"
    print("split L=red R=blue   http=200 %.1fs prompt=%s completion=%s image_tokens=%s finish=%s"
          % (res["elapsed"], res["prompt_tokens"], res["completion_tokens"],
             res["image_tokens"], res["finish"]))
    print("    Q: %s" % question)
    print("    A: %s" % first_line)
    if res["image_tokens"] in (None, 0):
        print("split assertion: FAIL (no image tokens: the request was not image-conditioned)")
        ok = False
    elif "red" in res["text"].lower():
        print("split assertion: PASS (the answer names red, which is on the left)")
    else:
        print("split assertion: FAIL (the answer does not name red)")
        ok = False
except Exception as exc:  # noqa: BLE001 - a report script must not mask the rest
    print("split smoke FAILED: %s %s" % (type(exc).__name__, str(exc)[:200]))
    ok = False

if real_image:
    print()
    print("== smoke (image): real file %s ==" % real_image)
    with open(real_image, "rb") as fh:
        blob = fh.read()
    question = "Describe this photograph in one sentence."
    try:
        res = ask(question, data_url(real_image, blob), max_tokens=64)
        print("real image: %s (%d B)   http=200 %.1fs prompt=%s completion=%s image_tokens=%s finish=%s"
              % (os.path.basename(real_image), len(blob), res["elapsed"], res["prompt_tokens"],
                 res["completion_tokens"], res["image_tokens"], res["finish"]))
        print("    Q: %s" % question)
        print("    A: %s" % res["text"])
        print("    note: informational only - this run does not establish object-recognition")
        print("          accuracy on real photographs (docs/TEST-RESULTS.md §6).")
    except Exception as exc:  # noqa: BLE001
        print("real-image smoke FAILED: %s %s" % (type(exc).__name__, str(exc)[:200]))

sys.exit(0 if ok else 1)
PY
smoke_rc=$?
echo
echo "== checkpoint index bytes =="
stat -c '%s %n' "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null \
  || echo "(no index at $MODEL_DIR/model.safetensors.index.json)"
echo
if [ "$smoke_rc" -eq 0 ]; then
  echo "CHECK_SERVING_OK (colour-split assertion passed)"
else
  echo "CHECK_SERVING_FAILED (colour-split assertion failed, rc=$smoke_rc)" >&2
fi
exit "$smoke_rc"
