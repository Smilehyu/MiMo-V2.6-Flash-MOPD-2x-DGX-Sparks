#!/usr/bin/env bash
# head-pull-base.sh — pull the pinned SGLang base image on the head through
# domestic mirrors, then tag it to the canonical name the kit builds FROM.
#
# Why it exists: on networks where Docker Hub is unreachable, `./start.sh pull`
# fails before the build can start. This tries a list of mirrors in order and
# retags the first success, so the rest of the kit needs no change.
#
# Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD deployment notes. Derived from
# MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see LICENSE and the
# "License & attribution" section of README.md.
#
# usage: ./head-pull-base.sh           # head; run ./start.sh pull on the worker
set -uo pipefail

BASE=lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5
MIRRORS="${MIRRORS:-docker.1ms.run docker.m.daocloud.io hub.rat.dev dockerproxy.net docker.xuanyuan.me docker.1panel.live}"

echo "[pull-base] $(date '+%F %T') start $BASE"
for m in $MIRRORS; do
  echo "== trying $m"
  if timeout 3600 docker pull "$m/$BASE"; then
    docker tag "$m/$BASE" "$BASE"
    echo "TAGGED $BASE <- $m/$BASE"
    break
  fi
done
docker image inspect "$BASE" >/dev/null 2>&1 && echo "PULL_BASE_OK" || echo "PULL_BASE_FAILED"
echo "== alpine for the NFS exporter image"
if timeout 600 docker pull docker.m.daocloud.io/library/alpine:latest; then
  docker tag docker.m.daocloud.io/library/alpine:latest alpine:latest
  echo "ALPINE_OK"
else
  timeout 600 docker pull docker.1ms.run/library/alpine:latest && \
    docker tag docker.1ms.run/library/alpine:latest alpine:latest && echo "ALPINE_OK" || echo "ALPINE_FAILED"
fi
echo "[pull-base] $(date '+%F %T') end"
