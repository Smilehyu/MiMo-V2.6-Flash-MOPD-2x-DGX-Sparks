#!/usr/bin/env bash
# mimo26.sh — operational wrapper for MiMo-V2.6-Flash-MOPD on two DGX Sparks.
#
# Why it exists: on GB10 host RAM is GPU memory, so a launch that follows big
# file reads (a weight download, a previous boot) sees a large page cache and
# SGLang sizes its KV budget from MemAvailable. Dropping the caches and setting
# vm.swappiness=0 on BOTH nodes before a start restores that budget. Everything
# else is this repository's start.sh / stop.sh — there is no other control
# plane: start, stop and inspect the two containers directly.
#
# Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: the sudo password is read from
# $MIMO26_SUDO_PW and is never written into a command line (it is piped into
# `sudo -S` on stdin, locally and over ssh), the worker address comes from the
# profile instead of a literal, and the repo directory is the script's own.
# Derived from MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); see LICENSE
# and the "License & attribution" section of README.md.
#
# usage: ./mimo26.sh [start|restart|stop|status|logs [worker|N]|smoke|doctor|download]
#
# Environment:
#   MIMO26_SUDO_PW  sudo password used by the ritual (required for start/restart).
#                   Leave it unset on hosts with passwordless sudo and run
#                   `./start.sh` directly instead.
#   MIMO26_WORKER   override the worker as user@host (default: WORKER_USER and
#                   the first entry of WORKER_IPS from .env).
#   MIMO26_REPO     override the kit directory (default: this script's directory).
set -uo pipefail

REPO="${MIMO26_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

WORKER="${MIMO26_WORKER:-${WORKER_USER:-$USER}@${WORKER_IPS%%,*}}"
RITUAL='sync; echo 3 > /proc/sys/vm/drop_caches; sysctl -q -w vm.swappiness=0'
cmd="${1:-status}"

mem_available() {  # $1 = user@host, or empty for the local node
  if [ -n "${1:-}" ]; then
    ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$1" \
      "awk '/MemAvailable/{printf \"%.1f GiB available\", \$2/1048576}' /proc/meminfo" 2>/dev/null
  else
    awk '/MemAvailable/{printf "%.1f GiB available", $2/1048576}' /proc/meminfo
  fi
}

ritual() {
  if [ -z "${MIMO26_SUDO_PW:-}" ]; then
    echo "[mimo26] MIMO26_SUDO_PW is not set; refusing to start without the" >&2
    echo "         pre-launch cache ritual (SGLang would size its KV pool from" >&2
    echo "         a page cache left over from the previous run)." >&2
    echo "         Set MIMO26_SUDO_PW, or run ./start.sh serve directly." >&2
    exit 1
  fi
  echo "[mimo26] pre-launch ritual: sync + drop_caches + swappiness=0 (both nodes)"
  # The password goes in on stdin, so it never appears in argv or in `ps`.
  printf '%s\n' "$MIMO26_SUDO_PW" | sudo -S -p '' sh -c "$RITUAL" 2>/dev/null
  ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "sudo -S -p '' sh -c '$RITUAL'" \
    <<<"$MIMO26_SUDO_PW" 2>/dev/null
  echo "[mimo26] head:   $(mem_available)"
  echo "[mimo26] worker: $(mem_available "$WORKER")"
}

cd "$REPO" || { echo "no kit at $REPO" >&2; exit 1; }

case "$cmd" in
  start)   ritual; exec ./start.sh serve ;;
  restart) ./start.sh stop || true; ritual; exec ./start.sh serve ;;
  stop)    exec ./start.sh stop ;;
  status)  exec ./start.sh status ;;
  logs)    shift || true; exec ./start.sh logs "$@" ;;
  smoke)   exec ./start.sh smoke ;;
  doctor)  exec ./start.sh doctor ;;
  download) exec ./start.sh download ;;
  *) echo "usage: $0 [start|restart|stop|status|logs [worker|N]|smoke|doctor|download]" >&2; exit 2 ;;
esac
