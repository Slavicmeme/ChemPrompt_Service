#!/usr/bin/env bash
# docker compose for this project: compose file in config/, paths and .env relative to the project root.
#   bin/compose.sh up -d --build      bin/compose.sh ps      bin/compose.sh logs -f api      bin/compose.sh down
#
# When REPRESENTATION_MODE is http (default; mock only for development), the internal aya service (gpu profile) is enabled automatically and
# the GPUs listed in AYA_GPUS (e.g. "0" or "0,1") are attached to it through a generated override file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

env_get() {  # value of KEY in .env (last match, quotes stripped), empty if missing
  [ -f "$ROOT/.env" ] || return 0
  sed -n "s/^$1=//p" "$ROOT/.env" | tail -n 1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

# create <DATA_DIR>/storage and <DATA_DIR>/llm as the calling user before Docker would create them as root
case "${1:-}" in up|run|create|start|restart)
  data="${DATA_DIR:-$(env_get DATA_DIR)}"; data="${data:-./data}"
  case "$data" in /*) ;; *) data="$ROOT/$data" ;; esac
  mkdir -p "$data/storage" "$data/llm"
esac

files=(-f "$ROOT/config/docker-compose.yml")
mode="${REPRESENTATION_MODE:-$(env_get REPRESENTATION_MODE)}"; mode="${mode:-http}"   # default: real Aya
if [ "$mode" = "http" ]; then
  export COMPOSE_PROFILES="${COMPOSE_PROFILES:+$COMPOSE_PROFILES,}gpu"
  gpus="${AYA_GPUS:-$(env_get AYA_GPUS)}"; gpus="${gpus:-0}"
  ids=""
  IFS=',' read -ra list <<< "$gpus"
  for g in "${list[@]}"; do
    g="$(echo "$g" | tr -d '[:space:]')"
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "AYA_GPUS must be GPU numbers like 0 or 0,1 (got '$gpus')" >&2; exit 1; }
    ids+="${ids:+, }\"$g\""
  done
  override="$(mktemp "${TMPDIR:-/tmp}/chemprompt-gpu.XXXXXX.yml")"
  trap 'rm -f "$override"' EXIT
  cat > "$override" <<EOF
services:
  aya:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [${ids}]
              capabilities: [gpu]
EOF
  files+=(-f "$override")
fi

docker compose --project-directory "$ROOT" "${files[@]}" "$@"
