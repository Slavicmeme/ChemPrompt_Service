#!/usr/bin/env bash
# Container entrypoint: run the service as the owner of the mounted storage folder (${DATA_DIR}/storage on the
# host), so files created by the app belong to the host user who created that folder.
# The LLM folder (${DATA_DIR}/llm = HF_HOME, aya service) is given to the same owner when Docker created it.
set -euo pipefail
STORE="${MODEL_STORAGE_ROOT:-/data/storage}"

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$STORE"
  [ -n "${HF_HOME:-}" ] && mkdir -p "$HF_HOME"
  owner="$(stat -c '%u:%g' "$STORE")"
  if [ "$owner" = "0:0" ]; then          # folder auto-created by Docker (root): use the LLM folder's owner, else the image user
    if [ -n "${HF_HOME:-}" ] && [ "$(stat -c '%u:%g' "$HF_HOME")" != "0:0" ]; then
      owner="$(stat -c '%u:%g' "$HF_HOME")"
    else
      owner="$(id -u "$MAMBA_USER"):$(id -g "$MAMBA_USER")"
    fi
    chown "$owner" "$STORE"
  fi
  # the LLM folder must be writable by the same account (first start downloads the model into it)
  if [ -n "${HF_HOME:-}" ] && [ "$(stat -c '%u:%g' "$HF_HOME")" != "$owner" ]; then
    chown -R "$owner" "$HF_HOME"
  fi
  exec setpriv --reuid="${owner%:*}" --regid="${owner#*:}" --clear-groups \
       env HOME=/tmp USER=app /usr/local/bin/_entrypoint.sh "$@"
fi
exec /usr/local/bin/_entrypoint.sh "$@"
