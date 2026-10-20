#!/usr/bin/env bash
# (installed in the image as /usr/local/bin/chemprompt)
# Run a backend module as the storage owner, e.g.  docker compose exec api chemprompt scripts.create_admin admin
cd /app/backend
exec /usr/local/bin/chemprompt-entrypoint.sh python -m "$@"
