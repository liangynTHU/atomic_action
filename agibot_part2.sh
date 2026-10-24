#!/usr/bin/env bash
set -euo pipefail
AGIBOT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$AGIBOT_REPO/scripts/agibot_part.sh" 2 "$@"
