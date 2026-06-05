#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/reproduce_defectflywheel_common.sh" "wfdd" "WFDD" "WFDD" "wfdd"
