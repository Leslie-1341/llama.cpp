#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec python3 "$ROOT/scripts/kv-paged-identity-controlled-ab-runner.py"
