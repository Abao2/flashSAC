#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEX4D_CONFIG=dex4d_m6_wuji_stage12_flashsac exec "${repo_root}/scripts/run_dex4d.sh" "$@"
