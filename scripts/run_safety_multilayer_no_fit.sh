#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_multilayer_variant.sh" --mode no_fit "$@"
