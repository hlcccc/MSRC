#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_acceptance_variant.sh" --variant safety_no_fit "$@"
