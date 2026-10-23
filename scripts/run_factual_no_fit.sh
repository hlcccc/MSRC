#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_acceptance_variant.sh" --variant factual_no_fit "$@"
