#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 STDOUT STDERR COMMAND [ARG ...]" >&2
  exit 2
fi

stdout_path=$1
stderr_path=$2
shift 2

exec "$@" >"$stdout_path" 2>"$stderr_path"
