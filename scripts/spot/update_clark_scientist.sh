#!/usr/bin/env bash
# Build and atomically install the canonical Clark Scientist/Engineer worker.
set -euo pipefail

CLARK_REPO="${CLARK_REPO_DIR:-/workspace/git/clark}"
DESKTOP_REPO="${CLARK_DESKTOP_DIR:-/workspace/git/clark-desktop}"
PRODUCT="${CLARK_PRODUCT_DIR:-${CLARK_REPO}/crates/clark-desktop-product}"
AUTORESEARCH="${CLARK_AUTORESEARCH_DIR:-/workspace/git/clark-autoresearch}"
COMPACTION="${CLARK_COMPACTION_DIR:-/workspace/git/clark-agent-compaction}"
STATE_ROOT="${QUANT_STATE_ROOT:-/workspace/quant-state}"
BUILD_ROOT="${CLARK_SCIENTIST_BUILD_ROOT:-/workspace/build-cache/clark-scientist}"
RUSTUP_ROOT="${CLARK_RUSTUP_ROOT:-/workspace/rustup}"
CARGO_ROOT="${CLARK_CARGO_ROOT:-/workspace/cargo}"
BUILD_GIT_CONFIG="${STATE_ROOT}/clark-scientist-gitconfig"

test -f "${PRODUCT}/Cargo.toml"
test -f "${DESKTOP_REPO}/Cargo.toml"
test -f "${AUTORESEARCH}/Cargo.toml"
test -f "${COMPACTION}/Cargo.toml"
mkdir -p "${STATE_ROOT}" "${BUILD_ROOT}" "${RUSTUP_ROOT}" "${CARGO_ROOT}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl pkg-config libssl-dev

export RUSTUP_HOME="${RUSTUP_ROOT}"
export CARGO_HOME="${CARGO_ROOT}"
export PATH="${CARGO_HOME}/bin:${PATH}"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
fi
rustup toolchain install stable --profile minimal
rustup default stable

git config --file "${BUILD_GIT_CONFIG}" protocol.file.allow always
git config --file "${BUILD_GIT_CONFIG}" \
  url."file://${COMPACTION}".insteadOf \
  https://github.com/clark-labs-inc/clark-agent-compaction
# The staged dependency repositories remain owned by the SSH operator while
# this installer runs as root. Trust only these exact local worktrees and their
# Git directories; do not weaken Git's global ownership checks.
git config --file "${BUILD_GIT_CONFIG}" --unset-all safe.directory 2>/dev/null || true
for safe_path in \
  "${AUTORESEARCH}" "${AUTORESEARCH}/.git" \
  "${COMPACTION}" "${COMPACTION}/.git"
do
  git config --file "${BUILD_GIT_CONFIG}" --add safe.directory "${safe_path}"
done
export GIT_CONFIG_GLOBAL="${BUILD_GIT_CONFIG}"
export CLARK_SCIENTIST_TARGET_DIR="${BUILD_ROOT}/target"
export CLARK_SCIENTIST_OUTPUT_DIR="${BUILD_ROOT}/binaries"

"${PRODUCT}/scripts/stage-scientist-worker.sh" \
  --target x86_64-unknown-linux-gnu \
  --autoresearch-source "${AUTORESEARCH}" \
  --profile release

WORKER="${CLARK_SCIENTIST_OUTPUT_DIR}/clark-code-headless-x86_64-unknown-linux-gnu"
SELF_TEST="$(${WORKER} --self-test)"
EXPECTED_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "${PRODUCT}/Cargo.toml" | head -n 1)"
ACTUAL_VERSION="$(jq -r '.worker_version' <<<"${SELF_TEST}")"
if [[ "${ACTUAL_VERSION}" != "${EXPECTED_VERSION}" ]]; then
  echo "worker version ${ACTUAL_VERSION} does not match Clark ${EXPECTED_VERSION}" >&2
  exit 1
fi

install -m 0755 "${WORKER}" /usr/local/bin/clark-code-headless.new
mv /usr/local/bin/clark-code-headless.new /usr/local/bin/clark-code-headless
mkdir -p "${PRODUCT}/target/release"
install -m 0755 "${WORKER}" "${PRODUCT}/target/release/clark-code-headless"

WORKER_SHA="$(sha256sum "${WORKER}" | awk '{print $1}')"
source_revision() {
  local source="$1"
  local provenance_key="$2"
  local revision
  if [[ -f "${source}/.source-revision" ]]; then
    revision="$(tr -d '[:space:]' <"${source}/.source-revision")"
  elif [[ -d "${source}/.git" ]]; then
    revision="$(git -C "${source}" rev-parse HEAD)"
  else
    revision="$(python3 - "${provenance_key}" <<'PY'
import json
from pathlib import Path
import sys

receipt = json.loads(Path(
    '/workspace/clark-quantize/experiments/qwen38_27b_spot/source_provenance.json'
).read_text())
print(receipt[sys.argv[1]]['revision'])
PY
)"
  fi
  if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "invalid source revision for ${source}: ${revision}" >&2
    exit 1
  fi
  printf '%s\n' "${revision}"
}
PRODUCT_REVISION="$(source_revision "${CLARK_REPO}" clark_private_product)"
DESKTOP_REVISION="$(source_revision "${DESKTOP_REPO}" clark_desktop_foundation)"
AUTORESEARCH_REVISION="$(source_revision "${AUTORESEARCH}" clark_autoresearch)"
COMPACTION_REVISION="$(source_revision "${COMPACTION}" clark_agent_compaction)"
python3 - "${STATE_ROOT}/clark-scientist-update.json" "${PRODUCT_REVISION}" \
  "${DESKTOP_REVISION}" "${AUTORESEARCH_REVISION}" "${COMPACTION_REVISION}" \
  "${WORKER_SHA}" "${SELF_TEST}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

path = Path(sys.argv[1])
receipt = {
    'schema': 'clark_scientist_quant_update_v1',
    'clark_revision': sys.argv[2],
    'clark_desktop_revision': sys.argv[3],
    'autoresearch_revision': sys.argv[4],
    'compaction_revision': sys.argv[5],
    'worker_sha256': sys.argv[6],
    'self_test': json.loads(sys.argv[7]),
    'installed_path': '/usr/local/bin/clark-code-headless',
    'canonical_source': '/workspace/git/clark/crates/clark-desktop-product/scientist',
    'source_transport': 'clean Git archives plus locked dependency bundles transferred directly from local canonical repositories',
}
fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
with os.fdopen(fd, 'w') as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o644)
os.replace(temporary, path)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
