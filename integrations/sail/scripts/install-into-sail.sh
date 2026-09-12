#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
#
# Installs the Lance table format into a Sail checkout:
#
#   1. copies the `sail-lance` crate to `crates/sail-lance` in that checkout, and
#   2. applies `patches/lance-table-format.patch`, which registers the format
#      with Sail's session and lets Sail's error conversion tolerate the
#      DataFusion `sql` feature that Lance turns on.
#
# The checkout must be at tag v0.7.1, the Sail release built against the same
# DataFusion and Arrow versions as Lance.
#
# Usage: scripts/install-into-sail.sh <path-to-sail-checkout>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <path-to-sail-checkout>" >&2
    exit 2
fi

sail=$(cd "$1" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "${sail}/crates/sail-session/src/formats.rs" ]]; then
    echo "error: ${sail} does not look like a Sail checkout" >&2
    exit 1
fi

crate="${sail}/crates/sail-lance"
rm -rf "${crate}"
mkdir -p "${crate}"
cp -r "${here}/sail-lance/." "${crate}/"

patch="${here}/patches/lance-table-format.patch"
if git -C "${sail}" apply --check "${patch}" 2>/dev/null; then
    git -C "${sail}" apply "${patch}"
    echo "registered the Lance table format with sail-session"
else
    echo "note: ${patch} did not apply; the checkout may already be patched" >&2
fi

cat <<EOF

Installed into ${sail}/crates/sail-lance.

Next:
  cd ${sail}
  cargo test -p sail-lance
  cargo run --bin sail -- spark server --port 50051
EOF
