#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
#
# Installs the Lance data source into a Sail checkout:
#
#   1. copies this crate's sources to `crates/sail-lance` in that checkout,
#      with a manifest that uses Sail's workspace dependencies, and
#   2. registers the data source and its physical planner with Sail's session.
#
# Usage: scripts/install-into-sail.sh <path-to-sail-checkout>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <path-to-sail-checkout>" >&2
    exit 2
fi

sail=$(cd "$1" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lance=$(cd "${here}/../.." && pwd)

if [[ ! -f "${sail}/crates/sail-session/src/formats.rs" ]]; then
    echo "error: ${sail} does not look like a Sail checkout" >&2
    exit 1
fi

crate="${sail}/crates/sail-lance"
mkdir -p "${crate}"
rm -rf "${crate}/src" "${crate}/tests"
cp -r "${here}/src" "${crate}/src"
cp -r "${here}/tests" "${crate}/tests"

# The manifest differs from the standalone one: inside the Sail workspace the
# Sail crates and their DataFusion and Arrow versions come from the workspace
# rather than from crates.io and a pinned Git revision.
cat > "${crate}/Cargo.toml" <<EOF
[package]
name = "sail-lance"
version = { workspace = true }
edition = { workspace = true }

[dependencies]
sail-common-datafusion = { path = "../sail-common-datafusion" }

datafusion = { workspace = true }
datafusion-common = { workspace = true }
datafusion-expr = { workspace = true }
arrow = { workspace = true, features = ["ffi"] }

# Lance is built against Arrow 58, which is why it brings its own Arrow.
lance = { path = "${lance}/rust/lance", default-features = false }
lance-file = { path = "${lance}/rust/lance-file" }
lance-table = { path = "${lance}/rust/lance-table" }
arrow-lance = { package = "arrow", version = "58.3", default-features = false, features = ["ffi"] }

async-trait = { workspace = true }
futures = { workspace = true }
tokio = { workspace = true }
url = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }

[lints]
workspace = true
EOF

patch="${here}/patches/register-lance-data-source.patch"
if git -C "${sail}" apply --check "${patch}" 2>/dev/null; then
    git -C "${sail}" apply "${patch}"
    echo "registered the Lance data source with sail-session"
else
    echo "note: ${patch} did not apply; sail-session may already be patched" >&2
fi

cat <<EOF

Installed into ${sail}/crates/sail-lance.

Next:
  cd ${sail}
  cargo test -p sail-lance
  cargo run --bin sail -- spark server --port 50051
EOF
