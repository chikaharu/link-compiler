#!/bin/sh
set -eu
cd "$(dirname "$0")"
if command -v rustc >/dev/null 2>&1; then
  echo 'Building genuine Rust-native binaries via rustc'
  for name in identity rectangular phases; do
    python3 -m linkc "examples/$name.link" -o "examples/$name.lbc"
    python3 -m linkc rust "examples/$name.lbc" --emit-rust "examples/$name.rs" --rust-bin "examples/$name-rust"
  done
else
  echo 'rustc not installed; genuine Rust binaries cannot be produced here.' >&2
  echo 'Install rustc and rerun this script; C fallback binaries are separate.' >&2
  exit 2
fi
