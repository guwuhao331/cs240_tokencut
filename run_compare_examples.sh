#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ARCH="${ARCH:-vit_base}"
PATCH_SIZE="${PATCH_SIZE:-8}"

EXAMPLES=(
  "9fa1dbca4dab51c6aeb76b9113327bbc.jpg"
  "ex1.jpg"
  "IMG_2843.png"
  "VOC07_000012.jpg"
  "005519.jpg"
)

for name in "${EXAMPLES[@]}"; do
  image_path="examples/${name}"
  if [[ ! -f "$image_path" ]]; then
    echo "[skip] missing: $image_path" >&2
    continue
  fi

  echo "[run] $image_path (arch=$ARCH, patch_size=$PATCH_SIZE)"
  python compare_methods.py --image_path "$image_path" --arch "$ARCH" --patch_size "$PATCH_SIZE"
done

