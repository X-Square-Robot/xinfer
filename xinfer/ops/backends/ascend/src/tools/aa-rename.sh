#!/bin/bash
# aa-rename.sh — rename the delivered AA-resample real6 asset object's xcompute_aot_ launcher
# symbols to xinfer_aot_ (the device-kernel registration is symbol-address based, so the rename is
# internally consistent; build-libraries.sh carries the full asset story). The real6 objects are
# assets, never regenerated.
#
#   aa-rename.sh <asset.o> <renamed.o>
set -euo pipefail

SRC=$1
DST=$2
MAP="$DST.map"

mkdir -p "$(dirname "$DST")"
nm "$SRC" | awk '{print $3}' | grep xcompute_aot_ | sort -u \
  | while read -r s; do printf '%s %s\n' "$s" "${s/xcompute_aot_/xinfer_aot_}"; done > "$MAP"
objcopy --redefine-syms="$MAP" "$SRC" "$DST"
