#!/bin/bash
# Rename the original launcher's symbols after compiling its checked-in source.
# The device program and launch arguments are unchanged.
#
#   aa-rename.sh <freshly-compiled.o> <renamed.o>
set -euo pipefail

SRC=$1
DST=$2
MAP="$DST.map"

mkdir -p "$(dirname "$DST")"
nm "$SRC" | awk '{print $3}' | grep xcompute_aot_ | sort -u \
  | while read -r s; do printf '%s %s\n' "$s" "${s/xcompute_aot_/xinfer_aot_}"; done > "$MAP"
objcopy --redefine-syms="$MAP" "$SRC" "$DST"
