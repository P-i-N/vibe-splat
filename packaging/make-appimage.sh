#!/usr/bin/env bash
# Packs a Linux release build into splat-x86_64.AppImage: the program, a
# desktop entry, an icon, and the licences. Run from the repository root:
#
#   packaging/make-appimage.sh [build dir] [output]
#
# Needs appimagetool; set APPIMAGETOOL to use one already on the machine,
# otherwise it is downloaded. The icon is rendered by splat itself, so this
# needs a Vulkan device (lavapipe will do).
set -euo pipefail

build=${1:-build/linux-release}
out=${2:-splat-x86_64.AppImage}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
appdir="$work/splat.AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/doc/splat"

cp "$build/src/splat" "$appdir/usr/bin/"
cp packaging/splat.desktop "$appdir/"
cp README.md LICENSE "$appdir/usr/share/doc/splat/"
if [ -f "$build/THIRD_PARTY_NOTICES.txt" ]; then cp "$build/THIRD_PARTY_NOTICES.txt" "$appdir/usr/share/doc/splat/"; fi

"$build/src/splat" make-test-scene "$work/icon.ply" --count 30000 > /dev/null
"$build/src/splat" render "$work/icon.ply" -o "$appdir/splat.png" --width 256 --height 256 --pitch 35 > /dev/null

# Started from a desktop or a file manager without arguments, it opens the viewer.
cat > "$appdir/AppRun" <<'APPRUN'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
if [ $# -eq 0 ]; then exec "$here/usr/bin/splat" view; fi
exec "$here/usr/bin/splat" "$@"
APPRUN
chmod +x "$appdir/AppRun"

tool=${APPIMAGETOOL:-}
if [ -z "$tool" ]; then
  tool="$work/appimagetool"
  curl -fsSL -o "$tool" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$tool"
fi
# --appimage-extract-and-run: works without FUSE, as on CI runners.
ARCH=x86_64 "$tool" --appimage-extract-and-run "$appdir" "$out"
echo "Wrote $out"
