#!/usr/bin/env bash
#
# Build the Zenodo upload bundle for ExSASCA.
#
# Produces  dist/exsasca-v<VERSION>.zip  containing every git-tracked file plus
# the compiled SDD artifacts, so that the archived record is self-contained and
# does not depend on the Google Drive link surviving.
#
# Usage:  ./make_zenodo_bundle.sh [--no-sdd]
#
set -euo pipefail

cd "$(dirname "$0")"

INCLUDE_SDD=1
[[ "${1:-}" == "--no-sdd" ]] && INCLUDE_SDD=0

VERSION="$(sed -n 's/^version: *"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' CITATION.cff | head -1)"
[[ -n "$VERSION" ]] || { echo "error: could not read version from CITATION.cff" >&2; exit 1; }

NAME="exsasca-v${VERSION}"
DIST="dist"
STAGE="${DIST}/${NAME}"
ZIP="${DIST}/${NAME}.zip"

echo "==> Building ${NAME}"

# --- sanity checks -----------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
    echo "warning: working tree is dirty; the bundle ships committed content only." >&2
fi
if ! git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
    echo "warning: tag v${VERSION} does not exist yet. Create it with:" >&2
    echo "           git tag -a v${VERSION} -m 'ExSASCA v${VERSION}' && git push origin v${VERSION}" >&2
fi

# --- stage tracked sources ---------------------------------------------------
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
git archive HEAD | tar -x -C "$STAGE"
echo "    staged $(find "$STAGE" -type f | wc -l) tracked files"

# --- compiled SDD ------------------------------------------------------------
if [[ "$INCLUDE_SDD" == "1" ]]; then
    SDD="compilation/compiled_sdd_g_254/out.sdd"
    if [[ ! -f "$SDD" ]]; then
        echo "==> $SDD not present; downloading (~1 GB)"
        ( cd compilation/compiled_sdd_g_254 && python download_sdd.py )
    fi
    [[ -f "$SDD" ]] || { echo "error: $SDD still missing after download" >&2; exit 1; }
    cp "$SDD" "$STAGE/$SDD"
    echo "    included $SDD ($(du -h "$SDD" | cut -f1))"
else
    echo "==> skipping compiled SDD (--no-sdd)"
fi

# --- zip ---------------------------------------------------------------------
( cd "$DIST" && zip -qr "${NAME}.zip" "$NAME" )
rm -rf "$STAGE"

echo
echo "==> $ZIP  ($(du -h "$ZIP" | cut -f1))"
sha256sum "$ZIP" | tee "${ZIP}.sha256"
echo
echo "Upload this ZIP at https://zenodo.org/uploads/new together with the"
echo "metadata in .zenodo.json (resource type: Software)."
