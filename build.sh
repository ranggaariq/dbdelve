#!/usr/bin/env bash
# Builds DBDelve from this checkout as the dev variant: the same profiles,
# passwords and support directory as the bundled dev build, never the
# installed app's. Release is the default because it is the build whose
# typing and rendering feel is honest; a debug build runs the same logic an
# order of magnitude slower and will read as lag it does not have.
#
#   ./build.sh              release build
#   ./build.sh --debug      debug build, for fast visual iteration only
#
# `./run.sh` execs what this produces and builds nothing itself.
set -euo pipefail

cd "$(dirname "$0")"

# The variant is exported for the build, so every step of the pipeline sees
# the same environment the binary will run under.
export DBDELVE_VARIANT=dev
case "${1:-}" in
  "" | --release) cargo build --release ;;
  --debug) cargo build ;;
  *) echo "usage: ./build.sh [--release|--debug]" >&2; exit 2 ;;
esac
