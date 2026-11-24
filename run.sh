#!/usr/bin/env bash
# Runs DBDelve from this checkout as the dev variant: it shares the profiles,
# passwords and support directory with the bundled dev build and never touches
# the installed app's. Release is the default because it is the build whose
# typing and rendering feel is honest; a debug build runs the same logic an
# order of magnitude slower and will read as lag it does not have.
#
#   ./run.sh              release build, dev variant
#   ./run.sh --debug      debug build, for fast visual iteration only
set -euo pipefail

cd "$(dirname "$0")"

profile=release
cargo_flags=(--release)
case "${1:-}" in
  "" | --release) ;;
  --debug) profile=debug; cargo_flags=() ;;
  *) echo "usage: ./run.sh [--release|--debug]" >&2; exit 2 ;;
esac

# The variant is exported for the build too, so every step of the pipeline
# sees the same environment the binary will run under.
export DBDELVE_VARIANT=dev
cargo build "${cargo_flags[@]}"

# exec, so Ctrl-C and a window close reach the app directly rather than
# through this script.
exec "./target/$profile/dbdelve"
