#!/usr/bin/env bash
# Execs the binary ./build.sh produced, as the dev variant: the same
# profiles, passwords and support directory as the bundled dev build, never
# the installed app's. There is deliberately no build step here — running
# launches instantly and deterministically, exactly what is on disk.
#
#   ./run.sh                the release binary
#   ./run.sh --debug        the debug binary
#
# If the binary is missing or feels stale, build first: ./build.sh
set -euo pipefail

cd "$(dirname "$0")"

# The variant selects the support directory at run time; export it so the
# app cannot fall back to the installed release's data.
export DBDELVE_VARIANT=dev
case "${1:-}" in
  "" | --release) profile=release ;;
  --debug) profile=debug ;;
  *) echo "usage: ./run.sh [--release|--debug]" >&2; exit 2 ;;
esac

binary="./target/$profile/dbdelve"
if [ ! -x "$binary" ]; then
  echo "no binary at $binary -- build first: ./build.sh" >&2
  exit 1
fi

# exec, so Ctrl-C and a window close reach the app directly rather than
# through this script.
exec "$binary"
