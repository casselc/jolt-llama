#!/usr/bin/env bash
# Run the jolt-llama suite. Every path comes from the environment; nothing in
# this file is specific to the machine it was written on.
#
#   JOLT_LLAMA_LIB    path to libjolt_llama.so   (build it: see native/Makefile)
#   JOLT_LLAMA_MODEL  path to a .gguf model
#   JOLT_CACHE_DIR    writable AOT cache         (optional but recommended)
#
# Optional: source a gitignored .env.local from the repo root to set these.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env.local ] && . ./.env.local

: "${JOLT_LLAMA_LIB:?set JOLT_LLAMA_LIB to libjolt_llama.so}"
: "${JOLT_LLAMA_MODEL:?set JOLT_LLAMA_MODEL to a .gguf file}"

[ -f "$JOLT_LLAMA_LIB" ]   || { echo "no such library: $JOLT_LLAMA_LIB" >&2; exit 2; }
[ -f "$JOLT_LLAMA_MODEL" ] || { echo "no such model: $JOLT_LLAMA_MODEL" >&2; exit 2; }

only="${1:-all}"
fail=0

# THE hard requirement: the core library must run on stock Jolt. This runs the
# example with no alias at all, so jolt-hegel is not on the classpath and the
# :test alias contributes nothing. If jolt-llama ever acquires a runtime
# dependency, this is what catches it.
run_stock() {
  echo "=== stock Jolt check (no alias, no jolt-hegel on the classpath) ==="
  if jolt run examples/rank_actions.clj >/dev/null 2>&1; then
    echo "  ok   core library runs on stock Jolt with :deps {}"
  else
    echo "  FAIL core library did not run without the :test alias"
    fail=1
  fi
}

# The C reference. Run FIRST and separately: it exercises the shim with no Jolt
# in the picture, so when it and the Jolt tests disagree the binding is what is
# wrong. Skipped silently if it has not been built.
run_smoke() {
  if [ -x native/smoke ]; then
    echo "=== native/smoke (independent C reference, no Jolt) ==="
    native/smoke "$JOLT_LLAMA_MODEL" 2>&1 | grep -Ev '^(llama_|load_|print_info|init_tokenizer|sched_|graph_|state_|~llama|ggml_)' || true
  else
    echo "=== native/smoke: not built, skipping ==="
  fi
}

# The script-style milestone tests do their work at load time and have no
# -main, so they are loaded as files rather than entered through -m.
run_file() {
  echo
  echo "=== $1 ==="
  if jolt -A:test run "$2" 2>&1 | grep -Ev '^(llama_|load_|print_info|init_tokenizer|sched_|graph_|state_|~llama|ggml_)'; then :; fi
  # shellcheck disable=SC2181
  [ "${PIPESTATUS[0]:-0}" -eq 0 ] || fail=1
}

run_main() {
  echo
  echo "=== $1 ==="
  if jolt -A:test -m "$2" 2>&1 | grep -Ev '^(llama_|load_|print_info|init_tokenizer|sched_|graph_|state_|~llama|ggml_)'; then :; fi
  [ "${PIPESTATUS[0]:-0}" -eq 0 ] || fail=1
}

case "$only" in
  all|stock)  run_stock ;;&
  all|m0)     [ "$only" = all ] && run_smoke
              run_file "M0 vertical slice" test/jolt/m0_test.clj ;;&
  all|m1)     run_file "M1 exact spine" test/jolt/m1_exact_spine_test.clj ;;&
  all|alt)    run_file "M1 alternating domains" test/jolt/m1_alternating_test.clj ;;&
  all|hegel)  run_main "jolt-hegel properties" jolt.hegel-properties ;;&
  probes)     for f in test/jolt/probe_*.clj; do run_file "probe $(basename "$f")" "$f"; done ;;
esac

echo
[ "$fail" -eq 0 ] && echo "SUITE OK" || echo "SUITE FAILURES"
exit "$fail"
