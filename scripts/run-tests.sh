#!/usr/bin/env bash
# Run the jolt-llama suite. Every path comes from the environment; nothing in
# this file is specific to the machine it was written on.
#
#   JOLT_LLAMA_LIB    path to libjolt_llama.so   (build it: see native/Makefile)
#   JOLT_LLAMA_MODEL  path to a .gguf model
#   JOLT_CACHE_DIR    writable AOT cache         (optional but recommended)
#
# Optional: source a gitignored .env.local from the repo root to set these.
set -uo pipefail

cd "$(dirname "$0")/.."
# .env.clean pins the PROMOTED llama.cpp build and wins when present. Sourcing
# .env.local unconditionally silently pointed LD_LIBRARY_PATH at an older
# llama.cpp than the shim was linked against, so the oracle loaded a different
# libllama than the one under test -- which is precisely the mismatch the
# runtime id exists to detect, arriving through the test harness instead.
if [ -f .env.clean ]; then . ./.env.clean
elif [ -f .env.local ]; then . ./.env.local
fi

: "${JOLT_LLAMA_LIB:?set JOLT_LLAMA_LIB to libjolt_llama.so}"
: "${JOLT_LLAMA_MODEL:?set JOLT_LLAMA_MODEL to a .gguf file}"

[ -f "$JOLT_LLAMA_LIB" ]   || { echo "no such library: $JOLT_LLAMA_LIB" >&2; exit 2; }
[ -f "$JOLT_LLAMA_MODEL" ] || { echo "no such model: $JOLT_LLAMA_MODEL" >&2; exit 2; }

only="${1:-all}"
fail=0
oracle_skipped=0

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
# THE ORACLE. Both programs emit ORACLE: lines carrying raw IEEE-754 bits, and
# this DIFFS them. Previously the smoke output was printed and eyeballed -- its
# exit status swallowed by `|| true`, a missing binary silently skipped, and
# nothing anywhere compared the two. "Agrees with the independent C reference"
# was a manual claim, not a gate.
#
# JOLT_LLAMA_ALLOW_NO_SMOKE=1 skips it for a local inner loop. The promotion
# target must never set it.
run_oracle() {
  echo "=== M0 oracle: native C vs Jolt, raw float bits ==="
  if [ ! -x native/smoke ]; then
    if [ "${JOLT_LLAMA_ALLOW_NO_SMOKE:-0}" = "1" ]; then
      # The escape hatch is for a local inner loop. It used to print one line
      # and let the run finish as "SUITE OK", so a promotion log could not be
      # told apart from a real one. It now marks the whole run.
      echo "  SKIPPED (JOLT_LLAMA_ALLOW_NO_SMOKE=1)"
      oracle_skipped=1
      return 0
    fi
    echo "  FAIL: native/smoke is not built; the oracle cannot run" >&2
    fail=1; return 0
  fi

  local c_out j_out
  c_out=$(mktemp); j_out=$(mktemp)
  if ! native/smoke "$JOLT_LLAMA_MODEL" 2>/dev/null | grep '^ORACLE ' | sed 's/^ORACLE //' > "$c_out"; then
    echo "  FAIL: native/smoke exited nonzero" >&2; fail=1
    rm -f "$c_out" "$j_out"; return 0
  fi
  jolt -A:test run test/jolt/m0_test.clj 2>/dev/null | grep '^ORACLE ' | sed 's/^ORACLE //' > "$j_out"

  if [ ! -s "$c_out" ] || [ ! -s "$j_out" ]; then
    echo "  FAIL: one side produced no oracle lines" >&2; fail=1
    rm -f "$c_out" "$j_out"; return 0
  fi

  if diff -u "$c_out" "$j_out" > /dev/null; then
    echo "  ok   C and Jolt agree on every oracle value ($(wc -l < "$c_out") lines,"
    echo "       including raw float bits for the whole top-k)"
  else
    echo "  FAIL: C and Jolt disagree:" >&2
    diff -u "$c_out" "$j_out" | head -30 >&2
    fail=1
  fi
  rm -f "$c_out" "$j_out"
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
  all|m0)     run_oracle
              run_file "M0 vertical slice" test/jolt/m0_test.clj ;;&
  all|m1)     run_file "M1 exact spine" test/jolt/m1_exact_spine_test.clj ;;&
  all|alt)    run_file "M1 alternating domains" test/jolt/m1_alternating_test.clj ;;&
  all|hegel)  run_main "jolt-hegel properties" jolt.hegel-properties ;;&
  probes)     for f in test/jolt/probe_*.clj; do run_file "probe $(basename "$f")" "$f"; done ;;
esac

echo
if [ "$fail" -ne 0 ]; then
  echo "SUITE FAILURES"
elif [ "$oracle_skipped" -eq 1 ]; then
  # Not a promotion result: the C-vs-Jolt oracle did not run.
  echo "SUITE OK (NOT A PROMOTION RUN -- the M0 oracle was skipped)"
else
  echo "SUITE OK"
fi
exit "$fail"
