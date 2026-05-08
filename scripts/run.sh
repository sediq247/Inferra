set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

MODE="${1:-help}"

echo ""
echo "========================================="
echo " Inferra Runtime Launcher"
echo " Project Root: $PROJECT_ROOT"
echo " Mode: $MODE"
echo "========================================="
echo ""

run_cmd () {
  echo "[Inferra] Running: $*"
  "$@"
}

case "$MODE" in

  eval)
    run_cmd python -m evaluation.run_eval
    ;;

  bench)
    run_cmd python -m benchmarks.benchmark
    ;;

  stress)
    run_cmd python -m benchmarks.stress_test
    ;;

  replay)
    run_cmd python -m benchmarks.replay_engine
    ;;

  test)
    run_cmd pytest -q tests
    ;;

  *)
    echo "Usage:"
    echo "  ./scripts/run.sh eval"
    echo "  ./scripts/run.sh bench"
    echo "  ./scripts/run.sh stress"
    echo "  ./scripts/run.sh replay"
    echo "  ./scripts/run.sh test"
    echo ""
    exit 1
    ;;
esac

EXIT_CODE=$?

echo ""
echo "========================================="
echo " Done (exit code: $EXIT_CODE)"
echo "========================================="

exit $EXIT_CODE