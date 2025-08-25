for BASELINE in "mock_unify" "mock_docetl" "mock_lotus"; do
  for SEED in 42; do
    for BENCHMARK in "mock_dsbench" "mock_kramabench" "mock_crag"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED
    done
  done
done