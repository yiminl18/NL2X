for BASELINE in "docetl"; do
  for SEED in 42; do
    for BENCHMARK in "dsbench"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED
    done
  done
done