for BASELINE in "dummy"; do
  for SEED in 42; do
    for BENCHMARK in "karmabench"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED
    done
  done
done