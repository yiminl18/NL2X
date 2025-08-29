for BASELINE in "gpt4o_zero_shot"; do
  for SEED in 42; do
    for BENCHMARK in "kramabench"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED
    done
  done
done