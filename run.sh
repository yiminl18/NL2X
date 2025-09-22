#!/bin/bash
for BASELINE in "docetl"; do
  for SEED in 42; do
    for BENCHMARK in "cuad"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED --validate-answer false --confirm --max-attempts 1
    done
  done
done


# for BASELINE in "docetl" "lotus"; do
#   for SEED in 42; do
#     for BENCHMARK in "dsbench"; do
#         python evaluate.py \
#             --baseline $BASELINE \
#             --benchmark $BENCHMARK \
#             --seed $SEED \
#             --max-attempts 5 \
#             --validate-answer false \
#             --max-samples 10 \
#             --verbose
#     done
#   done
# done