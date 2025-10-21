#!/bin/bash
for BASELINE in "abstract_step"; do
  for SEED in 42; do
    for BENCHMARK in "medication"; do
        python evaluate.py --baseline $BASELINE --benchmark $BENCHMARK --seed $SEED --max-samples 5 --validate-answer false --confirm --max-attempts 1 --verbose
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