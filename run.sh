for ALGO in "unify" "docetl" "lotus"; do
  for SEED in 42; do
    for WORKLOAD in "dsbench" "kramabench" "crag"; do
        python evaluate.py --workload $WORKLOAD --algo $ALGO --seed $SEED
    done
  done
done