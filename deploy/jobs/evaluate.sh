#!/bin/bash

# Evaluate model
set -xe
if [[ -z "${run_id}" ]]; then  # if RUN_ID is set use it, else get the best run
    run_id=$(python -c "from madewithml.predict import get_best_run_id as g; print(g('${experiment_name}', 'val_loss', 'ASC'))")
fi
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python src/madewithml/evaluate.py \
    --run-id $run_id \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --results-fp ./evaluation_results.json

# Print evaluation results
set +x
echo "####EVAL_OUT####"
cat ./evaluation_results.json
echo "####EVAL_END####"
