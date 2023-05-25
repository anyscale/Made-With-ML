#!/bin/bash

# Evaluate model
set -xe
if [[ -z "${RUN_ID}" ]]; then  # if RUN_ID is set use it, else get the best run
    RUN_ID=$(python -c "from madewithml.predict import get_best_run_id as g; print(g('${EXPERIMENT_NAME}', 'val_loss', 'ASC'))")
fi
HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/madewithml/holdout.csv"
python src/madewithml/evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc $HOLDOUT_LOC \
    --num-cpu-workers 2 \
    --results-fp ./evaluation_results.json

# Print evaluation results
set +x
echo "####EVAL_OUT####"
cat ./evaluation_results.json
echo "####EVAL_END####"
