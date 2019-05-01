#! /bin/sh
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

DATASET='nucoco_sw_f'

CFG="$ROOT_DIR/experiments/cfgs/fast_rcnn_R-50-C4_2x_finetune_nucoco.yaml"
LOG_FILE="$ROOT_DIR/output/models/train_log.txt"
OUT_DIR="$ROOT_DIR/output/models/R_50_C4_2x_ft40000_nucoco_sw_f"
cd $ROOT_DIR/detectron

python tools/test_net.py \
    --cfg $ROOT_DIR/experiments/cfgs/fast_rcnn_R-50-C4_2x_test_nucoco.yaml \
    OUTPUT_DIR $ROOT_DIR/output
