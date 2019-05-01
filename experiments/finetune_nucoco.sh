#! /bin/sh

# Finetune Fast-RCNN (Pretrained on COCO2017) on the NuCOCO dataset.

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Set the parameters
#CUDA_VISIBLE_DEVICES=1,3
DATASET='nucoco_sw_fb'
PROPOSAL_METHOD='eb'
CFG="$ROOT_DIR/experiments/cfgs/fast_rcnn_X-101-32x8d-FPN_1x_finetune_nucoco.yaml"
TRAIN_WEIGHTS="$ROOT_DIR/output/models/X_101_32x8d_FPN_1x_original_RW/model_final.pkl"
OUT_DIR="$ROOT_DIR/output/models/X_101_32x8d_FPN_1x_ft30000_nucoco_sw_fb_eb"

##------------------------------------------------------------------------------
TRAIN_PROP_FILES="('$ROOT_DIR/output/proposals/$DATASET/$PROPOSAL_METHOD/proposals_nucoco_train.pkl',)"
TEST_PROP_FILES="('$ROOT_DIR/output/proposals/$DATASET/$PROPOSAL_METHOD/proposals_nucoco_val.pkl',)"
LOG_FILE="$OUT_DIR/train_log.txt"
LOG_FILE_MAT="$OUT_DIR/train_log.mat"
RES_DIR="$OUT_DIR/results"

set -e
mkdir -p $OUT_DIR
mkdir -p $RES_DIR
ln -sfn $ROOT_DIR/output/datasets/$DATASET $ROOT_DIR/output/nucoco
cp $CFG $OUT_DIR

echo "INFO: Starting training..."
cd $ROOT_DIR/detectron
python tools/train_net.py \
--cfg $CFG \
OUTPUT_DIR $OUT_DIR \
TRAIN.PROPOSAL_FILES $TRAIN_PROP_FILES \
TRAIN.WEIGHTS $TRAIN_WEIGHTS \
TEST.PROPOSAL_FILES $TEST_PROP_FILES | tee $LOG_FILE
echo "INFO: Trainig log saved to: $LOG_FILE"

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
