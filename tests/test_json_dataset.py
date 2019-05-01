
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace
from detectron.datasets.json_dataset import JsonDataset
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils


dataset_name = 'nucoco_val'
proposal_file = '/ssd_scratch/mrnabati/RRPN/output/proposals/nucoco_sw_fb/rrpn_v5/proposals_nucoco_val.pkl'
proposal_limit = 2000


dataset = JsonDataset(dataset_name)
roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=proposal_limit
)