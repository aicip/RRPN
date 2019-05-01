# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

import _init_paths
import cv2
import os
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from pycocotools_plus.coco import COCO_PLUS
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud
from nuscenes_utils.radar_utils import *


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Explore NuCOCO')
    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../output/datasets/nucoco_sw_f/annotations/instances_train.json')
    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../output/datasets/nucoco_sw_f/train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    coco = COCO_PLUS(args.ann_file, args.imgs_dir)
    #fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    pprint(coco.cats)
    input('something')

    # Open samples from the NuScenes dataset
    for i in tqdm(range(0, len(coco.dataset['images']))):
        image_id = coco.dataset['images'][i]['id']
        ann_ids = coco.getAnnIds(image_id)
        anns = coco.loadAnns(ann_ids)

        # print('Image ID: {}'.format(image_id))
        # print('Ann IDs: {}'.format(ann_ids))
        # print(anns[0])

        img_path = os.path.join(args.imgs_dir, coco.imgs[image_id]["file_name"])
        img = cv2.imread(img_path)
        coco.showImgAnn(img, anns, bbox_only=True)
        input('something')
