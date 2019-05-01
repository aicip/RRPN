#!/usr/bin/env python

import _init_paths
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import pickle
import argparse

from visualization import draw_xyxy_bbox, draw_points
from detectron.utils.io import load_object
import detectron.datasets.dataset_catalog as dataset_catalog
from pycocotools_plus.coco import COCO_PLUS
from rrpn_generator import rrpn_loader


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Test the object proposals file')
    parser.add_argument('--proposals_file', dest='proposals_file',
                        help='Proposals file',
                        default='../output/proposals/nucoco_sw_f/rrpn_v4/proposals_nucoco_train.pkl')
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='Dataset name according to dataset_catalog',
                        default='nucoco_train')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # fig = plt.figure(figsize=(16, 6))

    # Load the nucoco dataset
    dataset_name = args.dataset_name
    ann_file = dataset_catalog.get_ann_fn(dataset_name)
    img_dir = dataset_catalog.get_im_dir(dataset_name)
    coco = COCO_PLUS(ann_file, img_dir)

    # Load the proposals
    proposals = rrpn_loader(args.proposals_file)

    for i in range(1, len(coco.dataset['images']),10):
        fig = plt.figure(figsize=(16, 6))
        img_id = coco.dataset['images'][i]['id']
        scores = proposals[img_id]['scores']
        boxes = proposals[img_id]['boxes']
        points = coco.imgToPointcloud[img_id]['points']

        img_path = os.path.join(img_dir, coco.imgs[img_id]["file_name"])
        # print(img_path)

        img = np.array(plt.imread(img_path))
        img = draw_points(img, points, color=(0,0,255), radius=5, thickness=-1, format='RGB')
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img)

        img = draw_xyxy_bbox(img, list(boxes), lineWidth=3)
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(img)

        # plt.show(block=False)
        plt.show()
        plt.pause(0.1)
        plt.cla()
        # img = draw_points(img, points, color=(0,0,255), radius=5, thickness=-1)
