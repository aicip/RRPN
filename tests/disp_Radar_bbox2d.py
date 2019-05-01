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

from visualization import draw_xywh_bbox, draw_points
from detectron.utils.io import load_object
import detectron.datasets.dataset_catalog as dataset_catalog
from pycocotools_plus.coco import COCO_PLUS
from rrpn_generator import rrpn_loader


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Display images from dataset with 2D boundig box and Radar detections.')
    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../output/datasets/nucoco_sw_fb/annotations/instances_val.json')
    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../output/datasets/nucoco_sw_fb/val')
    parser.add_argument('--out_dir', dest='out_dir',
                        help='Output directory',
                        default='../output/results/detections')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # Load the nucoco dataset
    ann_file = args.ann_file
    img_dir = args.imgs_dir
    coco = COCO_PLUS(ann_file, img_dir)
    #img_ids=[10000323,10000193]
    img_ids=[10009844,10026493]
    # print(coco.imgs)
    # input('something')

    for img_id in img_ids:
        print(img_id)
        #fig, ax = plt.subplots( nrows=1, ncols=1 )
        points = coco.imgToPointcloud[img_id]['points']
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        # bboxes = anns['bbox']

        img_path = os.path.join(img_dir, coco.imgs[img_id]["file_name"])
        img = cv2.imread(img_path)
        #img = draw_points(img, points, color=(0,0,200), radius=8, thickness=-1)
        
        fig = plt.figure(figsize=(9, 16))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        fig.add_axes(ax)

        coco.showImgAnn(img, anns, bbox_only=True, ax=ax)

        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], c='r', s=5)

        out_name = os.path.join(args.out_dir, '{}.pdf'.format(img_id))
        fig.savefig(out_name, bbox_inches='tight', pad_inches = 0)
        plt.close(fig)
