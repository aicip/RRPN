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
from detectron.utils.io import load_object



def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Test the object proposals file')

    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../output/datasets/nucoco_sw_f/annotations/instances_train.json')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../output/datasets/nucoco_sw_f/train')

    parser.add_argument('--proposals_file1', dest='proposals_file1',
                        help='Proposals file 1',
                        default='../output/proposals/nucoco_sw_f/ss/proposals_nucoco_train.pkl')

    parser.add_argument('--proposals_file2', dest='proposals_file2',
                        help='Proposals file 2',
                        default='../output/proposals/nucoco_sw_f/eb/proposals_nucoco_train.pkl')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    fig = plt.figure(figsize=(16, 6))

    coco = COCO_PLUS(args.ann_file, args.imgs_dir)

    # Load the proposals
    proposals1 = load_object(args.proposals_file1)
    proposals2 = load_object(args.proposals_file2)
    num_imgs = len(proposals1['ids'])
    assert len(proposals1['ids']) == len(proposals2['ids']), \
        "Number of images are different in proposal files"

    # print(len(proposals1['indexes']))
    # print(proposals2['ids'])

    # np.set_printoptions(threshold=np.nan)
    # for i in range(20):
    #     print(proposals1['boxes'][i][0:10,:])
    #     print('---------------------------')
    #     print(proposals2['boxes'][i][0:10,:])
    #     input('something')

    # print('Nucoco proposal keys:')
    # for key, value in proposals1.items() :
    #     print(key, value)
    #
    # print('\nCoco proposal keys:')
    # for key, value in proposals2.items() :
    #     print(key, value)

    step = 20
    for i in range(0, num_imgs, step):
        img_id = proposals1['ids'][i]
        points = coco.imgToPointcloud[img_id]['points']
        img_path = os.path.join(args.imgs_dir, coco.imgs[img_id]["file_name"])
        img = np.array(plt.imread(img_path))

        assert proposals1['ids'][i] == proposals2['ids'][i], \
            "image IDs are different in proposals."
        p1_boxes = proposals1['boxes'][i]
        p2_boxes = proposals2['boxes'][i]

        ## Plot proposals1
        # img1 = draw_points(img, points, color=(0,0,255), radius=5, thickness=-1, format='RGB')
        img1 = draw_xyxy_bbox(img, list(p1_boxes)[0:20], lineWidth=2)
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img1)
        plt.axis('off')

        ## Plot proposals2
        img2 = draw_xyxy_bbox(img, list(p2_boxes)[0:20], lineWidth=2)
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(img2)
        plt.axis('off')

        plt.show(block=False)
        # plt.show()
        plt.pause(0.5)
        plt.clf()
