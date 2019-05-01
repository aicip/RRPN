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

from visualization import draw_xyxy_bbox
from detectron.utils.io import load_object


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Display Selective Search proposals for a single image.')

    parser.add_argument('--image_file', dest='image_file',
                        help='Image file',
                        default='../output/image.png')

    parser.add_argument('--proposals_file', dest='proposals_file',
                        help='Proposals file',
                        default='../output/temp_proposals_ss.pkl')


    args = parser.parse_args()
    args.image_file = os.path.abspath(args.image_file)
    args.proposals_file = os.path.abspath(args.proposals_file)
    return args


if __name__ == '__main__':
    args = parse_args()
    fig = plt.figure(figsize=(16, 6))

    # Load the proposals
    proposals = load_object(args.proposals_file)

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

    img = np.array(plt.imread(args.image_file))
    p_boxes = proposals['boxes'][i]

    ## Plot proposals1
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img)

    ## Plot proposals2
    img2 = draw_xyxy_bbox(img, list(p_boxes), lineWidth=1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img2)

    # plt.show(block=False)
    plt.show()
    plt.pause(0.5)
    plt.clf()
