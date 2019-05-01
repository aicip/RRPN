# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

import _init_paths
import os
import sys
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from tqdm import tqdm
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud
from nuscenes_utils.radar_utils import *
from datasets import nuscene_cat_to_coco
from visualization import draw_xywh_bbox

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Test the 3D to 2D bbox conversion')
    parser.add_argument('--nuscene_root', dest='dataroot',
                        help='NuScene dataroot',
                        default='../data/nuscenes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version='v0.1', dataroot=args.dataroot, verbose=True)
    fig = plt.figure(figsize=(16, 6))

    for i in tqdm(range(88, len(nusc.scene))):
        scene = nusc.scene[i]
        scene_rec = nusc.get('scene', scene['token'])
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        f_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT'])
        f_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])

        has_more_frames = True
        while has_more_frames:
            bboxes = []
            ax1 = fig.add_subplot(1,1,1)
            # ax2 = fig.add_subplot(1,2,2)
            camera_token = f_cam_rec['token']
            radar_token = f_rad_rec['token']

            ## FRONT CAM + RADAR
            impath, boxes, camera_intrinsic = nusc.get_sample_data(camera_token)
            points, coloring, image = nusc.explorer.map_pointcloud_to_image(radar_token,
                                                                        camera_token)
            points[2, :] = coloring

            # Plot the 3D boxes and create 2D COCO bboxes
            # for box in boxes:
            #     coco_cat, cat_id, coco_supercat = nuscene_cat_to_coco(box.name)
            #     if coco_cat is None:
            #         continue
            #     c = np.array(nusc.explorer.get_color(box.name)) / 255.0
            #     box.render(ax1, view=camera_intrinsic, normalize=True, colors=[c, c, c])
            #     bboxes.append(box.to_coco_bbox(camera_intrinsic, image.size))

            # # Plot the 3D boxes
            # nusc.render_sample_data(camera_token, ax=ax1)

            # Plot the 2D boxes
            img = np.array(plt.imread(impath))
            # img = draw_xywh_bbox(img, bboxes, format = 'RGB')
            ax1.imshow(img)

            # Plot Radar pointclouds
            ax1.scatter(points[0, :], points[1, :], c='r', s=20)
            # ax2.scatter(points[0, :], points[1, :], c=coloring, s=5)

            plt.show(block=False)
            # plt.show()
            plt.pause(0.3)
            plt.cla()

            # Get the next image/pointcloud in scene
            if not f_cam_rec['next'] == "" and not f_rad_rec['next'] == "":
                f_cam_rec = nusc.get('sample_data', f_cam_rec['next'])
                f_rad_rec = nusc.get('sample_data', f_rad_rec['next'])
            else:
                has_more_frames = False
