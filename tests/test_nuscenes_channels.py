# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

import _init_paths
import os
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from pycocotools_plus.coco import COCO_PLUS
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud
from nuscenes_utils.radar_utils import *
from nuscenes_utils.geometry_utils import BoxVisibility


#-------------------------------------------------------------------------------
if __name__ == '__main__':

    nusc = NuScenes(version='v0.1', dataroot='../data/nuscenes', verbose=True)
    # fig, axes = plt.subplots(1, 1, figsize=(16, 9))
    fig = plt.figure(figsize=(16, 6))

    skip_scenes = 30
    for scene in nusc.scene:
        skip_scenes -= 1
        if skip_scenes > 0:
            continue

        scene_rec = nusc.get('scene', scene['token'])
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        f_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT'])
        br_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_RIGHT'])
        bl_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_LEFT'])
        fr_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT_RIGHT'])
        fl_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT_LEFT'])

        f_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
        b_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_BACK'])
        fr_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT_RIGHT'])
        fl_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT_LEFT'])
        br_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_BACK_RIGHT'])
        bl_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_BACK_LEFT'])

        # ann_recs = [nusc.get('sample_annotation', token) for token in sample_rec['anns']]

        has_more_frames = True
        while has_more_frames:

            ## FRONT
            impath, boxes, camera_intrinsic = nusc.get_sample_data(f_cam_rec['token'])
            points_f, coloring_f, im1 = nusc.explorer.map_pointcloud_to_image(f_rad_rec['token'],
                                                                            f_cam_rec['token'])
            points_fr, coloring_fr, im2 = nusc.explorer.map_pointcloud_to_image(fr_rad_rec['token'],
                                                                            f_cam_rec['token'])
            points_fl, coloring_fl, im3 = nusc.explorer.map_pointcloud_to_image(fl_rad_rec['token'],
                                                                            f_cam_rec['token'])
            print(points_fr)
            print(points_fl)
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            # ax1.imshow(img1)
            ax1.imshow(im1)
            # nusc.render_sample_data(f_cam_rec['token'], True, ax=ax1)
            ax1.scatter(points_f[0, :], points_f[1, :], c=coloring_f, s=5)

            # nusc.render_sample_data(f_cam_rec['token'], True, ax=ax2)
            ax2.imshow(im2)
            ax2.scatter(points_fr[0, :], points_fr[1, :], c=coloring_fr, s=10)
            ax2.scatter(points_fl[0, :], points_fl[1, :], c=coloring_fl, s=10)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()


            ## BACK
            # impath, boxes, camera_intrinsic = nusc.get_sample_data(b_cam_rec['token'])
            # points_br, coloring_br, im = nusc.explorer.map_pointcloud_to_image(br_rad_rec['token'],
            #                                                                 b_cam_rec['token'])
            # points_bl, coloring_bl, _ = nusc.explorer.map_pointcloud_to_image(bl_rad_rec['token'],
            #                                                                 b_cam_rec['token'])
            # nusc.render_sample_data(b_cam_rec['token'], True, ax=axes)
            # plt.scatter(points_br[0, :], points_br[1, :], c=coloring_br, s=5)
            # plt.scatter(points_bl[0, :], points_bl[1, :], c=coloring_bl, s=5)
            # plt.axis('off')
            # plt.show(block=False)


            # FRONT RIGHT
            # impath, boxes, camera_intrinsic = nusc.get_sample_data(fr_cam_rec['token'])
            # points_fr, coloring_fr, im = nusc.explorer.map_pointcloud_to_image(fr_rad_rec['token'],
            #                                                                 fr_cam_rec['token'])
            # points_f, coloring_f, _ = nusc.explorer.map_pointcloud_to_image(f_rad_rec['token'],
            #                                                                 fr_cam_rec['token'])
            # nusc.render_sample_data(fr_cam_rec['token'], True, ax=axes)
            # plt.scatter(points_fr[0, :], points_fr[1, :], c=coloring_fr, s=5)
            # plt.scatter(points_f[0, :], points_f[1, :], c=coloring_f, s=15)
            # plt.axis('off')
            # plt.show(block=False)


            # ## FRONT LEFT
            # impath, boxes, camera_intrinsic = nusc.get_sample_data(fl_cam_rec['token'])
            # points_fl, coloring_fl, im = nusc.explorer.map_pointcloud_to_image(fl_rad_rec['token'],
            #                                                                 fl_cam_rec['token'])
            # points_f, coloring_f, _ = nusc.explorer.map_pointcloud_to_image(f_rad_rec['token'],
            #                                                                 fl_cam_rec['token'])
            # nusc.render_sample_data(fl_cam_rec['token'], True, ax=axes)
            # plt.scatter(points_fl[0, :], points_fl[1, :], c=coloring_fl, s=5)
            # plt.scatter(points_f[0, :], points_f[1, :], c=coloring_f, s=5)
            # plt.axis('off')
            # plt.show(block=False)


            if not f_cam_rec['next'] == "":
                f_cam_rec = nusc.get('sample_data', f_cam_rec['next'])
                # b_cam_rec = nusc.get('sample_data', b_cam_rec['next'])
                # fr_cam_rec = nusc.get('sample_data', fr_cam_rec['next'])
                # fl_cam_rec = nusc.get('sample_data', fl_cam_rec['next'])
                # br_cam_rec = nusc.get('sample_data', br_cam_rec['next'])
                # bl_cam_rec = nusc.get('sample_data', bl_cam_rec['next'])

                f_rad_rec = nusc.get('sample_data', f_rad_rec['next'])
                fr_rad_rec = nusc.get('sample_data', fr_rad_rec['next'])
                fl_rad_rec = nusc.get('sample_data', fl_rad_rec['next'])
                # br_rad_rec = nusc.get('sample_data', br_rad_rec['next'])
                # bl_rad_rec = nusc.get('sample_data', bl_rad_rec['next'])
            else:
                has_more_frames = False

            # print(impath)
            # input('something')
            # time.sleep(1)
            # plt.cla()
