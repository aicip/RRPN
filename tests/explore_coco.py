import _init_paths
import cv2
import os
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from pycocotools_plus.coco import COCO_PLUS

coco_ann_file = '/ssd_scratch/mrnabati/occlusionDetection/Code/data/coco/annotations/instances_train2017.json'
coco_imgs_dir = '/ssd_scratch/mrnabati/occlusionDetection/Code/data/coco/images/train2017'

coco = COCO_PLUS(coco_ann_file, coco_imgs_dir)
count = 0
for key, val in coco.cats.items():
    count+=1
    print('{}:{}'.format(key, val['name']))
print(count)

for image in coco.dataset['images']:
    img_id = image['id']
    ann_ids = coco.getAnnIds([img_id])
    anns = coco.loadAnns(ann_ids)
    print(anns[0])
    print(anns[0]['bbox'][0].dtype)
    input('something')

