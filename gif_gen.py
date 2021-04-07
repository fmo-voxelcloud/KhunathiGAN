#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
import imageio
import sys
import cv2

images = sys.argv[1]
output = sys.argv[2]
fps = sys.argv[3]

images = [x.strip() for x in open(images).readlines()]
gif_ims = []

print("loading img")
for im in images:
    if not os.path.exists(im):
        img = imageio.imread(os.getcwd()+'/'+im)
        img = cv2.resize(img, (128, 128))
        gif_ims.append(img)
    else:
        img = imageio.imread(im)
        img = cv2.resize(img, (128, 128))
        gif_ims.append(img)

print("generating gif")
imageio.mimsave(output, gif_ims, fps=fps)
print("Done")
