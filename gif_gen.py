#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
import imageio
import sys

images = sys.argv[1]
output = sys.argv[2]
fps = sys.argv[3]

images = [x.strip() for x in open(images).readlines()]
gif_ims = []

for im in images:
    if not os.path.exists(im):
        gif_ims.append(imageio.imread(os.getcwd()+'/'+im))
    else:
        gif_ims.append(imageio.imread(im))

imageio.mimsave(output, gif_ims, fps=fps)
