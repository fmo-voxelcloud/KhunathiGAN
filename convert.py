#!/usr/bin/env python
# encoding: utf-8
# author: fan.mo
# email: fmo@voxelcloud.net.cn

import os
from PIL import Image
import sys
from tqdm import tqdm

input_list = [x.strip() for x in open(sys.argv[1]).readlines()]

input_list = sorted(input_list,
    key=lambda x: (int(os.path.basename(x).split('_')[0]),
                   int(os.path.basename(x).split('_')[1].split('.')[0])))

os.makedirs("video_ims", exist_ok=True)
for idx, f in tqdm(enumerate(input_list)):
    new_name = os.path.join("video_ims", str(idx) + '.jpg')
    im = Image.open(f)
    im = im.resize([1920, 1080])
    im.save(new_name)
