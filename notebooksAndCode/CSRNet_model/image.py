import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2


import os
from PIL import Image
import numpy as np
import h5py
import cv2

def load_data(img_path, train=True):
    img = Image.open(img_path).convert('RGB')

    img_dir = os.path.dirname(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # Replace folder name automatically
    gt_dir = img_dir.replace('unlabeled_frames', 'ground_truth')
    gt_path = os.path.join(gt_dir, base + '.h5')

    with h5py.File(gt_path, 'r') as gt_file:
        target = np.asarray(gt_file['density'])

    target = cv2.resize(
        target,
        (target.shape[1] // 8, target.shape[0] // 8),
        interpolation=cv2.INTER_CUBIC
    ) * 64

    return img, target