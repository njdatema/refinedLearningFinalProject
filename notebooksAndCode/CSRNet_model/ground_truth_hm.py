import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

root = "training_data"
image_dir = os.path.join(root, "unlabeled_frames_02_07")
ann_dir = os.path.join(root, "annotations_02_07")
gt_dir = os.path.join(root, "ground_truth_02_07")

os.makedirs(gt_dir, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

sigma = 12

for img_path in img_paths:
    print("Processing:", img_path)

    img = plt.imread(img_path)
    h, w = img.shape[0], img.shape[1]

    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(ann_dir, base + ".npy")

    points = np.load(ann_path)

    k = np.zeros((h, w), dtype=np.float32)

    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < w and 0 <= y < h:
            k[y, x] = 1

    density = gaussian_filter(k, sigma=sigma)

    gt_path = os.path.join(gt_dir, base + ".h5")
    with h5py.File(gt_path, "w") as hf:
        hf["density"] = density