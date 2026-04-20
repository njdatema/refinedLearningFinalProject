from PIL import Image
import h5py
import numpy as np

img = Image.open("training_data/unlabeled_frames_02_05/frame_0043.png")
with h5py.File("training_data/ground_truth_02_05/frame_0043.h5", "r") as f:
    gt = np.asarray(f["density"])

print("Image size:", img.size)
print("GT shape:", gt.shape)
print("GT sum:", gt.sum())