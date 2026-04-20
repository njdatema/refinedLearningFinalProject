import h5py
import numpy as np
import matplotlib.pyplot as plt

gt_file = h5py.File("training_data/ground_truth_02_07/frame_0050.h5", "r")
density = np.asarray(gt_file["density"])

plt.imshow(density, cmap="jet")
plt.colorbar()
plt.show()

print("Count from density map:", density.sum())