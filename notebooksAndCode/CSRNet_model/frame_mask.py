import cv2
import numpy as np


img = cv2.imread("img_to_predict.png")
# Create empty mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Define polygon around the useful region
pts = np.array([
    [120, 180],   # top-left useful boundary
    [1680, 180],  # top-right useful boundary
    [1750, 820],  # bottom-right
    [60, 820]     # bottom-left
], dtype=np.int32)

# Fill polygon with white
cv2.fillPoly(mask, [pts], 255)

# Apply mask
masked_img = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("masked_frame.png", masked_img)