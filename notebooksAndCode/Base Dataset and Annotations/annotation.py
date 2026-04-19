import numpy as np
import cv2 # if error, command "pip install opencv-python"

# paste the coordinates from (https://pixspy.com/), copy option is "JASON Array"
image_id = 28 #(be sure to change the image id correspondingly)
points = []

# Verify the data points on the image 
filename_read = f'unlabeled_frames/frame_{image_id:04d}.png'
img = cv2.imread(filename_read)
for x, y in points:
    cv2.circle(img, (int(x), int(y)), 3, (0,0,255), -1)
cv2.imshow("check", img)
cv2.waitKey(0)

# Save the points to the annotations folder (need to create the folder first)
filename_save = f'annotations/image_{image_id:04d}.png'
np.save(filename_save, points)