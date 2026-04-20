import numpy as np
import cv2 # if error, command "pip install opencv-python"

# paste the coordinates from (https://pixspy.com/), copy option is "JASON Array"
image_id = 75 #(be sure to change the image id correspondingly)
points = [
(320, 184),(547, 98),(645, 93),(712, 98),(750, 122),(817, 99),(853, 153),(990, 98),(1020, 110),(1066, 121),(1102, 123),(1121, 124),(1142, 134),(1156, 119),(1178, 135),(1195, 123),(1225, 126),(1269, 124),(1285, 129),(1310, 131),(1335, 131),(1360, 128),(1372, 142),(1398, 144),(1417, 155),(1438, 160),(1457, 170),(1472, 176),(1507, 195),(1592, 219),(1858, 837),(1747, 801),(1700, 826),(1661, 832),(1553, 851),(1456, 843),(1429, 865),(1331, 879),(1326, 801),(1028, 975),(913, 976),(104, 926),(390, 782),(794, 755),(920, 681),(432, 388),(447, 490),(666, 401),(641, 362),(624, 242),(715, 229),(988, 212),(1227, 356),(1626, 415),(1323, 799)
]

# Verify the data points on the image 
filename_read = f'training_data/unlabeled_frames_02_07/frame_{image_id:04d}.png'
img = cv2.imread(filename_read)
for x, y in points:
    cv2.circle(img, (int(x), int(y)), 5, (0,0,255), -1)
cv2.imshow("check", img)
cv2.waitKey(0)

# Save the points to the annotations folder (need to create the folder first)
filename_save = f'training_data/annotations_02_07/image_{image_id:04d}.png'
np.save(filename_save, points)