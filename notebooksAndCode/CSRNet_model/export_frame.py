import cv2
import os
import numpy as np


def createMask(img):
  width, height = img.shape[1], img.shape[0]
  mask = np.zeros(img.shape[:2], dtype="uint8")

  upperLeftCutPoints = np.array([[[0,0],[0,int(height*1/3)],[int(width*1/3),0]]],dtype=np.int32)
  upperRightCutPoints = np.array([[[width,0],[int(5/8*width),0],[width,int(1/3*height)]]],dtype=np.int32)
  cv2.fillPoly(mask, upperLeftCutPoints, 255)
  cv2.fillPoly(mask, upperRightCutPoints, 255)
  mask = 255-mask
  return mask

### Applies mask and fills the black space from the applied mask with the average color of the beginning image
def maskCropWithAvgColor(img,mask):
  imgVis =  cv2.bitwise_and(img,img,mask=mask) # applies mask
  maskRGB = cv2.cvtColor((255-mask),cv2.COLOR_GRAY2RGB) # sets up a rgb channel to add to img
  imgAvgColor = cv2.mean(img)   #takes overall img preprocessed average color
  r,g,b = cv2.split(maskRGB)
  r = cv2.multiply(r,int(imgAvgColor[2])/255)  # set the red mask values to the average color
  g = cv2.multiply(g,int(imgAvgColor[1])/255)  # set the green mask values to the average color
  b = cv2.multiply(b,int(imgAvgColor[0])/255)  # set the blue mask values to the average color
  invMaskRGB = cv2.merge([r,g,b])
  imgVis = cv2.add(imgVis,invMaskRGB)

  return imgVis

video_path = "Dataset/black-rink-26-02-07_10_10.mp4"
output_folder = "unlabeled_frames_07"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

interval = int(fps * 30)  # every 10 seconds
frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % interval == 0:
        mask = createMask(frame)
        frame = maskCropWithAvgColor(frame,mask)
        filename = os.path.join(output_folder, f"frame_{saved_id:04d}.png")
        cv2.imwrite(filename, frame)
        saved_id += 1
    
    frame_id += 1

cap.release()