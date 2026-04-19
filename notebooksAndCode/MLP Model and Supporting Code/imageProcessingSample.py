import cv2
import numpy as np

### Creating mask for overall crop, this will keep only what we want to see if we apply the mask with
# cv2.bitwise_and(img, img, mask=mask) {if variable output mask is defined below}
# Create this out of any loops as it is immutable and wont need to change
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


def highlightOut(imgBGR, imgBackgroundHSV, mean_colorBackground):
    #highlights the different of the two images by making the average color the same in the two images
    imgTestHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(imgTestHSV)
    backgroundMultiVal = mean_color[2]/ mean_colorBackground[2]
    
    h, s, v = cv2.split(imgBackgroundHSV)
    val = cv2.multiply(v,backgroundMultiVal)

    #shift = mean_color[0] - mean_colorBackground[0]
    #hue = (h.astype(int) + shift) % 180  # hue shift by the 

    shadowFixedcolorBackground = cv2.merge([h,s,val])

    imgsubtract = cv2.subtract(shadowFixedcolorBackground,imgTestHSV)
    imageOut = cv2.cvtColor(imgsubtract,cv2.COLOR_HSV2RGB)
    return imageOut

imageBackground = cv2.imread('background2.png') # example image

# Creates a background mask, for preprocessing information i.e. global variables
mask = createMask(imageBackground)
croppedBackground = maskCropWithAvgColor(imageBackground,mask)
imageBackgroundGRAY = cv2.cvtColor(croppedBackground, cv2.COLOR_BGR2GRAY)
imageBackgroundHSV = cv2.cvtColor(croppedBackground,cv2.COLOR_BGR2HSV)
mean_colorBackground = cv2.mean(imageBackgroundHSV)


#load image test
imageTest = cv2.imread('unlabeled_frames/frame_0024.png')

# Filtering out background
imgOUTBGR = highlightOut(imageTest,imageBackgroundHSV,mean_colorBackground)
imgOUTGray = cv2.cvtColor(imgOUTBGR,cv2.COLOR_BGR2GRAY)

# Finding thresholds for contours
_, thresh = cv2.threshold(imgOUTGray, 20, 255, cv2.THRESH_BINARY)
points = np.load('annotations/frame_0024.npy')

  
#dst = cv2.adaptiveThreshold(imgOUTGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)

contour, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = 20  # Define your minimum pixel area

filtered_contours = [c for c in contour if (cv2.contourArea(c) > min_area)]


cv2.drawContours(imgOUTBGR, filtered_contours, -1, (255, 255, 255),-1)

cv2.imshow(' Image out',imgOUTBGR)
cv2.imshow('Corrected Image out',imgOUTGray)
#cv2.imshow('Image Tested',imageTest)

maskForContours = np.zeros(imageTest.shape[:2], dtype="uint8")
cv2.drawContours(maskForContours, filtered_contours, -1, (255), -1)
cv2.drawContours(imageTest, filtered_contours, -1, (0, 255, 0), 3)

kernel = np.ones((5,5), np.uint8)
dilation = cv2.morphologyEx(maskForContours, cv2.MORPH_CLOSE, kernel)
print(len(points))
for j in (points):
  cv2.circle(imageTest,(int(j[0]),int(j[1])),3,(255, 0 ,0), -1)

### note bright orange doesnt show up too well
cv2.imshow('Image Dilation Mask',dilation)
#cv2.imshow('Image mask ',maskForContours)
cv2.imshow('Input',imageTest)

cv2.waitKey(0) # Waits indefinitely for a key press
cv2.destroyAllWindows()


""" 

fftBackground = np.fft.fftshift(np.fft.fft2(imageBackgroundGRAY))
imageTest = cv2.imread('unlabeled_frames/frame_0000.png')
imageTestGray = cv2.cvtColor(imageTest, cv2.COLOR_BGR2GRAY)
fftTest = np.fft.fftshift(np.fft.fft2(imageTestGray))
fftCompare = fftTest - fftBackground
fftOut = np.fft.ifft2(fftCompare)
cv2.imshow('background',(imageBackgroundGRAY))
cv2.imshow('subtraction',cv2.subtract(imageTestGray,imageBackgroundGRAY))
"""