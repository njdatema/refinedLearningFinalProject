import cv2
import numpy as np
import math

def main():
    lengthAnnotation =2
    centerPoint =  [1068, 959]  # Pixel directly under camera (y,x)
    imageBackground = cv2.imread('background2.png') # example background that is clear
    min_area = 20  # Define your minimum pixel area
    counter = []    # initializes counter for each 'blob' per frame counters
    missedHeads = []
    headsPer = []
    dataX = []   # Stores our input data
    dataY = []  # Stores output data
    my_list = range(0,100,1)  # Samples 
    # Creates a background mask, for preprocessing information i.e. global variables
    mask = createMask(imageBackground)
    croppedBackground = maskCropWithAvgColor(imageBackground,mask)
    imageBackgroundHSV = cv2.cvtColor(croppedBackground,cv2.COLOR_BGR2HSV)
    mean_colorBackground = cv2.mean(imageBackgroundHSV)

    for i in my_list:
        imageDirectory = f'unlabeled_frames/frame_{i:04d}.png'
        imageTest = cv2.imread(imageDirectory) # reads file
        
        if (i <= 50): 
            annotationsDirectory = f'annotations/frame_{i:04d}.npy'

        else:
            annotationsDirectory = f'annotations/image_{i:04d}.png.npy'
        annotationData = np.load(annotationsDirectory)
    
        

        ### Preprocessing Image into contours
        # image processing for contours
        imgOUTBGR = highlightOut(imageTest,imageBackgroundHSV,mean_colorBackground)
        imgOUTGray = cv2.cvtColor(imgOUTBGR,cv2.COLOR_BGR2GRAY)
        
        # Finding thresholds for contours
        _, thresh = cv2.threshold(imgOUTGray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        filtered_contours = [c for c in contours if (cv2.contourArea(c) > min_area)]
        
        maskForContours = np.zeros(imageTest.shape[:2], dtype="uint8")
        cv2.drawContours(maskForContours, filtered_contours, -1, (255), -1)
        kernel = np.ones((5,5), np.uint8)
        dilation = cv2.morphologyEx(maskForContours, cv2.MORPH_CLOSE, kernel)
        contours, __ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finds contours on the mask 
        counter.append(len(contours))
        
        emptyMask = np.zeros(imageTest.shape[:2], dtype="uint8") 
        contourMask = np.zeros(imageTest.shape[:2], dtype="uint8") 
        pointMask = np.zeros(imageTest.shape[:2], dtype="uint8") 
        
        headsPer.append(len(annotationData))

        for cnt in contours:   
            count = 0          
            IsInContour = []
            
            contourMask = emptyMask.copy()
            cv2.drawContours(contourMask, [cnt], -1, (255), -1) 

            if len(annotationData) != 0:
                for j in reversed(range(len(annotationData))):
                    #pt = (int(annotationData[j, 1]), int(annotationData[j, 0]))
                    pointMask = emptyMask.copy()
                    cv2.circle(pointMask, (int(annotationData[j,0]),int(annotationData[j,1])), 4, (255), -1) # Green circle, 3px thickness
                    hasOverlap = np.any(np.bitwise_and(pointMask,contourMask))
                    IsInContour.append(hasOverlap)

                    if hasOverlap == True:
                        count += 1
                        # Remove point so it's not counted in other contours
                        annotationData = np.delete(annotationData, j, axis=0)
                        #print(annotationData)
            print(len(annotationData))
                        
            
            area = cv2.contourArea(cnt)  # area of contour
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) # set5s up projection 
            u = [vx, vy] # forces the slope to the be accurate to the first quadrant tangent line
        
            pointRelativeX = x-centerPoint[1]
            pointRelativeY = y-centerPoint[0]
            radiusToPoint = math.sqrt(math.pow(pointRelativeX, 2)+math.pow(pointRelativeY, 2))
            
            thetaPointRelative = abs(math.atan2(pointRelativeY,pointRelativeX))
            
            v = [(pointRelativeX/radiusToPoint),(pointRelativeY/radiusToPoint)]
            dot_productNorm = (v[0] * u[0]) + (v[1] * u[1])/(math.sqrt(v[0]**2+v[1]**2)*math.sqrt(u[0]**2+u[1]**2))
            dot_productNorm = dot_productNorm.item()
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area  # percept of a complex hull 
            
            dataX.append([radiusToPoint, thetaPointRelative, dot_productNorm, area, solidity])  #radial data storage
            #dataX.append([centerPoint[1],centerPoint[0], area, solidity]) #Cartesian Strategy 
            dataY.append(count)
        missedHeads.append(len(annotationData))
        
        cv2.drawContours(contourMask, contours, -1, (255), -1) 
        cv2.bitwise_and(imageTest,imageTest,mask=contourMask)
        filename_save = f'annotatedMasked/V2image_{j:04d}.png'
        cv2.imwrite(filename_save, imageTest)

        print(f'{i:04d}')

    #print(dataX)
    #print(dataY)
    #print(missedHeads)
    #print(headsPer)
    #print(counter)


    np.save('version4ftrDataXWITHCART', dataX)
    np.save('version4ftrDataYWITHCART', dataY)
    np.save('version4ftrmissedHeadsWITHCART', missedHeads)
    np.save('version4ftrtotalHeadsWITHCART', headsPer)
    np.save('version4ftrblobsPerFrameWITHCART', counter)







### FUNCTIONS 

#  Creating mask for overall crop, this will keep only what we want to see if we apply the mask with
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

main()
