import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

inputImage = cv2.imread("crop7.jpg")
# dim = (1653,2339)
#inputImage = cv2.resize(inputImage, dim, interpolation = cv2.INTER_AREA)
# Store a copy for results:
inputCopy = inputImage.copy()

# Convert BGR to grayscale:
grayInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Set a lower and upper range for the threshold:
lowerThresh = 60 #This changes by image 
upperThresh = 190 #This changes by image

# Get the lines mask:

def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
    cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

mask = cv2.inRange(grayInput, lowerThresh, upperThresh)
cv2.imshow("horizontal_line",mask)
cv2.waitKey(0)

minArea = 100
mask = areaFilter(minArea, mask)
cv2.imshow("horizontal_line",mask)
cv2.waitKey(0)

reducedImage = cv2.reduce(mask, 1, cv2.REDUCE_MAX)

# Find the big contours/blobs on the filtered image:
contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Store the lines here:
separatingLines = []

# We need some dimensions of the original image:
imageHeight = inputCopy.shape[0]
imageWidth = inputCopy.shape[1]

# Look for the outer bounding boxes:
for _, c in enumerate(contours):

    # Approximate the contour to a polygon:
    contoursPoly = cv2.approxPolyDP(c, 3, True)

    # Convert the polygon to a bounding rectangle:
    boundRect = cv2.boundingRect(contoursPoly)

    # Get the bounding rect's data:
    [x, y, w, h] = boundRect

    # Start point and end point:
    lineCenter = y + (0.5 * h)
    startPoint = (0,int(lineCenter))
    endPoint = (int(imageWidth), int(lineCenter))

    # Store the end point in list:
    separatingLines.append( endPoint )

    # Draw the line using the start and end points:
    color = (0, 255, 0)
    cv2.line(inputCopy, startPoint, endPoint, color, 2)

    # Show the image:
    cv2.imshow("inputCopy", inputCopy)
    cv2.waitKey(0)

# Sort the list based on ascending Y values:
separatingLines = sorted(separatingLines, key=lambda x: x[1])

# The past processed vertical coordinate:
pastY = 0

# Crop the sections:
for i in range(len(separatingLines)):

    # Get the current line width and starting y:
    (sectionWidth, sectionHeight) = separatingLines[i]

    # Set the ROI:
    x = 0
    y = pastY
    cropWidth = sectionWidth
    cropHeight = sectionHeight - y

    # Crop the ROI:
    currentCrop = inputImage[y:y + cropHeight, x:x + cropWidth]

    #array.append(currentCrop)
    #plt.imsave('test1',array)
    
    cv2.imwrite(f'{i}' + ".jpg",currentCrop)
    #cv2.imshow(f'{i}' + ".jpg", currentCrop)
    #cv2.waitKey(0)

    # Set the next starting vertical coordinate:
    pastY = sectionHeight

    ##################################

array=[]

#Create the image array like: array = ['0.jpg','1.jpg'] 
##print(len(separatingLines))

for i in range(0,len(separatingLines)):

    s1 ='.jpg'
    array.append(str(i)+s1)
    
print('Image Array :',(array))

img_array = []

#append the array
for i in range(len(array)):

    img_array.append(cv2.imread(array[i],0))

img = img_array[4]

#convert to gray image

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#convert to binary  image
#function to count the tally marks for one image
def count(img):

    ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #draw contours

    cnts, heir = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img2 = img.copy()

    cv2.drawContours(img2, cnts, -1,(0,255,0),3)

    #print(len(cnts))

    array=[]
    def allPerimeter():

        for i in range(0,len(cnts)):
            array.append(cv2.arcLength(cnts[i],True))
        return array

    allPerimeter()

    array.sort()

    mid = (max(array)+min(array))/2

    def countTally():

        count = 0

        for i in range(0,len(array)):
            if array[i]<mid:
                count += 1
            else:
                count +=5
        return count;

    #print("Count :",countTally())
    return countTally();   

array_count =[]

for i in range(0,len(img_array)):

    array_count.append(count(img_array[i]))    

##print(array_count)
    
print('\n---- Frequency Table -----\n')

data = pd.DataFrame(array_count,columns=['Frequency'])

print(data)
