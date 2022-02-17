import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


# reading image
img = cv2.imread('image_01.jpg')
start = time.time()  
# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.Canny(gray, 150, 250)
# setting threshold of gray image
_, threshold = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
# plt.imshow(threshold)
# plt.show()
  
# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
# list for storing names of shapes
for contour in contours[1:]:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
  
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
      
    # # using drawContours() function
    # cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
  
    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    print("size", x, y)
  
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img, 'Spike', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
    elif len(approx) == 4:
        cv2.putText(img, 'Platform', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(img, 'Decoration', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
print(time.time() - start)

# displaying the image after drawing contours
cv2.imshow('shapes', img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()