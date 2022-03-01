import cv2
import numpy as np
 
# Read image
image = cv2.imread('image_01.jpg')
 
# Convert image to grayscale
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 
# Use canny edge detection
edges = cv2.Canny(image,800,800)
cv2.imshow("edges", edges)
# Apply HoughLinesP method to
# to directly obtain line end points
def LineDetection(edges, image):
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/2, # Angle resolution in radians
                threshold=20, # Min number of votes for valid line
                minLineLength=60, # Min allowed length of line
                maxLineGap=20 # Max allowed gap between line for joining them
                )
    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
        # Maintain a simples lookup list for points
        # lines_list.append([(x1,y1),(x2,y2)])
    
    return image
     
# Save the result image
cv2.imshow('detectedLines.png',image)
cv2.waitKey(0)

