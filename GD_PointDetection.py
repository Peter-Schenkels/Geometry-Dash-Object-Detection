import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time


filename = 'image_05.jpg'
img = cv.imread(filename)

timss = time.time()


gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,30,0.01,50)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),7,255,-1)

print(time.time() - timss)
plt.imshow(img),plt.show()

