from tempfile import template
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time


img_rgb = cv.imread('image_01.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
templates =  []

templates.append((cv.imread('platform_template_01.png',0), "platform_template_01", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_02.png',0), "platform_template_02", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_03.png',0), "platform_template_03", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_04.png',0), "platform_template_04", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_05.png',0), "platform_template_05", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_06.png',0), "platform_template_06", 0.85, (0, 255, 0)))
templates.append((cv.imread('platform_template_07.png',0), "platform_template_07", 0.85, (0, 255, 0)))

templates.append((cv.imread('spike_template.png',0), "Spike", 0.95, (255, 255, 0)))
templates.append((cv.imread('spikes_template.png',0), "small spikes", 0.72,  (255, 0, 255)))


output = []

start = time.time()

for template in templates:
    w, h = template[0].shape[::-1]
    output.append((cv.matchTemplate(img_gray,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3]))

threshold = 0.95

end = time.time() - start
print(end)

for set in output:
    positions = np.where( set[0] >= set[4])
    for position in list(zip(*positions[::-1])):
        cv.rectangle(img_rgb, position, (position[0] + set[2], position[1] + set[3]), set[5], 2)
        cv.putText(img_rgb, set[1], position, cv.FONT_HERSHEY_COMPLEX, 0.5, set[5],1 )



cv.imwrite('res.png',img_rgb)
img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
