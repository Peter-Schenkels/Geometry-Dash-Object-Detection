from tempfile import template
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time


def openAndScaleTemplate(filename):
    img = cv.imread(filename, 0)
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

def openAndScaleImage(filename):
    img = cv.imread(filename)
    scale_percent = 60 # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)
    

img_rgb = openAndScaleImage('image_01.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
templates =  []


platform_threshold = 0.85
spike_threshold = 0.85

templates.append((openAndScaleTemplate('Templates/platform_template_01.png'), "platform_01", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_02.png'), "platform_02", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_03.png'), "platform_03", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_04.png'), "platform_04", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_05.png'), "platform_05", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_06.png'), "platform_06", platform_threshold , (0, 255, 0)))
templates.append((openAndScaleTemplate('Templates/platform_template_07.png'), "platform_07", platform_threshold , (0, 255, 0)))

templates.append((openAndScaleTemplate('Templates/spike_template_01.png'), "Spike_01", spike_threshold, (255, 255, 0)))
templates.append((openAndScaleTemplate('Templates/spike_template_02.png'), "Spike_02", spike_threshold, (255, 255, 0)))
templates.append((openAndScaleTemplate('Templates/spike_template_03.png'), "Spike_03", spike_threshold, (255, 255, 0)))
templates.append((openAndScaleTemplate('Templates/spike_template_04.png'), "Spike_04", spike_threshold, (255, 255, 0)))



output = []

start = time.time()

for template in templates:
    w, h= template[0].shape[::-1]
    output.append((cv.matchTemplate(img_gray,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3]))


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
