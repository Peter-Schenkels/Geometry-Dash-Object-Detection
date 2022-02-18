from tempfile import template
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import random


def getRandomColor(nrOfColors):

    return (random.randint(1, 255), random.randint(1, 255) , random.randint(1, 255) )


def distance(kpt1, kpt2):
    #create numpy array with keypoint positions
    arr = np.array([kpt1.pt, kpt2.pt])
    #return distance, calculted by pythagoras
    return np.sqrt(np.sum((arr[0]-arr[1])**2))

class Rect:
    def __init__(self, blob) -> None:
        self.left = blob.pt[0]
        self.right = blob.pt[0]
        self.top = blob.pt[1]
        self.bottom = blob.pt[1]
        self.pos1 = (int(blob.pt[0]), int(blob.pt[1]))
        self.pos2 = (int(blob.pt[0] + 100), int(blob.pt[1] + 100))

    def update(self, newBlob):
        updated = False
        if(newBlob.pt[0] - newBlob.size < self.left):
            self.left = int(newBlob.pt[0] - newBlob.size)
            updated = True
        if(newBlob.pt[0] + newBlob.size > self.right):
            self.right = int(newBlob.pt[0] + newBlob.size)
            updated = True
        if(newBlob.pt[1] + newBlob.size > self.top):
            self.top = int(newBlob.pt[1] + newBlob.size)
            updated = True
        if(newBlob.pt[1] - newBlob.size < self.bottom):
            self.bottom = int(newBlob.pt[1] - newBlob.size)
            updated = True
        

        if(updated is True):
            self.pos1 = np.array([self.right, self.top], dtype=np.int64)
            self.pos2 = np.array([self.left, self.bottom], dtype=np.int64)

class Group:
    def __init__(self, blobs, color, rect) -> None:
        self.blobs = blobs
        self.color = color
        self.rect = rect
        

def classifyBlobs(blobs, range):
    groups = []
    lastBlob = None
    for blob in blobs:
        if(lastBlob == None):
            groups.append(Group([blob], getRandomColor(12), Rect(blob)))
        else:
            for group in groups:
                grouped = False
                for groupBlob in group.blobs:
                    dist = distance(groupBlob, blob)
                    if(dist < range):
                        group.rect.update(blob)
                        group.blobs.append(blob)
                        grouped = True
                        break
                    if(grouped is True):
                        break
                if(grouped is True):
                    break
            else:
                groups.append(Group([blob], getRandomColor(12), Rect(blob)))
        lastBlob = blob
    return groups
        

class AreaOfInterest:
    def __init__(self, img, pos1, pos2) -> None:
        self.pos1, self.pos2 = pos1, pos2
        if(pos1[0] > pos2[0]):
            self.pos1 = pos2
            self.pos2 = pos1
        self.img = cv.cvtColor(img[self.pos1[1]:self.pos2[1], self.pos1[0]:self.pos2[0]], cv.COLOR_BGR2GRAY)
        test = 1

def GetAreaOfInterest(image, range = 200):
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.2

    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # ori =  openAndScaleImage('image_01.jpg')
    start = time.time()

    detector = cv.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(image)
    img = cv.Canny(image, 140, 200)
    keypoints = detector.detect(img)
    groups = classifyBlobs(keypoints, range)

    print(time.time() - start)
    cv.imshow("Canny", img)

    im_with_keypoints = image.copy()
    for group in groups:
        cv.rectangle(im_with_keypoints , (group.rect.pos1[0],group.rect.pos1[1]) , (group.rect.pos2[0],group.rect.pos2[1]), group.color, 2)
        im_with_keypoints = cv.drawKeypoints(im_with_keypoints, group.blobs, np.array([]),  group.color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv.imshow("blobps", im_with_keypoints)

    aoi_list = []
    for group in groups:
        aoi_list.append(AreaOfInterest(image, group.rect.pos1, group.rect.pos2))
    
    return aoi_list

def openAndScaleTemplate(filename):
    img = cv.imread(filename, 0)
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

def scalePercentage(percentage, img):
    width = int(1920 * percentage / 100)
    height = int(1080 * percentage / 100)
    dim = (width, height)
    return cv.resize(img, dim)
    

def openAndScaleImage(filename):
    img = cv.imread(filename)
    scale_percent = 60 # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)
    
if __name__ == "__main__":


    img_rgb = openAndScaleImage('image_01.jpg')
    aoi_list = GetAreaOfInterest(img_rgb, 300)

    # for aoi in aoi_list:
    #     cv.imshow("Aoi", aoi.img)
    #     cv.waitKey(0)


    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    templates =  []

    platform_threshold = 0.86
    spike_threshold = 0.80

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
    templates.append((openAndScaleTemplate('Templates/spikes_template_01.png'), "Spikes_01", 0.7, (255, 255, 0)))

    output = []

    start = time.time()

    # canny = cv.Canny(img_rgb, 100, 200)
    # cv.imshow("Candy", canny)

    for template in templates:
        w, h= template[0].shape[::-1]
        for aoi in aoi_list:
            if(aoi.img.shape[0] > template[0].shape[0] and aoi.img.shape[1] > template[0].shape[1]):
                matches = (cv.matchTemplate(aoi.img,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], aoi.pos1)
                output.append(matches)


        # matches = (cv.matchTemplate(img_gray,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], np.array([0,0]))
        # output.append(matches)


    end = time.time() - start
    print(end)

    for set in output:
        positions = np.where( set[0] >= set[4])
        for position in list(zip(*positions[::-1])):
            position += set[6]
            cv.rectangle(img_rgb, position, (position[0] + set[2], position[1] + set[3]), set[5], 2)
            cv.putText(img_rgb, set[1], position, cv.FONT_HERSHEY_COMPLEX, 0.5, set[5],1 )



    cv.imwrite('res.png',img_rgb)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
