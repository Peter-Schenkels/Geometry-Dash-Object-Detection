from tempfile import template
import cv2 as cv
from cv2 import imshow
from cv2 import rectangle
import numpy as np
from matplotlib import pyplot as plt
import time
import random



#TODO detect grid en slice image in cells

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
        

class RegionOfInterest:
    def __init__(self, img, pos1, pos2) -> None:
        self.pos1, self.pos2 = pos1, pos2
        if(pos1[0] > pos2[0]):
            self.pos1 = pos2
            self.pos2 = pos1
        if(self.pos1[0] < 0):
            self.pos1[0] = 0
        if(self.pos2[0] < 0):
            self.pos2[0] = 0
        if(self.pos1[1] < 0):
            self.pos1[1] = 0
        if(self.pos2[1] < 0):
            self.pos2[1] = 0

        self.img = cv.Canny(img[self.pos1[1]:self.pos2[1], self.pos1[0]:self.pos2[0]], 800, 800)

def GetRegionOfInterest(image, range = 200):
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
    # start = time.time()
    # keypoints = detector.detect(image)
    img = cv.Canny(image, 100, 400)
    keypoints = cv.SimpleBlobDetector_create(params).detect(img)
    groups = classifyBlobs(keypoints, range)

    # print(time.time() - start)
    # cv.imshow("Canny", img)

    im_with_keypoints = image.copy()
    for group in groups:
        cv.rectangle(im_with_keypoints , (group.rect.pos1[0],group.rect.pos1[1]) , (group.rect.pos2[0],group.rect.pos2[1]), group.color, 2)
        im_with_keypoints = cv.drawKeypoints(im_with_keypoints, group.blobs, np.array([]),  group.color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv.imshow("blobps", im_with_keypoints)

    roi_list = []
    for group in groups:
        roi_list.append(RegionOfInterest(image, group.rect.pos1, group.rect.pos2))
    
    return roi_list

def openAndScaleTemplate(filename):
    img = cv.imread(filename, 0)
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

def scalePercentage(percentage, img):
    width = int(1920 * percentage / 100)
    height = int(1080 * percentage / 100)
    dim = (width, height)
    return cv.resize(img, dim)
    


def ScaleImage(img):
    scale_percent = 50 # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

def openAndScaleImage(filename):
    img = cv.imread(filename)
    scale_percent = 60 # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)

# def addTemplate(src, name, threshold, color):
#     # (cv.Canny(cv.copyMakeBorder(openAndScaleTemplate('YOLOstuff\RegularBlock01.png'), 140, 200), 1, 1, 1, 1, cv.BORDER_CONSTANT), "Test_01", platform_threshold , (0, 255, 0))

#     img = openAndScaleTemplate('YOLOstuff\RegularBlock01.png')
#     img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT)
#     img = cv.Canny(img, 140, 200)

#     plt.imshow(img)
#     plt.show()

#     return (img, name, threshold , color)

def contains(r1, r2):
       return r1.x1 < r2.x1 < r2.x2 < r1.x2 and r1.y1 < r2.y1 < r2.y2 < r1.y2

def GeomDecter(input, templates, roi_enabled=False):    

    img_rgb = ScaleImage(input)
    if(roi_enabled is True):
        roi_list = GetRegionOfInterest(img_rgb, 300)
    else:
        img = ScaleImage(cv.Canny(input, 800, 800))
    output = []
    for template in templates:
        w, h= template[0].shape[::-1]
        if(roi_enabled is True):
            for roi in roi_list:
                if(roi.img.any()):
                    if(roi.img.shape[0] > template[0].shape[0] and roi.img.shape[1] > template[0].shape[1]):
                        matches = (cv.matchTemplate(roi.img,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], roi.pos1)
                        output.append(matches)
        else:
            cv.imshow("Canny", img)
            if(img.shape[0] > template[0].shape[0] and img.shape[1] > template[0].shape[1]):
                matches = (cv.matchTemplate(img,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], (0,0))
                output.append(matches)

    
    for set in output:
        positions = np.where( set[0] >= set[4])
        mask = np.zeros(img_rgb.shape[:2], np.uint8)
        for position in list(zip(*positions[::-1])):
            if mask[position[1] + int(round(set[3]/2)), position[0] + int(round(set[2]/2))] != 255:
                mask[position[1]:position[1]+ set[3], position[0]:position[0]+ set[2]] = 255
                position = (position[0] + set[6][0], + position[1] + set[6][1])
                cv.rectangle(img_rgb, position, (position[0] + set[2], position[1] + set[3]), set[5], 2)
                cv.putText(img_rgb, set[1], position, cv.FONT_HERSHEY_COMPLEX, 0.5, set[5],1 )

    return img_rgb


def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return x, y


# def LinesToRectangles(lines):
#     rectangles = []
#     for selectedLineIndex in range(0, len(lines)):
#         Ax1, Ay1, Ax2, Ay2, = lines[selectedLineIndex][0]
#         intersections = []
#         for checkLineIndex in range(selectedLineIndex, len(lines)):
#             Bx1, By1, Bx2, By2 = lines[checkLineIndex][0]
#             intersection = line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2)
#             if(intersection):
#                 intersections.append(intersection)
#         if(len(intersections) == 4)
#             rectangles.append()


                
   
    
    # for line in lines:
    #     platform = (line[0], None, None, None)
    #     found_intersection = False
    #     for 
    # pass

# def LineDetection(edges, image):
#     image = cv.Canny(image, 10000, 10000)
#     lines = cv.HoughLinesP(
#                 edges, # Input edge image
#                 1, # Distance resolution in pixels
#                 np.pi/2, # Angle resolution in radians
#                 threshold=3, # Min number of votes for valid line
#                 minLineLength=30, # Min allowed length of line
#                 maxLineGap=10 # Max allowed gap between line for joining them
#                 )
#     print(len(lines))
#     LinesToRectangles(lines)
#     # Iterate over points
#     for points in lines:
#         # Extracted points nested in the list
#         x1,y1,x2,y2=points[0]
#         # Draw the lines joing the points
#         # On the original image
#         cv.line(image,(x1,y1),(x2,y2),(255,255,255),2)
#         # Maintain a simples lookup list for points
#         # lines_list.append([(x1,y1),(x2,y2)])
    
#     return image
     


if __name__ == "__main__":

    fps = 30
    frame_last = 1/30
    skyradio = cv.VideoCapture("video.mov")

    if not skyradio.isOpened():
        raise Exception("Skyradio staat niet aan!!!")
    skyradio.set(cv.CAP_PROP_POS_FRAMES, 300)
    templates =  []

    platform_threshold = 0.83
    spike_threshold = 0.80

    # templates.append((openAndScaleTemplate('Templates/platform_template_01.png'), "platform_01", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_02.png'), "platform_02", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_03.png'), "platform_03", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_04.png'), "platform_04", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_05.png'), "platform_05", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_06.png'), "platform_06", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/platform_template_07.png'), "platform_07", platform_threshold , (0, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/spike_template_01.png'), "Spike_01", spike_threshold, (255, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/spike_template_02.png'), "Spike_02", spike_threshold, (255, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/spike_template_03.png'), "Spike_03", spike_threshold, (255, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/spike_template_04.png'), "Spike_04", spike_threshold, (255, 255, 0)))
    # templates.append((openAndScaleTemplate('Templates/spikes_template_01.png'), "Spikes_01", 0.7, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates/spike_01.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates/spike_02.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates/horizontal.png'), "Platform", 0.7, (0, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates/vertical.png'), "Wall", 0.7, (0, 0, 255)))
    # templates.append((openAndScaleTemplate('CannyTemplates/platform_01.png'), "platform_01", 0.91, (0, 255, 0)))

    while skyradio.isOpened():
        start = time.time()
        read, frame = skyradio.read()
        i = 0
        if(read is True):
            i+=1

            output = GeomDecter(frame, templates)
            cv.imshow('frame', output)
            # now = time.time()
            # canny = cv.Canny(frame, 700, 800)
            # canny = cv.Canny(frame, 800, 800)
            # print(time.time() - now)
            # cv.imshow('canny', canny)
            # cv.imshow('frame lines',LineDetection(canny, frame))
            # cv.imwrite(str(i)+"canny.png", canny)
            if cv.waitKey(1) == ord('q'):
                break
        while time.time() - start < 1/fps:
            continue
