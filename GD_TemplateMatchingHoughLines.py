from tempfile import template
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import random



#TODO detect grid en slice image in cells

def getRandomColor(nrOfColors):

    return (random.randint(1, 255), random.randint(1, 255) , random.randint(1, 255) )


def distance(kpt1, kpt2):
    #create numpy array with keypoint positions
    arr = np.array([kpt1, kpt2])
    #return distance, calculted by pythagoras
    return np.sqrt(np.sum((arr[0]-arr[1])**2))

class BlobRect:
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

class PointRect:
    def __init__(self, blob) -> None:
        blob = blob[0]
        self.left = blob[0]
        self.right = blob[0]
        self.top = blob[1]
        self.bottom = blob[1]
        self.pos1 = (int(blob[0]), int(blob[1]))
        self.pos2 = (int(blob[0] + 100), int(blob[1] + 100))

    def update(self, newBlob):
        updated = False
        newBlob = newBlob[0]
        if(newBlob[0] - 10 < self.left):
            self.left = int(newBlob[0] - 10)
            updated = True
        if(newBlob[0] + 10 > self.right):
            self.right = int(newBlob[0] + 10)
            updated = True
        if(newBlob[1] + 10 > self.top):
            self.top = int(newBlob[1] + 10)
            updated = True
        if(newBlob[1] - 10 < self.bottom):
            self.bottom = int(newBlob[1] - 10)
            updated = True
        

        if(updated is True):
            self.pos1 = np.array([self.right, self.top], dtype=np.int64)
            self.pos2 = np.array([self.left, self.bottom], dtype=np.int64)

class Group:
    def __init__(self, blobs, color, rect) -> None:
        self.blobs = blobs
        self.color = color
        self.rect = rect
        

def classifyBlobs(blobs, range, image):
    groups = []
    lastBlob = None
    print(len(blobs))
    if(len(blobs) > 100):
        return False
    for blob in blobs:
        if(lastBlob == None):
            groups.append(Group([blob], getRandomColor(12), BlobRect(blob)))
        else:
            for group in groups:
                grouped = False
                for groupBlob in group.blobs:
                    dist = distance(groupBlob.pt, blob.pt)
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
                groups.append(Group([blob], getRandomColor(12), BlobRect(blob)))
        lastBlob = blob
    return groups

def classifyCorners(corners, range, image):
    groups = []
    first = True
    for blob in corners:
        if(first is True):
            groups.append(Group([blob], getRandomColor(12), PointRect(blob)))
            first = False
        else:
            for group in groups:
                grouped = False
                for groupBlob in group.blobs:
                    dist = distance(groupBlob[0], blob[0])
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
                groups.append(Group([blob], getRandomColor(12), PointRect(blob)))
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

        kernel = np.ones((5, 5), 'uint8')
        self.img = cv.dilate(cv.Canny(img[self.pos1[1]:self.pos2[1], self.pos1[0]:self.pos2[0]], 800, 800), kernel, iterations=1)

def GetRegionOfInterest(image, range, draw=True, blob=None):
    if(blob):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 50
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 90

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.2

        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        downscale_perc = 50
        img = cv.Canny(ScaleImage(image), 100, 400)
        LineDetection(img, img)
        keypoints = cv.SimpleBlobDetector_create(params).detect(img)
        groups = classifyBlobs(keypoints, range, img)
    else:
        canny = cv.Canny(image, 700, 800)
        kernel = np.ones((5, 5), 'uint8')
        canny = cv.dilate(canny, kernel, iterations=1)
        # gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(canny,40,0.01,50)
        corners = np.int0(corners)
        groups = classifyCorners(corners, range, image)
    
    if(draw is True):
        im_with_keypoints = image.copy()
    roi_list = []
    if(groups is not False):
        if(draw is True):
            for group in groups:
                cv.rectangle(im_with_keypoints , (group.rect.pos1[0],group.rect.pos1[1]) , (group.rect.pos2[0],group.rect.pos2[1]), group.color, 2)
                if(blob):
                    im_with_keypoints = cv.drawKeypoints(im_with_keypoints, group.blobs, np.array([]),  group.color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                else:
                    for i in corners:
                        x,y = i.ravel()
                        cv.circle(im_with_keypoints,(x,y),7,25,-1)

    
        for group in groups:
            roi_list.append(RegionOfInterest(image, group.rect.pos1, group.rect.pos2))
            
    else:
        if(draw is True):
            cv.rectangle(im_with_keypoints, (300,0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)
        roi_list.append(RegionOfInterest(image, (300,0), (img.shape[1], img.shape[0])))
    if(draw is True):  
        cv.imshow("blobps", im_with_keypoints)
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

def GeomDecter(input, templates, roi_enabled=False, blob=None, roi_sensitivity=100):    
    img_rgb = ScaleImage(input)
    canny = cv.Canny(img_rgb, 700, 1000)
    kernel = np.ones((3, 3), 'uint8')
    canny = cv.dilate(canny, kernel, iterations=1)
    cv.imshow("Kurkel", canny)
    
    if(roi_enabled is True):
        roi_list = GetRegionOfInterest(img_rgb, roi_sensitivity, True, blob)
    else:
        kernel = np.ones((5, 5), 'uint8')
        img = cv.dilate(ScaleImage(cv.Canny(input, 800, 800)), kernel, iterations=1)
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
            # cv.imshow("Canny", img)
            if(img.shape[0] > template[0].shape[0] and img.shape[1] > template[0].shape[1]):
                matches = (cv.matchTemplate(img,template[0],cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], (0,0))
                output.append(matches)

    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    for set in output:
        positions = np.where( set[0] >= set[4])

        for position in list(zip(*positions[::-1])):
            if mask[position[1] + int(round(set[3]/10)), position[0] + int(round(set[2]/10))] != 255:
                mask[position[1]:position[1]+ set[3], position[0]:position[0]+ set[2]] = 255
                position = (position[0] + set[6][0], + position[1] + set[6][1])
                cv.rectangle(img_rgb, position, (position[0] + set[2], position[1] + set[3]), set[5], 2)
                cv.putText(img_rgb, set[1], position, cv.FONT_HERSHEY_COMPLEX, 0.5, set[5],1 )
    if(roi_enabled is True):
        for roi in roi_list:
            if(roi.img.any()):
                img_rgb = LineDetection(roi.img, img_rgb, roi.pos1)
    else:
        img_rgb = LineDetection(canny, img_rgb)
    #get walls and platforms
    

    # return LineDetection(canny, img_rgb)
    return img_rgb

#This function will in the future replace the vertical and horizontal templates 
def LineDetection(edges, img, pos=(0,0)):
    # cv.imshow("edjes", edges)
    # cv.waitKey(0)
    lines = cv.HoughLinesP(
                edges, # Input edge image
                2, # Distance resolution in pixels
                np.pi/2, # Angle resolution in radians
                threshold=30, # Min number of votes for valid line
                minLineLength=40, # Min allowed length of line
                maxLineGap=1 # Max allowed gap between line for joining them
                )
    # Iterate over points
    if(lines is not None):
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            if(x1 != x2):
                cv.putText(img, "Platform", (x1+pos[0],y1+pos[1]), cv.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0),1 )
                cv.line(img,(x1+pos[0],y1+pos[1]),(x2+pos[0],y2+pos[1]),(0,255,0),2)
            else:
                cv.putText(img, "Wall", (x1+pos[0],y1+pos[1]), cv.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255),1 )
                cv.line(img,(x1+pos[0],y1+pos[1]),(x2+pos[0],y2+pos[1]),(0,0,255),2)
            # Maintain a simples lookup list for points
            # lines_list.append([(x1,y1),(x2,y2)])
    
    return img
     
if __name__ == "__main__":

    fps = 30
    skyradio = cv.VideoCapture("video.mov")
    
    if not skyradio.isOpened():
        raise Exception("Skyradio staat niet aan!!!")
    
    skyradio.set(cv.CAP_PROP_POS_FRAMES, 200)
    templates =  []

    # templates.append((openAndScaleTemplate('CannyTemplates/player.png'), "player", 0.5, (255, 0, 255)))
    # templates.append((openAndScaleTemplate('CannyTemplates/platform_01.png'), "platform_01", 0.91, (0, 255, 0)))
    # templates.append((openAndScaleTemplate('CannyTemplates2/horizontal_01.png'), "Platform", 0.7, (0, 255, 0)))
    # templates.append((openAndScaleTemplate('CannyTemplates2/vertical_01.png'), "Wall", 0.8, (0, 0, 255)))
    templates.append((openAndScaleTemplate('CannyTemplates2/spike_01.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates2/spike_02.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates2/spikes_01.png'), "Spikes_02", 0.6, (255, 255, 0)))
    templates.append((openAndScaleTemplate('CannyTemplates2/spikes_02.png'), "Spikes_03", 0.6, (255, 255, 0)))

    while skyradio.isOpened():
        start = time.time()
        read, frame = skyradio.read()
        if(read):
            output = GeomDecter(frame, templates, roi_enabled=False, roi_sensitivity=200)
            
            # cv.imshow('frame', LineDetection(cv.Canny(frame, 800, 800), frame))
            cv.imshow('frame', output)
            if cv.waitKey(1) == ord('q'):
                break
        # while time.time() - start < 1/fps:
            # continue
