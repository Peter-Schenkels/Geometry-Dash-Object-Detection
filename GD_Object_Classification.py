import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
from typing import *
import random


player_pos = ((0, 0), 100)


def getRandomColor(nrOfColors: int):
    """Returns a random color in a range/resolution of the given parameter

    Args:
        nrOfColors (int): Amount of possible colors

    Returns:
        tuple: returns an RGB formated tuple
    """
    return (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


def distance(kpt1: np.array, kpt2: np.ndarray):
    """Returns distance between points in with help of the Pythagorean theorem

    Args:
        kpt1 np.array: Vector2 point 1
        kpt2 np.array: Vector2 point 2

    Returns:
        float : distance between point 1 and 2
    """
    arr = np.array([kpt1, kpt2])
    return np.sqrt(np.sum((arr[0]-arr[1])**2))


class BlobRect:
    """ Rectangle made out of blob positions
    """

    def __init__(self, blob) -> None:
        self.left = blob.pt[0]
        self.right = blob.pt[0]
        self.top = blob.pt[1]
        self.bottom = blob.pt[1]
        self.pos1 = (int(blob.pt[0]), int(blob.pt[1]))
        self.pos2 = (int(blob.pt[0] + 100), int(blob.pt[1] + 100))

    def update(self, newBlob):
        """updates the blob rectangle based on added new blob

        Args:
            newBlob (blob): new blob added to the rectangle
        """
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
    """Rectangle made out of vector 2 points
    """

    def __init__(self, blob) -> None:
        blob = blob[0]
        self.left = blob[0]
        self.right = blob[0]
        self.top = blob[1]
        self.bottom = blob[1]
        self.pos1 = (int(blob[0]), int(blob[1]))
        self.pos2 = (int(blob[0] + 100), int(blob[1] + 100))

    def update(self, newBlob):
        """updates the rectangle based on new points

        Args:
            newBlob (tuple): vector 2 point
        """
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
    """group of blobs or rects
    """

    def __init__(self, blobs, color, rect: Union[PointRect, BlobRect]) -> None:
        self.blobs = blobs
        self.color = color
        self.rect = rect


def classifyBlobs(blobs, range):
    """Clustering algorithm for detected blobs

    Args:
        blobs (list): list of input blobs
        range (int): threshold of distance between blobs before they become a cluster/group

    Returns:
        list: list of clusters/groups
    """
    groups = []
    lastBlob = None
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
                groups.append(
                    Group([blob], getRandomColor(12), BlobRect(blob)))
        lastBlob = blob
    return groups


def classifyCorners(corners, range, image):
    """Clustering algorithm for detected corners

    Args:
        corners (list): list of detected corners
        range (int): threshold of distance between corners before they become a cluster/group

    Returns:
        list: list of clusters/groups
    """
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
                groups.append(
                    Group([blob], getRandomColor(12), PointRect(blob)))
        lastBlob = blob
    return groups


class RegionOfInterest:
    """ Region of an input image with features to extract 
    """

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
        self.img = cv.dilate(cv.Canny(
            img[self.pos1[1]:self.pos2[1], self.pos1[0]:self.pos2[0]], 800, 800), kernel, iterations=1)


def GetRegionOfInterest(image, range, draw=True, blob=None):
    """Functions that disects a Geometry Dash input frame and returns a group of regions of interest with features
    to tmeplate match.

    Args:
        image (np.array): input geometry dash frame
        range (_type_): group detection strength
        draw (bool, optional): Draws the preprocessing steps. Defaults to True.
        blob (_type_, optional): Assign value to use blobs instead of corners of ROI-finder . Defaults to None.

    Returns:
        list: List of region of interest.
    """
    performance_point = None
    if(blob):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 90

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.2

        # # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        downscale_perc = 50
        img = cv.Canny(ScaleImage(image), 100, 400)
        WallFloorDetection(img, img)
        keypoints = cv.SimpleBlobDetector_create(params).detect(img)
        start = time.time()
        groups = classifyBlobs(keypoints, range, img)
        performance_point = (time.time() - start, len(keypoints))

    else:
        canny = cv.Canny(image, 700, 800)
        kernel = np.ones((5, 5), 'uint8')
        canny = cv.dilate(canny, kernel, iterations=1)
        corners = np.int0(cv.goodFeaturesToTrack(canny, 40, 0.01, 50))
        groups = classifyCorners(corners, range, None)

    if(draw is True):
        im_with_keypoints = image.copy()
    roi_list = []
    if(groups is not False):
        if(draw is True):
            for group in groups:
                cv.rectangle(im_with_keypoints, (group.rect.pos1[0], group.rect.pos1[1]), (
                    group.rect.pos2[0], group.rect.pos2[1]), group.color, 2)
                if(blob):
                    im_with_keypoints = cv.drawKeypoints(im_with_keypoints, group.blobs, np.array(
                        []),  group.color, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                else:
                    for i in corners:
                        x, y = i.ravel()
                        cv.circle(im_with_keypoints, (x, y), 7, 25, -1)

        for group in groups:
            roi_list.append(RegionOfInterest(
                image, group.rect.pos1, group.rect.pos2))

    else:
        if(draw is True):
            cv.rectangle(im_with_keypoints, (300, 0),
                         (img.shape[1], img.shape[0]), (0, 255, 0), 2)
        roi_list.append(RegionOfInterest(
            image, (300, 0), (img.shape[1], img.shape[0])))
    if(draw is True):
        cv.imshow("blobps", im_with_keypoints)
    return roi_list, performance_point


def openAndScaleTemplate(filename):
    """Function that opens a template and scales the template a fixed percentage

    Args:
        filename (string): Input template directory

    Returns:
        np.array: Template
    """
    img = cv.imread(filename, 0)
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)


def ScaleImage(img):
    """Scales an image to a fixed percentage of 1080p resolution

    Args:
        img (np.array): input frame

    Returns:
        np.array: scaled image
    """
    scale_percent = 50  # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)


def openAndScaleImage(filename):
    """Scales an image to a fixed percentage of 1080p resolution

    Args:
        filename (np.array): input frame

    Returns:
        np.array: scaled image
    """
    img = cv.imread(filename)
    scale_percent = 60  # percent of original size
    width = int(1920 * scale_percent / 100)
    height = int(1080 * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim)


def getPlayer(img, last_player_pos):
    """Function to fetch probabol player position from a geometry dash frame

    Args:
        img (np.array): Input geometry dash frame
        last_player_pos (tuple): last detected position of the player

    Returns:
        tuple: tuple of player position and player image
    """
    offset_x = int(img.shape[0]/1.666)
    last_player_pos = (
        (last_player_pos[0][0]-offset_x, last_player_pos[0][1]), last_player_pos[1])
    img = img[0:img.shape[1], offset_x:offset_x+121]
    # Threshold of blue in HSV space
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([360, 255, 15])

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # preparing the mask to overlay
    mask = cv.inRange(hsv_img, lower_black, upper_black)

    possible_player_pos = []
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # backtorgb = cv.cvtColor(mask,cv.COLOR_GRAY2RGB)
    for contour in contours[1:]:

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv.approxPolyDP(
            contour, 0.15 * cv.arcLength(contour, True), True)
        M = cv.moments(contour)
        # cv.drawContours(backtorgb, [contour], 0, (0, 0, 255), 2)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            if len(approx) < 6:
                pos, _ = last_player_pos
                last_x, last_y = pos
                delta = (abs(last_x - x) + abs(last_y - y))
                if(delta == 0):
                    chance = 1
                else:
                    chance = 1 / delta

                possible_player_pos.append(((x, y), chance))
    # cv.imshow("mx", backtorgb)
    if(len(possible_player_pos) > 0):
        player_pos = max(possible_player_pos, key=lambda item: item[1])
    else:
        player_pos = last_player_pos

    # cv.putText(img, 'Player', (0, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    player_pos = ((offset_x, player_pos[0][1]), player_pos[1])
    return img, player_pos


def GeomDecter(input, templates, roi_enabled=False, blob=None, roi_sensitivity=100):
    """
    Object classification algorithm for Geometry dash, with use of template matching,
    ROI-finder, hough transform, canny edge detection and more computer vision techniques

    Args:
        input (np.array): input image
        templates (list): list of templates to detect
        roi_enabled (bool, optional): bool taht determines if the algorithm uses Regions of Interest. Defaults to False.
        blob (Object, optional): Leave None to use corner detction instead of blob detection in ROI-finder. Defaults to None.
        roi_sensitivity (int, optional): sensitivity of ROI-finder clustering. Defaults to 100.

    Returns:
        np.array: image with object classification
    """

    global player_pos
    start = time.time()
    img_rgb = ScaleImage(input)
    canny = cv.Canny(img_rgb, 700, 1000)
    kernel = np.ones((3, 3), 'uint8')
    canny = cv.dilate(canny, kernel, iterations=1)
    player_img, player_pos = getPlayer(img_rgb, player_pos)

    if(roi_enabled is True):
        roi_list, _ = GetRegionOfInterest(img_rgb, roi_sensitivity, True, blob)
    else:
        kernel = np.ones((5, 5), 'uint8')
        img = cv.dilate(ScaleImage(cv.Canny(input, 800, 800)),
                        kernel, iterations=1)
    output = []
    for template in templates:
        w, h = template[0].shape[::-1]
        if(roi_enabled is True):
            for roi in roi_list:
                if(roi.img.any()):
                    if(roi.img.shape[0] > template[0].shape[0] and roi.img.shape[1] > template[0].shape[1]):
                        matches = (cv.matchTemplate(
                            roi.img, template[0], cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], roi.pos1)
                        output.append(matches)

        else:
            # cv.imshow("Canny", img)
            if(img.shape[0] > template[0].shape[0] and img.shape[1] > template[0].shape[1]):
                matches = (cv.matchTemplate(
                    img, template[0], cv.TM_CCOEFF_NORMED), template[1], w, h, template[2], template[3], (0, 0))
                output.append(matches)
    end_time = time.time() - start
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    for set in output:
        positions = np.where(set[0] >= set[4])
        for position in list(zip(*positions[::-1])):
            if mask[position[1] + int(round(set[3]/10)), position[0] + int(round(set[2]/10))] != 255:
                mask[position[1]:position[1] + set[3],
                     position[0]:position[0] + set[2]] = 255
                position = (position[0] + set[6][0], + position[1] + set[6][1])
                cv.rectangle(
                    img_rgb, position, (position[0] + set[2], position[1] + set[3]), set[5], 2)
                cv.putText(img_rgb, set[1], position,
                           cv.FONT_HERSHEY_COMPLEX, 0.5, set[5], 1)

    img_rgb = WallFloorDetection(canny, img_rgb)

    cv.putText(img_rgb, 'Player', (player_pos[0][0], player_pos[0]
               [1]-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return img_rgb, end_time


def WallFloorDetection(edges, img, pos=(0, 0)):
    """Detect walls and floors from an edge detected geometry dash image

    Args:
        edges (np.array): edge detected geometry dash input frame
        img (np.array): geometry dash input frame without edge detection (for result drawing)
        pos (tuple, optional): offset of edge image from input img. Defaults to (0,0).

    Returns:
        np.array: image with floor and wall lines drawn.
    """

    lines = cv.HoughLinesP(edges, 2, np.pi/2, threshold=30,
                           minLineLength=40, maxLineGap=1)

    if(lines is not None):
        for points in lines:
            x1, y1, x2, y2 = points[0]

            if(x1 != x2):
                cv.putText(
                    img, "Platform", (x1+pos[0], y1+pos[1]), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv.line(img, (x1+pos[0], y1+pos[1]),
                        (x2+pos[0], y2+pos[1]), (0, 255, 0), 2)
            else:
                cv.putText(
                    img, "Wall", (x1+pos[0], y1+pos[1]), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv.line(img, (x1+pos[0], y1+pos[1]),
                        (x2+pos[0], y2+pos[1]), (0, 0, 255), 2)
    return img


if __name__ == "__main__":

    fps = 30
    skyradio = cv.VideoCapture("video.mov")

    if not skyradio.isOpened():
        raise Exception("Video could not be opened")

    skyradio.set(cv.CAP_PROP_POS_FRAMES, 200)
    templates = []

    templates.append((openAndScaleTemplate(
        'CannyTemplates2/spike_01.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate(
        'CannyTemplates2/spike_02.png'), "Spikes_01", 0.5, (255, 255, 0)))
    templates.append((openAndScaleTemplate(
        'CannyTemplates2/spikes_01.png'), "Spikes_02", 0.6, (255, 255, 0)))
    templates.append((openAndScaleTemplate(
        'CannyTemplates2/spikes_03.png'), "Spikes_04", 0.55, (255, 255, 0)))

    kernel = np.ones((5, 5), 'uint8')

    results = []
    frameNr = 0
    while skyradio.isOpened():
        start = time.time()
        read, frame = skyradio.read()
        if(read):
            output, end_time = GeomDecter(
                frame, templates, roi_enabled=True, roi_sensitivity=200)
            if(end_time == 0):
                break
            results.append(end_time)
            cv.imshow('frame', output)
            if cv.waitKey(1) == ord('q'):
                break
        # while time.time() - start < 1/fps:
        #     continue
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Time (s)")
    ax1.plot(results)
    print(str(sum(results)/len(results)) + " avg")
    plt.show()
    cv.waitKey(0)
