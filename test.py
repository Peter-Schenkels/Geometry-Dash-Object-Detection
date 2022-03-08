from __future__ import print_function
import sys
import cv2 as cv
import numpy as np
use_mask = True
img = None
templ1 = None
templ2 = None
templ3 = None
mask1 = None
mask2 = None
mask3 = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5
def main():
    
    global img
    global templ1
    global templ2
    global templ3
    img = cv.imread("image_01.jpg", cv.IMREAD_COLOR)
    templ1 = cv.imread("CannyTemplates\spike_01.png", cv.IMREAD_COLOR)
    templ2 = cv.imread("CannyTemplates\\vertical.png", cv.IMREAD_COLOR)
    templ3 = cv.imread("CannyTemplates\horizontal.png", cv.IMREAD_COLOR)
    global use_mask
    if (use_mask is True):
        use_mask = True
        global mask1
        global mask3
        global mask2
        mask1 = cv.imread("CannyTemplates\spike_01_mask.png", cv.IMREAD_COLOR )
        mask2 = cv.imread("CannyTemplates\\vertical.png", cv.IMREAD_COLOR )
        mask3 = cv.imread("CannyTemplates\horizontal.png", cv.IMREAD_COLOR )
    if ((img is None) or (templ1 is None) or (use_mask and (mask1 is None))):
        print('Can\'t read one of the images')
        return -1
    
    
    cv.namedWindow( image_window, cv.WINDOW_AUTOSIZE )
    cv.namedWindow( result_window, cv.WINDOW_AUTOSIZE )
    
    
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    cv.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
    
    MatchingMethod(match_method)
    
    cv.waitKey(0)
    return 0
    
def MatchingMethod(param):
    global match_method
    match_method = param
    
    img_display = img.copy()
    
    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        print("hoppaaa")
        result1 = cv.matchTemplate(img, templ1, match_method, None, mask1) 
        result2 = cv.matchTemplate(img, templ2, match_method, None, mask2)
        result3 = cv.matchTemplate(img, templ3, match_method, None, mask3)
        
        cv.normalize( result1, result1, 0, 1, cv.NORM_MINMAX, -1 )
        cv.normalize( result2, result2, 0, 1, cv.NORM_MINMAX, -1 )
        cv.normalize( result3, result3, 0, 1, cv.NORM_MINMAX, -1 )
        
        out = img.copy()
        threshold = 0.99
        w, h, d = templ1.shape
        loc = np.where( result1 >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv.imshow("urkel", out)
        
        threshold = 0.99
        w, h, d = templ2.shape
        loc = np.where( result2 >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        cv.imshow("urkel", out)
    
        threshold = 0.99
        w, h, d = templ3.shape
        loc = np.where( result3 >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(out, pt, (pt[0] + w, pt[1] + h), (255,255,0), 2)
        cv.imshow("urkel", out)

    else:
        result1 = cv.matchTemplate(img, templ1, match_method)
    
    
    cv.normalize( result1, result1, 0, 1, cv.NORM_MINMAX, -1 )
    
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result1, None)
    
   
    
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ1.shape[0], matchLoc[1] + templ1.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result1, matchLoc, (matchLoc[0] + templ1.shape[0], matchLoc[1] + templ1.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow(image_window, img_display)
    cv.imshow(result_window, result1)
    
    pass
if __name__ == "__main__":
    main()