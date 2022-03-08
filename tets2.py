from GD_TemplateMatching import *



if __name__ == "__main__":
    img = cv.imread("image_01.jpg")
    canny = cv.Canny(img, 700, 800)
    kernel = np.ones((3, 3), 'uint8')
    canny = cv.dilate(canny, kernel, iterations=1)
    cv.imshow("Opened",  canny)
    cv.imwrite("res.png", canny)
    cv.waitKey(0)