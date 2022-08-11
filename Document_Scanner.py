import cv2
import numpy as np
import math

def sort_points(points):

    points = points[0:4]
    return points

#This function is used to fit any image in a proper window and display it
def img_show(name, img):
    height = img.shape[0]
    width  = img.shape[1]
    ratio = width/height
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    window_height = 1000
    window_width = window_height*ratio
    window_width = round(window_width)
    cv2.resizeWindow(name,window_width,window_height)
    cv2.imshow(name,img)


def find_document_edges(image):
    height = image.shape[0]
    width = image.shape[1]

    cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found: ", len(cnts))

        # the contour with biggest area is considered
    big_contour = max(cnts, key=cv2.contourArea)
        #create a blank image of the same size of the original image
    blank_image = np.zeros((height, width, 1), np.uint8)
    img_show('blank_image', blank_image)
        #draw the big_conour on blank image
    cv2.drawContours(blank_image, big_contour, -1, (255), thickness=3)

        #repeated dilation and erosion the image
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(blank_image, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)
    img_show('eroded', eroded)
        #now find the contour from new image
    cnts2, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contour = max(cnts2, key=cv2.contourArea)
        #draw this contour on orginal image
    cv2.drawContours(img, big_contour, -1, (0, 255, 0), thickness=2)
    return big_contour

def get_lines(image,ratio):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 100  # angular resolution in radians of the Hough grid
    threshold = 150  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = round(100*ratio)  # minimum number of pixels making up a line
    max_line_gap = round(150*ratio)  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255,255,255), 1)

    img_show('line_image',line_image)
        #repeated dilation and erosion the image
    kernel = np.ones((round(5*ratio),round(5*ratio)), np.uint8)
    dilated = cv2.dilate(line_image, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)
    img_show('eroded', eroded)
    cnts2, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contour = max(cnts2, key=cv2.contourArea)

    # Draw the lines on the  image
    #lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return big_contour

def find_corner_points(document_edges,ratio):
    # for c in sorted_contours:
    epsilon = 0.02*ratio * cv2.arcLength(document_edges, True)
    points = cv2.approxPolyDP(document_edges, epsilon, True)
    if len(points) > 4:
        points = sort_points(points)

    for i in range(len(points)):
        cv2.circle(img, points[i, 0], round(5*ratio), (0, 0, 255), round(4*ratio))
    i=0


    return points

def rearrangePts(points):
    sum1 = 1000000
    sum2 = 0
    x1 = 0
    for i in range(len(points)):
        x= points[i,0,0]
        y= points[i,0,1]
        sum = x*y
        if sum < sum1:
            point1 = points[i,0]
            sum1 = sum
    i =0
    for i in range(len(points)):
        x= points[i,0,0]
        y= points[i,0,1]
        sum = x*y
        if sum > sum2:
            point4 = points[i,0]
            sum2 = sum
    i = 0
    for i in range(len(points)):
        if points[i,0,0] != point1[0] and points[i,0,0] != point4[0] :
            x = points[i, 0, 0]
            y = points[i, 0, 1]
            if x > x1:
                point2 = points[i,0]
                x1 = x
    i = 0
    for i in range(len(points)):
         if points[i,0,0] != point1[0] and points[i,0,0] != point4[0] and points[i,0,0] != point2[0]:
                    point3 = points[i,0]

    points1 = [point1,point2,point3,point4]

    return  points1


img = cv2.imread("C:\\Users\\Lenovo\\PycharmProjects\\OpenCVpython\\Resources\\Doc2.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img_gray.shape[0]
width = img_gray.shape[1]
print(height)
pixels = height*width

ratio = math.sqrt(pixels/1228800)

kernel =(3,3)
print(kernel)
img_gray = cv2.blur(img,kernel)

    #canny edge detection
        #threshold1 is the minimum value of the differentiation below which any edges will be deleted
        #threhold2 is the maximum value above which the edges are fixed to be considered
        # between these two values the edges will be considered only if they are attached to the high threshold edeges
canny = cv2.Canny(image=img_gray, threshold1=100, threshold2=260)

    #kernal to be used for the morphological functions
kernel = np.ones((round(3*ratio),round(3*ratio)), np.uint8)
dilated1 = cv2.dilate(canny, kernel, iterations=2)

#find the corner points using the the following user defined functions
#document_edges = find_document_edges(dilated1)
document_edges = get_lines(dilated1,ratio)

corner_points = find_corner_points(document_edges,ratio)
points = rearrangePts(corner_points)
print()

pts1 = np.float32(points)  # PREPARE POINTS FOR WARP
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # PREPARE POINTS FOR WARP

matrix = cv2.getPerspectiveTransform(pts1, pts2)

imgWarpColored = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
# REMOVE 20 PIXELS FORM EACH SIDE

imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
# imgWarpColored = cv2.resize(imgWarpColored, (img.shape[1], img.shape[0]))

# APPLY ADAPTIVE THRESHOLD
imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)


img_show('canny', canny)
img_show('dilated',dilated1)
img_show('img', img)
img_show('imgAdaptiveThre', imgAdaptiveThre)
cv2.imwrite('F:\\university\\lectures\\Computer vison\\Task 1\\output images\\document6.jpg', img)
cv2.imwrite('F:\\university\\lectures\\Computer vison\\Task 1\\output images\\document6_result.jpg', imgAdaptiveThre)
cv2.waitKey()
cv2.destroyAllWindows()
