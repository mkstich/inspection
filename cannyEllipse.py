import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math

def find_handles(img):
    '''Detect the orange handles in the image by applying the mask'''
    lower_handles = np.array([50,  50,  0], dtype=np.uint8)
    upper_handles = np.array([250, 150, 35], dtype=np.uint8)#35
    # lower_white = np.array([150, 100, 70], dtype=np.uint8)
    # upper_white = np.array([200, 200, 140], dtype=np.uint8)
    
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask1= cv2.inRange(RGB, lower_handles, upper_handles)
    # mask2 = cv2.inRange(RGB, lower_white, upper_white)

    mask = mask1 #+ mask2
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def pltplot(img, name):
    fname = name + '.png'
    cv2.imwrite(fname, img)
    cv2.imshow(fname, img)

def ellipseArea(a, b):
	return math.pi * a * b

def rectArea(w, h):
	return w * h

def compute_threshold(res, img, ratio):
    '''Use Canny edge detection to find the contours, 
    Sort the contours into shapes'''

    # Get the edge map of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # Find the contours and sort the contours from large to small area
    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)

    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.imwrite('contours.png', h_img)
    # cv2.waitKey(0)

    circles, ellipses, rectangles = [], [], []
    cnts = []
    for cnt in contours:
        try:
            # Cannot uniquely fit an ellipse with less than 5 points
            if len(cnt) > 4:
            	ellipse = cv2.fitEllipse(cnt)
            	area = ellipseArea(ellipse[1][0] * 0.5, ellipse[1][1] * 0.5)

            circle = cv2.minEnclosingCircle(cnt)
            rectangle = cv2.boundingRect(cnt)

            circles.append(circle)
            rectangles.append(rectangle)

            if(area > 250. and area < 1100.):
	            ellipses.append(ellipse)

        except Exception as e:
            # print(e)
            pass 

	# Drop repeats
    circles = list(set(circles))
    ellipses = list(set(ellipses))
    rectangles = list(set(rectangles))

    # Organize into clusters, find largest, sort by area
    out = cluster(ellipses, ratio)
    lg = max(out, key = len)
    lg.sort(key = lambda lg: lg[1][0] * lg[1][1] / 4.0)

    # Grab the largest elements
    c = circles
    e = lg[: - (int(len(lg) * 0.1))]
    r = rectangles
    area = []

    # Draw the elements
    for circle, ellipse, rectangle in zip(c, e, r):
        # # Draw an ellipse
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)
        area.append(ellipseArea(ellipse[1][0] * 0.5, ellipse[1][1] * 0.5))

    # # Show the result
    cv2.imshow('filtered_ellipse', img)
    cv2.imwrite('filtered_ellipse.png', img)
    cv2.waitKey(0)

    return c, e, r, contours, area

def cluster(data, maxgap):
    ''' 
    Arrange data into groups where successive elements differ by no more than
    maxgap. '''

    groups = [[data[0]]]
    for x in data[1:]:
    	x1 = np.array(x[1])

    	# minor axis to major axis ratio
    	x1_a = x1[1]
    	x1_b = x1[0]
    	x_r = x1_b / x1_a

    	if x_r <= maxgap:
    		groups[-1].append(x)

    return groups


imgL = cv2.imread('left.png')
imgR = cv2.imread('right.png')

handlesL = find_handles(imgL)
pltplot(handlesL, 'handlesL')
h_img = cv2.imread('handlesL.png')

c, e, r, cts = compute_threshold(handlesL, h_img.copy(), 0.95)


handlesR = find_handles(imgR)
pltplot(handlesR, 'handlesR')
r_img = cv2.imread('handlesR.png')

cr, er, rr, cts_r = compute_threshold(handlesR, r_img.copy(), 0.35)