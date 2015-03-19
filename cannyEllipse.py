import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math

class Ellipse:
    x, y, a, b, r = 0.0, 0.0, 0.0, 0.0, 0.0

    def __init__(self, x, y, b, a, r):
        self.data = []
        self.x = x
        self.y = y
        self.a = a 
        self.b = b
        self.r = r

def createEllipse(ellipse):
    x = ellipse[0][0]
    y = ellipse[0][1]
    a = ellipse[1][0]
    b = ellipse[1][1]
    r = ellipse[2]

    return Ellipse(x, y, a, b, r)

def shiftEllipse(ellipse, dx, dy):
    x = ellipse[0][0] + dx
    y = ellipse[0][1] + dy
    a = ellipse[1][0]
    b = ellipse[1][1]
    r = ellipse[2]

    return Ellipse(x, y, a, b, r)

def drawEllipse(e, img, color):
    cv2.ellipse(img, (e.x, e.y), (e.a, e.b), r, color, 2)

def meanEllipse(e1, e2):
    x = np.mean([e1.x, e2.x]) 
    y = np.mean([e1.y, e2.y]) 
    a = np.mean([e1.a, e2.a]) 
    b = np.mean([e1.b, e2.b]) 
    r = np.mean([e1.r, e2.r) 
    return Ellipse(x, y, a, b, r)

def distEllipse(e1, e2):
    return np.sqrt(np.square(e1.x - e2.x) + np.square(e1.y - e2.y))


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

def pltplot(img, name, folder):
    fname = folder + "\\" +name + '.png'
    cv2.imwrite(fname, img)
    cv2.imshow(fname, img)

def ellipseArea(a, b):
    return math.pi * a * b

def rectArea(w, h):
    return w * h

def compute_threshold(res, img, ratio, folder):
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
    cv2.imwrite(folder + '\filtered_ellipse.png', img)
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

def combineEllipse(e1, e2, img1, shift, folder):
    for ellipse in e1:
        cv2.ellipse(img1, ellipse, (0, 0, 255), 2)

    ellip2 = []

    for ellipse in e2:
        x = ellipse[0][0] 
        y = ellipse[0][1]
        a, b = ellipse[1][0], ellipse[1][1]
        r = ellipse[2]
        new_ellip = ((x + shift, y), (a, b), r)
        cv2.ellipse(img1, new_ellip, (0, 255, 0), 2)
        ellip2.append(new_ellip)
    
    cv2.imshow('joint_ellipse', img1)
    cv2.imwrite(folder + '\joint_ellipse.png', img1)
    cv2.waitKey(0)

    return e1, ellip2


def matching(e1, e2):

    orig = []
    matches = []
    for i in e1:
        min_val = 10.
        one_e = np.array(i[0])
        match = None
        for j in e2:
            two_e = np.array(j[0])
            dist = np.sqrt(np.sum(np.square(one_e - two_e)))

            if(dist < min_val):
                dist = min_val
                match = j 
        if(match is not None):  
            orig.append(i)
            matches.append(match)
        
    return orig, matches

def best_guess(orig, match, img1, folder):
    best = []

    for e1, e2 in zip(orig, match):
        x = np.mean([e1[0][0], e2[0][0]]) 
        y = np.mean([e1[0][1], e2[0][1]]) 
        a = np.mean([e1[1][0], e2[1][0]]) 
        b = np.mean([e1[1][1], e2[1][1]]) 
        r = np.mean([e1[2], e2[2]]) 
        new_ellip = ((x, y), (a, b), r)
        cv2.ellipse(img1, new_ellip, (0, 255, 0), 2)
        best.append(new_ellip)  

    cv2.imshow('best_ellipse', img1)
    cv2.imwrite(folder + '\best_ellipse.png', img1)
    cv2.waitKey(0)

    return best

def plot_all(e1, e2, best, img1, folder):
    for i in e1:
        cv2.ellipse(img1, i, (0, 255, 0), 2)
    
    for j in e2:
        cv2.ellipse(img1, j, (0, 0, 255), 2)
    
    for k in best:
        cv2.ellipse(img1, k, (255, 0, 0), 2)

    cv2.imshow('all', img1)
    cv2.imwrite(folder + '\all_and_best.png', img1)
    cv2.waitKey(0)


folders = ['Test '+ str(i) for i in range(1, 4)]

for folder in folders:
    imgL = cv2.imread(folder + '\left.jpeg')
    imgR = cv2.imread(folder + '\right.jpeg')

    handlesL = find_handles(imgL)
    pltplot(handlesL, '\handlesL', folder)
    h_img = cv2.imread(folder + '\handlesL.png')

    c, e, r, cts, a_L = compute_threshold(handlesL, h_img.copy(), 0.35, folder)


    handlesR = find_handles(imgR)
    pltplot(handlesR, '\handlesR', folder)
    r_img = cv2.imread(folder + '\handlesR.png')

    cr, er, rr, cts_r, a_R = compute_threshold(handlesR, r_img.copy(), 0.35, folder)

    e1, e2 = combineEllipse(e, er, h_img.copy(), 35., folder)
    orig, match = matching(e1, e2, folder)
    best = best_guess(orig, match, h_img.copy(), folder)

    plot_all(e1, e2, best, h_img.copy(), folder)
