import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math

class Ellipse:
    x, y, a, b, r = 0.0, 0.0, 0.0, 0.0, 0.0

    def __init__(self, x, y, a, b, r):
        self.data = []
        self.x = x
        self.y = y
        self.a = a 
        self.b = b
        self.r = r
    
def drawEllipse(img, e, color):
    cv2.ellipse(img, ((e.x, e.y), (e.a, e.b), e.r), color, 2)

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


def meanEllipse(e1, e2):
    x = np.mean([e1.x, e2.x]) 
    y = np.mean([e1.y, e2.y]) 
    a = np.mean([e1.a, e2.a]) 
    b = np.mean([e1.b, e2.b]) 
    r = np.mean([e1.r, e2.r])
    return Ellipse(x, y, a, b, r)


def distEllipse(e1, e2):
    return np.sqrt((e1.x - e2.x)**2 + (e1.y - e2.y)**2)

################################################################3333

def find_handles(img):
    '''Detect the orange handles in the image by applying the mask'''
    lower_handles = np.array([50,  50,  0], dtype=np.uint8)
    upper_handles = np.array([250, 150, 35], dtype=np.uint8)
    
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask= cv2.inRange(RGB, lower_handles, upper_handles)
    res = cv2.bitwise_and(img, img, mask=mask)

    return res

def pltplot(img, name):
    fname = name + '.png'
    cv2.imwrite(fname, img)
    # cv2.imshow(fname, img)

def ellipseArea(a, b):
    return math.pi * a * b

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

    ellipses = []
    cnts = []
    for cnt in contours:
        try:
            # Cannot uniquely fit an ellipse with less than 5 points
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                area = ellipseArea(ellipse[1][0] * 0.5, ellipse[1][1] * 0.5)

            if(area > 250. and area < 1100.):
                ellipses.append(ellipse)

        except Exception as e:
            pass 

    # Drop repeats
    ellipses = list(set(ellipses))

    # Organize into clusters, find largest, sort by area
    out = cluster(ellipses, ratio)
    lg = max(out, key = len)
    lg.sort(key = lambda lg: lg[1][0] * lg[1][1] / 4.0)

    # Grab the largest elements
    e = lg[: - (int(len(lg) * 0.1))]
    area = []

    # Draw the elements
    for ellipse in e:
        # # Draw an ellipse
        cv2.ellipse(img, ellipse, (0, 0, 255), 2)
        area.append(ellipseArea(ellipse[1][0] * 0.5, ellipse[1][1] * 0.5))

    # # Show the result
    cv2.imshow('filtered_ellipse', img)
    cv2.imwrite('filtered_ellipse.png', img)
    cv2.waitKey(0)

    return e, contours, area

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

def combineEllipse(e1, e2, img1, dx, dy):
    '''Combine left and right image ellipses
    into one image. Apply a shift dx and dy shift
    to the second set of ellipses to align on new image'''

    # Draw original ellipses
    # Create a new list containing Ellipse class objects
    ellip1, ellip2 = [], []

    for ellipse in e1:
        e_1 = createEllipse(ellipse)
        ellip1.append(e_1)
        drawEllipse(img1, e_1, (0, 0, 255))
    
    for ellipse in e2:
        e_2 = shiftEllipse(ellipse, dx, dy)
        ellip2.append(e_2)
        drawEllipse(img1, e_2, (0, 255, 0))

 
    cv2.imshow('joint_ellipse', img1)
    cv2.imwrite('joint_ellipse.png', img1)
    cv2.waitKey(0)

    return ellip1, ellip2

def matching(e1, e2):
    orig, matches = [], []

    for i in e1:
        min_val = 10.
        match = None

        for j in e2:
            dist = distEllipse(i, j)

            if(dist < min_val):
                dist = min_val
                match = j

        if(match is not None):
            orig.append(i)
            matches.append(match)
        
    return orig, matches

def best_guess(orig, match, img1):
    best = []

    for e1, e2 in zip(orig, match):
        new_ellip = meanEllipse(e1, e2)
        drawEllipse(img1, new_ellip, (0, 255, 0))
        best.append(new_ellip)  

    cv2.imshow('best_ellipse', img1)
    cv2.imwrite('best_ellipse.png', img1)
    cv2.waitKey(0)

    return best

def plot_all(e1, e2, best, img1):
    for i in e1:
        drawEllipse(img1, i, (0, 255, 0))
    
    for j in e2:
        drawEllipse(img1, j, (0, 0, 255))
    
    for k in best:
        drawEllipse(img1, k, (255, 0, 0))

    cv2.imshow('all', img1)
    cv2.imwrite('all_and_best.png', img1)
    cv2.waitKey(0)


def test_debug():
    imgL = cv2.imread('left.png')
    imgR = cv2.imread('right.png')

    handlesL = find_handles(imgL)
    pltplot(handlesL, 'handlesL')
    h_img = cv2.imread('handlesL.png')
    e, cts, a_L = compute_threshold(handlesL, h_img.copy(), 0.35)

    handlesR = find_handles(imgR)
    pltplot(handlesR, 'handlesR')
    r_img = cv2.imread('handlesR.png')
    er, cts_r, a_R = compute_threshold(handlesR, r_img.copy(), 0.35)

    e1, e2 = combineEllipse(e, er, h_img.copy(), 35., 0.)
    orig, match = matching(e1, e2)
    best = best_guess(orig, match, h_img.copy())
    plot_all(e1, e2, best, h_img.copy())

# folders = ['Test '+ str(i) for i in range(1, 4)]

# for folder in folders:
#   imgL = cv2.imread(folder + '\left.jpeg')
#   imgR = cv2.imread(folder + '\right.jpeg')

#   handlesL = find_handles(imgL)
#   pltplot(handlesL, '\handlesL', folder)
#   h_img = cv2.imread(folder + '\handlesL.png')

#   e, cts, a_L = compute_threshold(handlesL, h_img.copy(), 0.35, folder)


#   handlesR = find_handles(imgR)
#   pltplot(handlesR, '\handlesR', folder)
#   r_img = cv2.imread(folder + '\handlesR.png')

#   er, cts_r, a_R = compute_threshold(handlesR, r_img.copy(), 0.35, folder)

#   e1, e2 = combineEllipse(e, er, h_img.copy(), 35., folder)
#   orig, match = matching(e1, e2, folder)
#   best = best_guess(orig, match, h_img.copy(), folder)

#   plot_all(e1, e2, best, h_img.copy(), folder)
