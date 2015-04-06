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
    x = ellipse.x + dx
    y = ellipse.y + dy
    a = ellipse.a
    b = ellipse.b
    r = ellipse.r
    return Ellipse(x, y, a, b, r)


def meanEllipse(e1, e2):
    x = np.mean([e1.x, e2.x]) 
    y = np.mean([e1.y, e2.y]) 
    a = np.mean([e1.a, e2.a]) 
    b = np.mean([e1.b, e2.b])
    # print e1.r, e2.r
    if abs(e1.r - e2.r) >= 160.:
        lg = e1.r

        if e2.r > lg:
            e2.r -= 180.
        else:
            e1.r -= 180.

    r = np.mean([e1.r, e2.r])
    print e1.r, e2.r, r

    return Ellipse(x, y, a, b, r)


def distEllipse(e1, e2):
    return np.sqrt((e1.x - e2.x)**2 + (e1.y - e2.y)**2)

def distEnd(e1, e2):
    endL = np.sqrt(((e1.x - e1.a) - e2.x)**2 + (e1.y - e2.y)**2)
    endR = np.sqrt(((e1.x + e1.a) - e2.x)**2 + (e1.y - e2.y)**2)
    return endL, endR

def distBothEnd(e1, e2):
    endL1 = np.sqrt(((e1.x - e1.a) - (e2.x - e2.a))**2 + (e1.y - e2.y)**2)
    endL2 = np.sqrt(((e1.x - e1.a) - (e2.x + e2.a))**2 + (e1.y - e2.y)**2)
    endR1 = np.sqrt(((e1.x + e1.a) - (e2.x - e2.a))**2 + (e1.y - e2.y)**2)
    endR2 = np.sqrt(((e1.x + e1.a) - (e2.x + e2.a))**2 + (e1.y - e2.y)**2)
    return endL1, endL2, endR1, endR2

####################################################################

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
    cv2.waitKey(0)

def ellipseArea(a, b):
    return math.pi * a * b

def removeDuplicate(e):
    ellipse = []
    for i in e:
        ellipse.append(createEllipse(i))

    keep = []
    half = round(len(ellipse) / 2.)

    for i in range(len(ellipse)):
        match = False#True
        one = ellipse[i]

        for j in range(len(ellipse[i+1:])):
            two = ellipse[j + i]
            far_away = False

            d = distEllipse(one, two)
            if d > 1000.:
                far_away = True

            left = one.x - one.a - 10.
            right = one.x + one.a + 10.
            top = one.y + one.b
            bottom = one.y - one.b
            # print one.x, one.y, '\t', two.x, two.y, '\t', 
            print d, left, right, top, bottom

            # ellipse 2 is near ellipse 1
            case1 = left <= two.x <= right
            case6 = bottom <= two.y <= top
            case2 = left <= (two.x - two.a) <= right
            case3 = left <= (two.x + two.a) <= right
            case4 = bottom <= (two.y - two.b) <= top
            case5 = bottom <= (two.y + two.b) <= top 

            if far_away == True:
                match = True#False

            if(i <= half):
                if (case1 and case6) == True \
                    and (case3 == True or case2 == True\
                    or case4 == True or case5 == True):

                    if ellipseArea(one.a, one.b) > ellipseArea(two.a, two.b):
                        match = True
            elif far_away == False:
                match = False

        if (match is True):
            keep.append(one)

    return keep


def compute_threshold(res, img, fname):
    '''Use Canny edge detection to find the contours, 
    Sort the contours into ellipses and then filter out ellipses 
    that do not have a small minor axis to major axis ratio'''

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

    ellipses, full_area, temp = [], [], []
    cnts = []
    for cnt in contours:
        try:
            # Cannot uniquely fit an ellipse with less than 5 points
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)

            temp.append(ellipse)

        except Exception as e:
            pass 

    # Drop repeats
    ellipses = list(set(temp))

    # Organize into clusters, find largest, sort by area
    e = cluster(ellipses)
    print len(ellipses), len(e)

    # sort ellipses based on center x - coord
    e = sorted(e, key = lambda e: e[1][0] * e[1][1], reverse = True)
    e = removeDuplicate(e)
    # area = []

    # Draw the elements
    for ellipse in e:
        # # Draw an ellipse
        drawEllipse(img, ellipse, (0, 0, 255))
        # area.append(ellipseArea(ellipse[1][0] * 0.5, ellipse[1][1] * 0.5))

    # # Show the result
    pltplot(img, fname)

    return e, contours

def cluster(data):
    ''' 
    Arrange data into groups where successive elements differ by no more than
    maxgap. '''
    min_ratio, i = 0., 0
    max_ratio = 0.
    groups, final = [], []

    for x in data:
        x1 = np.array(x[1])
        # minor axis to major axis ratio
        x1_a = x1[1]
        x1_b = x1[0]
        
        x_r = x1_b / x1_a
        groups.append((x, x_r))

    # Grab the smallest 17.5 % ratios
    lbound = 0.01
    ubound = 0.30 #175 # 0.15

    full_area = []
    for x in groups:
        x_r = x[1]

        if x_r >= lbound and x_r <= ubound:
            final.append(x[0])
            area = ellipseArea(x[0][1][0] * 0.5, x[0][1][1] * 0.5)
            full_area.append(area)

    result = []
    dev = np.array(full_area).std()
    # Remove any remaining outliers
    for area, x in zip(full_area, final):
        if area > 100. and area < np.array(full_area).mean() + 2.5 * dev:
            result.append(x)

    return result

def convertTranslation(translation, z, mag):
    inchToPx = 960.00
    scale = z / 1600.
    shift = ((translation * inchToPx) / 566.89611162999995) / scale * mag

    return shift

def combineEllipse(e1, e2, img1, disp, fname, points, stereoCams): #, disp_value, mag):
    '''Combine left and right image ellipses
    into one image. Apply a shift dx and dy shift
    to the second set of ellipses to align on new image'''

    # Draw original ellipses
    # Create a new list containing Ellipse class objects
    ellip1, ellip2 = [], []

    for e_1 in e1:
        # e_1 = createEllipse(ellipse)
        ellip1.append(e_1)
        drawEllipse(img1, e_1, (0, 0, 255))
    
    for e_2 in e2:
        # e_2_temp = createEllipse(ellipse)
        y, x = round(e_2.x), round(e_2.y)

        Tnorm = np.linalg.norm(stereoCams.T)
        d = disp[x - 1][y - 1]
        z = points[x- 1][y - 1][2]
        dx = d + (abs(stereoCams.T[0]) * z) #(d + abs(stereoCams.T[0]))
        dy = abs(stereoCams.T[0] * stereoCams.T[1]) #* z #d + z
        print dx, dy, disp[x - 1][y - 1], points[x - 1][y - 1][2]

        e_2 = shiftEllipse(e_2, dx, dy)

        ellip2.append(e_2)
        drawEllipse(img1, e_2, (0, 255, 0))

    pltplot(img1, fname)
    cv2.imshow("joint", img1)
    cv2.waitKey(0)

    return ellip1, ellip2

def bestDetection(e, er, img1, disp, folder, points, stereoCams):

    if folder != None:
        j_name = folder + '/joint_ellipse'
        b_name = folder + '/best_ellipse'
        all_name = folder + '/all_and_best'
    else:
        j_name = 'joint_ellipse'
        b_name = 'best_ellipse'
        all_name = 'all_and_best'

    e1, e2 = combineEllipse(e, er, img1.copy(), disp, j_name, points, stereoCams)
    orig, match = matching(e1, e2)
    
    # orig, match = results[index][0], results[index][1]
    best = best_guess(orig, match, img1.copy(), b_name)
    # print max_match
    return e1, e2, best


def matching(e1, e2):
    orig, matches = [], []

    for i in e1:
        min_val = 40.
        match = None

        for j in e2:
            dist = distEllipse(i, j)
            e1L, e1R = distEnd(i, j)
            e2L, e2R = distEnd(j, i)
            e11, e12, e21, e22 = distBothEnd(i, j)

            if(e11 < min_val) or (e12 < min_val) \
                or (e21 < min_val) or (e22 < min_val):
                match = j

            if(dist < min_val) or (e1L < min_val) \
                or (e2L < min_val) or (e1R < min_val) or (e2R < min_val):
                match = j

        if(match is not None):
            orig.append(i)
            matches.append(match)
        
    return orig, matches

def best_guess(orig, match, img1, fname):
    best = []

    for e1, e2 in zip(orig, match):
        new_ellip = meanEllipse(e1, e2)
        drawEllipse(img1, new_ellip, (0, 255, 0))
        best.append(new_ellip)  

    pltplot(img1, fname)

    return best

def plot_all(e1, e2, best, img1, fname):
    for i in e1:
        drawEllipse(img1, i, (0, 255, 0))
    
    for j in e2:
        drawEllipse(img1, j, (0, 0, 255))
    
    for k in best:
        drawEllipse(img1, k, (255, 0, 0))

    pltplot(img1, fname)


def test_debug():
    imgL = cv2.imread('rectify_imgL.png')
    imgR = cv2.imread('rectify_imgR.png')

    hL_name = 'filtered_ellipse_L'
    hR_name = 'filtered_ellipse_R'
    folder = None
    # j_name = 'joint_ellipse'
    # b_name = 'best_ellipse'
    all_name = 'all_and_best'

    handlesL = find_handles(imgL)
    pltplot(handlesL, 'handlesL')
    h_img = cv2.imread('handlesL.png')
    e, cts, a_L = compute_threshold(handlesL, h_img.copy(), hL_name)

    handlesR = find_handles(imgR)
    pltplot(handlesR, 'handlesR')
    r_img = cv2.imread('handlesR.png')
    er, cts_r, a_R = compute_threshold(handlesR, r_img.copy(), hR_name)

    # disparity = get_disparity(h_img, r_img)
    # disp_pts, points, dvalue = disparity(h_img, r_img)
    e1, e2, best = bestDetection(e, er, h_img.copy(), disp, folder, points, stereoCams)

    e1, e2 = combineEllipse(e, er, h_img.copy(), disp, j_name, points, stereoCams)
    orig, match = matching(e1, e2)
    best = best_guess(orig, match, h_img.copy(), b_name)
    plot_all(e1, e2, best, h_img.copy(), all_name)

def multiTest():
    folders = ['Test '+ str(i) for i in range(1, 5)]

    for folder in folders:
        print folder 
        h1 = folder + '/handlesL'
        h1_full = h1 + '.png'
        h2 = folder + '/handlesR'
        h2_full = h2 + '.png'
        hL_name = folder + '/filtered_ellipse_L'
        hR_name = folder + '/filtered_ellipse_R'
        # j_name = folder + '/joint_ellipse'
        # b_name = folder + '/best_ellipse'
        all_name = folder + '/all_and_best'

        imgL = cv2.imread(folder + '/rectify_imgL.png')
        imgR = cv2.imread(folder + '/rectify_imgR.png')

        handlesL = find_handles(imgL)
        pltplot(handlesL, h1)
        h_img = cv2.imread(h1_full)

        e, cts, a_L = compute_threshold(handlesL, h_img.copy(), hL_name)

        handlesR = find_handles(imgR)
        pltplot(handlesR, h2)
        r_img = cv2.imread(h2_full)

        er, cts_r, a_R = compute_threshold(handlesR, r_img.copy(), hR_name)

        # disp_pts, points, dvalue = disparity(h_img, r_img)
        e1, e2, best = bestDetection(e, er, h_img.copy(), disp, folder, points, stereoCams)

        # e1, e2 = combineEllipse(e, er, h_img.copy(), disp_pts, j_name, points, dvalue)
        # orig, match = matching(e1, e2)
        # best = best_guess(orig, match, h_img.copy(), b_name)

        plot_all(e1, e2, best, h_img.copy(), all_name)


