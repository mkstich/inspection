import sys
import cv2
from cv2 import cv
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import glob
import re
import pandas as pd
from scipy import ndimage
from stereo_calibrate import *


class KalmanFilter(object):
    def __init__(self, process_var, measurement_var):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.posteri_est = None
        self.posteri_error_est = 1.0

    # Update a Kalman Filter with the measured value
    def update(self, measurement):
        if self.posteri_est is None:
            self.posteri_est = measurement

        priori_est = self.posteri_est
        priori_error_est = self.posteri_error_est + self.process_var

        gain = priori_error_est / (priori_error_est + self.measurement_var)
        self.posteri_est = priori_est + gain * (measurement - priori_est)
        self.posteri_error_est = (1 - gain) * priori_error_est

    # Predict a filtered value for the measured value
    def predict(self):
        return self.posteri_est

def imageName(folder, name):
    '''Create the proper img name path'''
    if len(folder) != 0:
        imgName = folder + '/' + name
    else:
        imgName = name

    return imgName


def splitString(name):
    '''Extract the numbers from the img name'''
    vals = name.split('_')
    y = vals[-1].split('.')[0]
    val1 = [float(vals[-3]), float(vals[-2]), float(y)]

    return val1


def findRectangle(stereoCams, folder, imsize, L_name, R_name, loc):
    '''Locate rectangles in a set of images
    @stereoCams = stereoCams camera object
    @folder = folder to save images to
    @imsize = img size
    @L_name = L img name
    @R_name = R img name
    @loc = EDGE location of where pic was taken

    return = calculated L and R handle rectangles
    '''
    # Read in the images
    imgL_unrect = cv2.imread(L_name)
    imgR_unrect = cv2.imread(R_name)

    # Rectify the images
    left = 'rectify_imgL' + loc + '.png'
    right = 'rectify_imgR' + loc + '.png'
    r_imgL, r_imgR = rectifyImage(
        imgL_unrect, imgR_unrect, imsize, stereoCams, folder, right, left)

    # Create all image names needed
    h1 = imageName(folder, 'handlesL.png')
    h2 = imageName(folder, 'handlesR.png')
    hL_name = imageName(folder, 'filtered_rect_L' + loc + '.png')
    hR_name = imageName(folder, 'filtered_rect_R' + loc + '.png')

    # Compute handle gradient and threshold for L and R imgs
    handlesL = find_handles(r_imgL, h1)

    handlesR = find_handles(r_imgR, h2)

    # Find rectangles from gradient images
    rectL = compute_threshold(handlesL, hL_name, False)
    rectR = compute_threshold(handlesR, hR_name, False)

    return rectL, rectR

def find_handles(img, name):
    '''Detect the yellow/orange handles in the image by applying the mask
    @img = input image
    @name = image name

    return = the calculated bitwise image residual
    '''
    # gray / white sticker color
    lower_handles = np.array([125,  125,  125], dtype=np.uint8)
    upper_handles = np.array([215, 215, 215], dtype=np.uint8)

    # Apply the color range to the image
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(RGB, lower_handles, upper_handles)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Save the parsed image
    # cv2.imwrite(name, res)
    return res


def compute_threshold(img, fname, small):
    '''Use Canny edge detection to find the contours, 
    Sort the contours into ellipses and then filter out ellipses 
    that do not have a small minor axis to major axis ratio
    @img = input img
    @fname = image name
    @small = round specification. If small = False, this is the first time
        the img's handles are being detected. If small = True, this is the 
        second time the handles are being detected, and instead of grabbing the
        whole handle, use minAreaRect to help locate the handle's number tag.

    return = list of either the largest or the two largest (nonoverlapping) rects
    '''

    # Get the edge map of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # Find the contours and sort the contours from large to small area
    (contours, _) = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    rectangles = []
    cnts = []
    for cnt in contours:
        try:
            if small is True:
                rectangle = cv2.minAreaRect(cnt)
                rectangles.append(rectangle)

            else:
                x, y, w, h = cv2.boundingRect(cnt)
                if (w * h > 500 and w * h < 50000):
                    if (float(w) / h > 2.45 and float(w) / h < 2.9):
                        rectangles.append((x, y, w, h))

        except Exception as e:
            pass

    # Drop repeats
    rectangles = list(set(rectangles))

    # Draw the elements
    # Keep the largest
    finalRect = []
    vertical = False

    rectangles.sort(
        key=lambda rectangle: rectangle[2] * rectangle[3], reverse=True)

    r_temp, r = [], []
    r_temp = rectangles[0:3]

    # only grab values within the similar y value
    # this range was tested for 24 inch images
    for i in r_temp:
        x, y, w, h = i
        if y > 360. and y < 390.:
            r.append(i)

    if len(r) > 0:
        x1, y1, w1, h1 = r[0]

        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        finalRect.append(r[0])

        repeat = False
        # Keep the second largest, if the second is not contained
        # within the first rectangle
        if len(r) > 1:
            x2, y2, w2, h2 = r[1]

            if (x1 - 0.5 * w1 <= x2 and x2 <= x1 + w1) and (y1 - 0.5 * h1 <= y2 and y2 <= y1 + h1):
                repeat = True

            # Draw the second rect. if not a repeat
            # Also draw if the second rect is not vertical and not small
            if repeat is False:
                cv2.rectangle(
                    img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                finalRect.append(r[1])

        if len(r) > 2:
            x3, y3, w3, h3 = r[2]
            if (x1 - 0.5 * w1 <= x3 and x3 <= x1 + w1) and\
                (y1 - 0.5 * h1 <= y3 and y3 <= y1 + h1):
                repeat = True

            if repeat is False:
                if w3 > 100:
                    cv2.rectangle(
                        img, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)
                    finalRect.append(r[-1])
        # Save the result
        cv2.imwrite(fname, img)

    return finalRect


def calcLength(rect):
    '''Return the length of the rectangle by 
    finding the max value between the rect's width and height'''

    if len(rect) == 3:
        x, y = rect[0]
        w, h = rect[1]

    else:
        x, y, w, h = np.array(rect).ravel()
        
    length = max(w, h)
    width = min(w, h)

    return length, width

def findDistance(L, R, imsize, stereoCams, folder):
    '''Call findRectangles to locate the two largest
    handle rectangles
    @L = L img
    @R = R img

    return = computed L and R rectangles, and EDGE location
    '''
    x, y, z = splitString(L)
    location = str(x) + '_' + str(y) + '_' + str(z)
    print location

    # Find rectangles
    rectL, rectR = findRectangle(stereoCams,
                                 folder, imsize, L, R, location)

    return rectL, rectR, location


def distLengthPredict(stereoCams, length, obj_real_world):
    '''Compute the estimated distance to the handle
    using its actual real world length (mm) and its estimated 
    rectangle length (pixels)
    @length = length of handle as predicted by the handle's rectangleL
    @obj_real_world = actual ISS length of the corresponding handle

    return = estimated distance to the ISS
    '''
    # Compute the calibrated focal length using both L and R
    # camera matrices
    fL = np.mean([stereoCams.M1[0][0], stereoCams.M1[1][1]])
    fR = np.mean([stereoCams.M2[0][0], stereoCams.M2[1][1]])
    f1 = np.mean([fL, fR])

    f_img = 35.  # Actual Focal Length
    m = f1 / f_img  # Scaling ratio
    # Length of image on the camera sensor
    sensor = length / m

    # Estimate the distance using the pinhole projection model
    distance = obj_real_world * f_img / sensor

    # convert from mm to inches
    distance = (distance / 10.) / 2.54

    return distance

def calcDistWithWidth(stereoCams, rect):
    distanceW, distanceL = [], []
    width, length = [], []
    for r in rect:
        l, w = calcLength(r)
        width.append(w)
        length.append(l)

        # Used as a width test for EVA handrails
        # ratio = float(w) / l
        # if ratio > 0.035:
        #     width.append(w)
        
    for w in width:
        # Actual decal width
        actual_width = 1.50 * 25.4 #convert width from inch to mm
        dw = distLengthPredict(stereoCams, w, actual_width)
        distanceW.append(dw)

    for l in length:
        # Actual decal length
        actual_length = 3.75 * 25.4
        dl = distLengthPredict(stereoCams, l, actual_length)
        distanceL.append(dl)

    return distanceW, distanceL

def findAverage(left, right):
    avg = []
    avg_left, avg_right = [], []

    for l, r in zip(left, right):
        finite_L = np.sum(np.isfinite(l))
        finite_R = np.sum(np.isfinite(r))

        if finite_L > 0 and finite_R > 0:
            avg.append(np.mean([np.mean(l), np.mean(r)]))

        elif finite_L > 0 and finite_R == 0:
            avg.append(np.mean(l))

        else:
            avg.append(np.mean(r))

        avg_left.append(np.mean(l))
        avg_right.append(np.mean(r))

    return avg, avg_left, avg_right

def applyKalmanFilter(Q, R, vec):
    # # Create Kalman Filter
    # Q = process variance
    # R = measurement variance
    kf = KalmanFilter(Q, R)
    kf_distance = []

    for i in vec:
        # Update value using KF if it's finite
        if np.isfinite(i):
            kf.update(i)
            new_dist = kf.predict()
            kf_distance.append(new_dist)
        else:
            kf_distance.append(None)

    return kf_distance

def boxPlotData(data):
    data.boxplot(sym = '')
    plt.title('Estimated Error')
    plt.xlabel('Relative Distance Approximations')
    plt.ylabel('Error (Percentage)')
    plt.savefig('distance/24inch/boxplot_box.pdf')
    plt.show()

imsize, stereoCams = stereoCalibration()

imgsL = sorted(glob.glob("distance/24inch/left/left_*.jpeg"))
imgsR = sorted(glob.glob("distance/24inch/right/right_*.jpeg"))
# handle dimensions (number), (length), (width)
# unit = inches

folder = "distance/24inch/filtered"
rectangleL, rectangleR = [], []
width_L, width_R = [], []
length_R, length_L = [], []
avg = []

for L, R in zip(imgsL, imgsR):

    x, y, z = splitString(L)
    loc = str(x) + '_' + str(y) + '_' + str(z)

    rectL, rectR, loc = findDistance(L, R, imsize, stereoCams, folder)
    rL_name = imageName(folder, 'rectify_imgL' + loc + '.png')
    rR_name = imageName(folder, 'rectify_imgR' + loc + '.png')

    # print 'left images'
    width_dist_L, length_dist_L = \
        calcDistWithWidth(stereoCams, rectL)

    # print 'right images'
    width_dist_R, length_dist_R = calcDistWithWidth(stereoCams, rectR)

    rectangleR.append(rectR)
    rectangleL.append(rectL)

    width_R.append(width_dist_R)
    length_R.append(length_dist_R)

    width_L.append(width_dist_L)
    length_L.append(length_dist_L)

    avg_w, avg_wL, avg_wR = findAverage(width_L, width_R)
    avg_l, avg_lL, avg_lR = findAverage(length_L, length_R)

    print '############################################'

# # Create Kalman Filter
Q, R = 1., 200.
actual = [24] * len(filter_avg_l)

# Filtered Width Approximations
filter_avg_w = applyKalmanFilter(Q, R, avg_w)
filter_avg_wL = applyKalmanFilter(Q, R, avg_wL)
filter_avg_wR = applyKalmanFilter(Q, R, avg_wR)

plt.plot(filter_avg_w, 'b-', label = 'filtered average')
plt.plot(filter_avg_wL, 'r-', label = 'filtered left')
plt.plot(filter_avg_wR, 'g-', label = 'filtered right')
plt.ylabel('Relative Distance (inches)')
plt.xlabel('Trial')
plt.title('Filtered Average Relative Distance - Width Approximation')
plt.legend(loc = 3)
plt.ylim([10., 25.])
plt.savefig('filtered_avg_width.pdf')
plt.show()

# Filtered Length Approximations
filter_avg_l = applyKalmanFilter(Q, R, avg_l)
filter_avg_lL = applyKalmanFilter(Q, R, avg_lL)
filter_avg_lR = applyKalmanFilter(Q, R, avg_lR)

plt.plot(filter_avg_l, 'b-', label = 'filtered average')
plt.plot(filter_avg_lL, 'r-', label = 'filtered left')
plt.plot(filter_avg_lR, 'g-', label = 'filtered right')
plt.ylabel('Relative Distance (inches)')
plt.xlabel('Trial')
plt.title('Filtered Average Relative Distance - Length Approximation')
plt.legend(loc = 3)
plt.ylim([10., 25.])
plt.savefig('filtered_avg_length.pdf')
plt.show()

# plt.plot(avg_w, label = 'width avg')
# plt.plot(avg_wL, label = 'width L')
# plt.plot(avg_wR, label = 'width R')
# plt.ylabel('Relative Distance (inches)')
# plt.xlabel('Trial')
# plt.title('Average Relative Distance - Width Approximation')
# plt.ylim([10., 25.])
# plt.xlim([0., 74.])
# plt.yticks(np.arange(0, 24., 1.0))
# plt.legend(loc = 3)
# plt.savefig('avg_dist_width.pdf')
# plt.show()

# plt.plot(avg_l, label = 'length avg')
# plt.plot(avg_lL, label = 'length L')
# plt.plot(avg_lR, label = 'length R')
# plt.ylabel('Relative Distance (inches)')
# plt.xlabel('Trial')
# plt.title('Average Relative Distance - Length Approximation')
# plt.ylim([10., 25.])
# plt.xlim([0., 74.])
# plt.yticks(np.arange(0, 24., 1.0))
# plt.legend(loc = 3)
# plt.savefig('avg_dist_length.pdf')
# plt.show()

plt.plot(actual, 'g', label = 'actual rel distance')
plt.plot(avg_w, 'r--', label = 'width avg')
plt.plot(filter_avg_w, 'r-', label = 'filtered width')
plt.plot(avg_l, 'b--', label = 'length avg')
plt.plot(filter_avg_l, 'b-', label = 'filtered length')
plt.ylabel('Relative Distance (inches)')
plt.xlabel('Trial')
plt.title('Average Relative Distance')
plt.ylim([10., 25.])
plt.xlim([0., 74.])
plt.yticks(np.arange(0, 24., 1.0))
plt.legend(loc = 3)
plt.savefig('filtered_width_v_length.pdf')
plt.show()

# Error stats ###############################################3
actual = [24] * len(filter_avg_l)
actual = np.array(actual)
filter_avg_l = np.array(filter_avg_l)
filter_avg_w = np.array(filter_avg_w)
avg_l = np.array(avg_l)
avg_w = np.array(avg_w)

filter_error_l = 1. - filter_avg_l / actual
filter_error_w = 1. - filter_avg_w / actual 
error_l = 1. - avg_l / actual
error_w = 1. - avg_w / actual

fe_w = pd.DataFrame(filter_error_w)#.describe()
fe_l = pd.DataFrame(filter_error_l)#.describe()
e_w = pd.DataFrame(error_w)#.describe()
e_l = pd.DataFrame(error_l)#.describe()

result = pd.concat([fe_w, e_w, fe_l, e_l], axis = 1, \
    join_axes = [fe_w.index])
result.columns = ['filtered_w', 'unfiltered_w', \
        'filtered_l', 'unfiltered_l']
result *= 100.
boxPlotData(result)