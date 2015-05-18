import sys
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import glob
from scipy import ndimage
from find_rectangles import *
from stereo_calibrate import *


# try:
#     import Image
# except ImportError:
#     from PIL import Image
# import pytesseract

def imageName(folder, name):
    if len(folder) != 0:
        imgName = folder + '/' + name
    else:
        imgName = name

    return imgName

def splitString(name):
    vals = name.split('_')
    y = vals[-1].split('.')[0]
    val1 = [float(vals[-6]), float(vals[-5]), float(vals[-4]), \
        float(vals[-3]), float(vals[-2]), float(y)]

    return val1

def findRectangle(stereoCams, folder, imsize, L_name, R_name, loc):
    '''Locate ellipses in a set of images'''

    imgL_unrect = cv2.imread(L_name)
    imgR_unrect = cv2.imread(R_name)
    left = 'rectify_imgL' + loc + '.png'
    right = 'rectify_imgR' + loc + '.png'
    r_imgL, r_imgR = rectifyImage(imgL_unrect, imgR_unrect, imsize, stereoCams, folder, right, left)

    h1 = imageName(folder,'handlesL.png')
    h2 = imageName(folder,'handlesR.png')
    hL_name = imageName(folder, 'filtered_ellipse_L' + loc + '.png')
    hR_name = imageName(folder, 'filtered_ellipse_R' + loc + '.png')

    rL_name = imageName(folder, 'rectify_imgL' + loc + '.png')
    rR_name = imageName(folder, 'rectify_imgR' + loc + '.png')
    imgL = cv2.imread(rL_name)
    imgR = cv2.imread(rR_name)

    # Compute gradient and threshold for L and R imgs
    handlesL = find_handles(imgL, h1)
    h_img = cv2.imread(h1)

    handlesR = find_handles(imgR, h2)
    r_img = cv2.imread(h2)

    # Find rectangles from gradient images
    rectL = compute_threshold(h_img.copy(), hL_name, False)
    rectR = compute_threshold(r_img.copy(), hR_name, False)

    return rectL, rectR

def calcLength(rect):

    x, y, w, h = np.array(rect).ravel()
    length = max(w, h)

    return length

def findDistance(L, R, imsize, stereoCams, folder):

    x, y, z, p, yw, r = splitString(L)
    location = str(x) + '_' + str(y) + '_' + str(z)
    print location

    # Find rectangles
    rectL, rectR = findRectangle(stereoCams,
                                 folder, imsize, L, R, location)

    # disparity between the left and right image
    # dL, points = computeDisparity(r_img, l_img, stereoCams)

    return rectL, rectR, location

def distLengthPredict(stereoCams, length, obj_real_world, side):
    '''Compute the estimated distance to the handle
    using its actual real world length (mm) and its estimated 
    rectangle length (pixels)'''
    # if side == 1:
    fL = np.mean([stereoCams.M1[0][0], stereoCams.M1[1][1]])
    # elif side == 2:
    fR = np.mean([stereoCams.M2[0][0], stereoCams.M2[1][1]])
    f1 = np.mean([fL, fR])
    f_img = 35. #Actual Focal Length
    m = f1 / f_img
    sensor = length / m

    distance = obj_real_world * f_img / sensor

    # convert from mm to inches
    distance = (distance / 10.) / 2.54

    return distance


def prepareImage(images):

    handle_contour = []

    for i, name in zip(range(len(images)), images):
        im = cv2.imread(name)
        x, y, z = name.split('_')[1:4]
        im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        im2 = cv2.adaptiveThreshold(
            im1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)
        cv2.floodFill(im2, None, (0, 0), 255)

        im5 = cv2.erode(
            255 - im2.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)

        cnt = cv2.findContours(
            im5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnt = [c for c in cnt if (cv2.contourArea(c) > 800)]  # > 1000

        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.drawContours(mask, cnt, -1, 255, -1)
        # cv2.imshow('', mask)
        # cv2.waitKey()
        # plt.imshow(mask)
        # plt.show()

        dst = mask - im2
        dst[dst[:, ] == 1] = 0
        cnt = cv2.findContours(
            dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50

        cv2.drawContours(dst, cnt, -1, 0, -1)
        cv2.floodFill(dst, None, (0, 0), 255)

        h, w = dst.shape
        if abs(np.sum(dst == 255) - (h * w)) <= 100:
            dst = 255 - im2 - mask
            dst[dst[:, ] == 1] = 0
            cnt = cv2.findContours(
                dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50

            cv2.drawContours(dst, cnt, -1, 0, -1)
            cv2.floodFill(dst, None, (0, 0), 255)
            dst = 255 - dst
            cv2.floodFill(dst, None, (0, 0), 255)

        img_contour = "ocr/contour_results/skel_contour_" + str(x) + "_" + \
            str(y) + "_" + (z) + "rect_" + str(i) + ".png"

        cv2.imwrite(img_contour, dst)

        handle_contour.append(img_contour)
        plt.imshow(dst)
        plt.show()
        # cv2.imshow('regfill', dst)
        # cv2.waitKey()

        return handle_contour


def TesseractOCR(images, rect):
    digit, rectMatch = [], []

    for i, r in zip(images, rect):
        digit_str = (pytesseract.image_to_string(Image.open(i)))
        print digit_str
        digit_str = digit_str.split(' ')

        for j in digit_str:
            if j.isdigit() is True:
                digit.append(int(j))
                length = calcLength(r)
                rectMatch.append(length)

    return digit, rectMatch

def smallerRectImage(img_name):
    im = cv2.imread(img_name)
    h, w, _ = im.shape
    q = h/4
    crop = im[10 + q: 3 * q, 0.2 * w :w * 0.8]
    new_rect = compute_threshold(crop, img_name, True)

    if len(new_rect) > 1:
        x, y, w, h = new_rect[-1]
        if h > w:
            x, y, w, h = new_rect[0]

    else:
        x, y, w, h = new_rect[0]
    crop2 = crop[y: y + h, x + 0.1 * w: x + 0.9 * w]
    cv2.imwrite(img_name, crop2)

    cv2.imshow('', crop2)

    return img_name

def cropImage(img_name, rect, name, direc):
    img = cv2.imread(img_name)
    cropped_imgs = []
    name_vec = []

    for i, r in zip(range(len(rect)), rect):
        x, y, w, h = r
        x1 = int(x - .5 * w)
        x2 = int(x + .5 * w)

        if x1 < 0:
            x1 = 0

        cropped = img[y: y + h, x: x + w]
        cL_name = name + direc + '_rect' + str(i) + '.png'
        name_vec.append(cL_name)

        if h > w:
            cropped = ndimage.rotate(cropped, -90)

        cv2.imwrite(cL_name, cropped)

        cropped_imgs.append(cropped)

    return cropped_imgs, name_vec


def matchHandle(digit):

    # convert inches to mm
    convert = 25.4
    w = 1.38 * convert
    handles = [[1141, 8.64 * convert, w],
               [1150, 22.7 * convert, w],
               [1156, 22.7 * convert, w],
               [1161, 22.7 * convert, w],
               [1169, 22.7 * convert, w],
               [1174, 22.7 * convert, w],
               [1140, 8.64 * convert, w],
               [1149, 22.7 * convert, w],
               [1155, 22.7 * convert, w],
               [1160, 22.7 * convert, w],
               [1166, 8.64 * convert, w],
               [1169, 8.64 * convert, w],
               [1173, 8.64 * convert, w],
               [1186, 22.7 * convert, w],
               [1212, 8.64 * convert, w]
               ]

    matches = []
    for d in digit:
        if len([i for i, x in enumerate(handles) if x[0] == d]) > 0:
            idx = [i for i, x in enumerate(handles) if x[0] == d][0]
            matches.append(np.array(handles[idx]))

    return matches

def computeAvgDistance(distanceL, distanceR):
    if len(distanceL) is 0:
        print 'No left handle matches'

    if len(distanceR) is 0:
        print 'No right handle matches'

    if len(distanceL) == 0 and len(distanceR) != 0:
        for d in distanceR:
            print 'Handle: ', d[0], 'Distance: ', d[1]

    elif len(distanceR) == 0 and len(distanceL) != 0:
        for d in distanceL:
            print 'Handle: ', d[0], 'Distance: ', d[1]

    elif len(distanceR) != 0 and len(distanceL) != 0:
        if len(distanceR) > len(distanceL):
            for dR in distanceR:
                idx = []
                idx = [i for i, x in enumerate(distanceL) if x[0] == dR[0]]

                # Match found
                if len(idx) > 0:
                    print 'Handle: ', dR[0], ' Distance: ', \
                        np.mean([dR[-1], distanceL[idx[0]][-1]])

                # No Match
                else:
                    print 'Handle: ', dR[0], ' Distance: ', dR[1]

        elif len(distanceL) > len(distanceR):
            for dL in distanceL:
                idx = []
                idx = [i for i, x in enumerate(distanceR) if x[0] == dL[0]]

                # Match found
                if len(idx) > 0:
                    print 'Handle: ', dL[0], ' Distance: ', \
                        np.mean([dL[-1], distanceR[idx[0]][-1]])

                # No Match
                else:
                    print 'Handle: ', dL[0], ' Distance: ', dL[1]
        else:
            index = []
            for dL in distanceL:
                idx = []
                idx = [i for i, x in enumerate(distanceR) if x[0] == dL[0]]
                index.append(idx[0])

                # Match found
                if len(idx) > 0:
                    print 'Handle: ', dL[0], ' Distance: ', \
                        np.mean([dL[-1], distanceR[idx[0]][-1]])

                # No Match
                else:
                    print 'Handle: ', dL[0], ' Distance: ', dL[1]

            if len(index) != len(distanceR):
                for i in range(len(distanceR)):
                    if i not in j is True:
                        print 'Handle: ', distanceR[i][0], ' Distance: ',
                        distanceR[i][1]

imsize, stereoCams = stereoCalibration()

images = glob.glob("ocr/input/25/handle_*.jpeg")
imgsL = glob.glob("ocr/input/25/left/left_*.jpeg")
imgsR = glob.glob("ocr/input/25/right/right_*.jpeg")
# handle dimensions (number), (length), (width)
# unit = inches


folder = "ocr/input/25/joint"
rectangleL, rectangleR = [], []
for L, R in zip(imgsL, imgsR):

    rectL, rectR, loc = findDistance(L, R, imsize, stereoCams, folder)
    rL_name = imageName(folder, 'rectify_imgL' + loc + '.png')
    rR_name = imageName(folder, 'rectify_imgR' + loc + '.png')

    cL_name = imageName(folder, 'cropL_' + loc)
    cR_name = imageName(folder, 'cropR_' + loc)

    print 'left images'
    croppedL, cropL_name = cropImage(rL_name, rectL, cL_name, 'L')
    cntL = prepareImage(cropL_name)
    # digitL, lengthL = TesseractOCR(cntL)

    # If no digit was found, crop image to approx. tag size
    if len(digitL) is 0:
        for i in cntL:
            cntL_new = smallerRectImage(i)

        # digitL, lengthL = TesseractOCR(cntL_new)

    print 'right images'
    croppedR, cropR_name = cropImage(rR_name, rectR, cR_name, 'R')
    cntR = prepareImage(cropR_name)
    # digitR, lengthR = TesseractOCR(cntR)

    # If no digit was found, crop image to approx. tag size
    if len(digitR) is 0:
        for i in cntR:
            cntR_new = smallerRectImage(i)

        # digitR, lengthR = TesseractOCR(cntR_new)

    # digitL = [1141]
    # digitR = [1141]
    # distanceL, distanceR = [], []
    # side = 0.
    # if len(digitL) is not 0:
    #     # Match the handle to info in the database
    #     handlePropertiesL = matchHandle(digitL)

    #     # left side indicator
    #     side = 1
    #     # Predict the distance to the handle
    #     for h, length in zip(handlePropertiesL, lengthL):
    #         n, l, w = h
    #         obj_real_world = l
    #         d = distLengthPredict(stereoCams, length, obj_real_world, side)
    #         distanceL.append(np.array([n, d]))

    # if len(digitR) is not 0:
    #     handlePropertiesR = matchHandle(digitR)

    #     # right side indicator
    #     side = 2
    #     for h, length in zip(handlePropertiesR, lengthR):
    #         n, l, w = h
    #         obj_real_world = l
    #         d = distLengthPredict(stereoCams, length, obj_real_world, side)
    #         distanceR.append(np.array([n, d]))

    # computeAvgDistance(distanceL, distanceR)
    # print '############################################'
    # break

im = cv2.imread(cntR[0])
h, w, _ = im.shape
q = h/4
crop = im[q:2*q + 10, 0:w]
