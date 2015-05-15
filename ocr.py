import sys
import cv2
# import skimage.morphology as morphology
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

    imgL = cv2.imread(L_name)
    imgR = cv2.imread(R_name)
    r_imgL, r_imgR = rectifyImage(imgL, imgR, imsize, stereoCams, folder)

    h1 = imageName(folder,'handlesL.png')
    h2 = imageName(folder,'handlesR.png')
    hL_name = imageName(folder, 'filtered_ellipse_L' + loc + '.png')
    hR_name = imageName(folder, 'filtered_ellipse_R' + loc + '.png')

    rL_name = imageName(folder, 'rectify_imgL.png')
    rR_name = imageName(folder, 'rectify_imgR.png')
    imgL = cv2.imread(rL_name)
    imgR = cv2.imread(rR_name)

    # Compute gradient and threshold for L and R imgs
    handlesL = find_handles(imgL, h1)
    h_img = cv2.imread(h1)

    handlesR = find_handles(imgR, h2)
    r_img = cv2.imread(h2)

    # Find rectangles from gradient images
    rectL, cts = compute_threshold(handlesL, h_img.copy(), hL_name)
    rectR, cts_r = compute_threshold(handlesR, r_img.copy(), hR_name)

    return rectL, rectR, r_imgL, r_imgR

def calcLength(rect):
    length_vec = []

    if(len(rect) > 1):
        for r in rect:
            x, y, w, h = r
            length = max(w, h)
            length_vec.append(length)
        length = np.mean(length_vec)

    else:
        x, y, w, h = np.array(rect).ravel()
        length = max(w, h)

    return length

def findDistance(imgsL, imgsR, imsize, stereoCams, folder):
    # imsize, stereoCams = stereoCalibration()

    folder1 = folder + '/'
    output, dist, a = [], [], []

    # for L, R in zip(imgsL, imgsR):
    x, y, z, p, yw, r = splitString(L)
    location = str(x) + '_' + str(y) + '_' + str(z)
    print location
    # name = folder1 + location + '.png'

    rectL, rectR, l_img, r_img = findRectangle(stereoCams, \
        folder, imsize, L, R, location)

    # Determine handle length
    lengthL = calcLength(rectL)
    lengthR = calcLength(rectR)

    # disparity between the left and right image 
    # dL, points = computeDisparity(r_img, l_img, stereoCams)

    return lengthL, lengthR, rectL, rectR, location

def distLengthPredict(stereoCams, length, obj_real_world):
    '''Compute the estimated distance to the handle
    using its actual real world length (mm) and its estimated 
    rectangle length (pixels)'''

    f1 = np.mean([stereoCams.M1[0][0], stereoCams.M2[0][0]])
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
        plt.imshow(mask)
        plt.show()

        dst = mask - im2
        dst[dst[:, ] == 1] = 0
        cnt = cv2.findContours(
            dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50

        cv2.drawContours(dst, cnt, -1, 0, -1)
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


def TesseractOCR(images):
    digit = []

    for i in images:
        digit_str = (pytesseract.image_to_string(Image.open(i)))
        print digit_str
        digit_str = digit_str.split(' ')

        for j in digit_str:
            if j.isdigit() is True:
                digit.append(int(j))

    return digit


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

        cv2.imwrite(cL_name, cropped)

        cropped_imgs.append(cropped)

    return cropped_imgs, name_vec


def matchHandle(digit):

    # convert inches to mm
    convert = 25.4
    w = 1.38 * convert
    handles = [[1141, 8. * convert, w],
               [1150, 16. * convert, w],
               [1156, 16. * convert, w],
               [1161, 16. * convert, w],
               [1169, 16. * convert, w],
               [1174, 16. * convert, w],
               [1140, 8. * convert, w],
               [1149, 16. * convert, w],
               [1155, 16. * convert, w],
               [1160, 16. * convert, w],
               [1169, 8. * convert, w],
               [1173, 8. * convert, w],
               [1186, 16. * convert, w],
               [1212, 8. * convert, w]
               ]

    matches = []
    for d in digit:
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

    elif len(distanceR) == 0 and len(distanceR) != 0:
        for d in distanceR:
            print 'Handle: ', d[0], 'Distance: ', d[1]

    elif len(distanceR) != 0 and len(distanceL) != 0:
        if len(distanceR) > len(distanceL):
            for dR in distanceR:
                idx = []
                idx = [i for i, x in enumerate(distanceL) if x[0] == dR[0]]
                
                # Match found
                if len(idx) > 0:
                    print 'Handle: ', dR[0], ' Distance: ', \
                    np.mean([dR[-1], distanceL[idx[0]][-1] )

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
                    np.mean([dL[-1], distanceR[idx[0]][-1]] )

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

for L, R in zip(imgsL, imgsR):
    lengthL, lengthR, rectL, rectR, loc = findDistance(L,
                                                       R, imsize, stereoCams, folder)
    rL_name = imageName(folder, 'rectify_imgL.png')
    rR_name = imageName(folder, 'rectify_imgR.png')

    cL_name = imageName(folder, 'cropL_' + loc)
    cR_name = imageName(folder, 'cropR_' + loc)

    print 'left images'
    croppedL, cropL = cropImage(rL_name, rectL, cL_name, 'L')
    cntL = prepareImage(cropL)
    # digitL = TesseractOCR(cntL)

    print 'right images'
    croppedR, cropR = cropImage(rR_name, rectR, cR_name, 'R')
    cntR = prepareImage(cropR)
    # digitR = TesseractOCR(cntR)

    digitL = [1141]
    digitR = [1141]

    if len(digitL) is not 0:
        # Match the handle to info in the database
        handlePropertiesL = matchHandle(digitL)

        # Predict the distance to the handle
        distanceL, distanceR = [], []
        for h in handlePropertiesL:
            n, l, w = h
            obj_real_world = l
            d = distLengthPredict(stereoCams, lengthL, obj_real_world)
            distanceL.append(np.array([n, d]))


    if len(digitR) is not 0:
        handlePropertiesR = matchHandle(digitR)
        for h in handlePropertiesR:
            n, l, w = h
            obj_real_world = l
            d = distLengthPredict(stereoCams, lengthR, obj_real_world)
            distanceR.append(np.array([n, d]))

    # dL = np.mean(distanceL)
    # dR = np.mean(distanceR)
    computeAvgDistance(distanceL, distanceR)

    break

    # break
    # distanceL = distLengthPredict(stereoCams, lengthL, obj_real_world)
    # distanceR = distLengthPredict(stereoCams, lengthR, obj_real_world)

    # distance = np.mean([distanceL, distanceR])