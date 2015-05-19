import sys
import cv2
from cv2 import cv
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
    val1 = [float(vals[-6]), float(vals[-5]), float(vals[-4]), \
        float(vals[-3]), float(vals[-2]), float(y)]

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
    r_imgL, r_imgR = rectifyImage(imgL_unrect, imgR_unrect, imsize, stereoCams, folder, right, left)

    # Create all image names needed
    h1 = imageName(folder,'handlesL.png')
    h2 = imageName(folder,'handlesR.png')
    hL_name = imageName(folder, 'filtered_ellipse_L' + loc + '.png')
    hR_name = imageName(folder, 'filtered_ellipse_R' + loc + '.png')

    rL_name = imageName(folder, 'rectify_imgL' + loc + '.png')
    rR_name = imageName(folder, 'rectify_imgR' + loc + '.png')
    imgL = cv2.imread(rL_name)
    imgR = cv2.imread(rR_name)

    # Compute handle gradient and threshold for L and R imgs
    handlesL = find_handles(imgL, h1)
    h_img = cv2.imread(h1)

    handlesR = find_handles(imgR, h2)
    r_img = cv2.imread(h2)

    # Find rectangles from gradient images
    rectL = compute_threshold(h_img.copy(), hL_name, False)
    rectR = compute_threshold(r_img.copy(), hR_name, False)

    return rectL, rectR

def calcLength(rect):
    '''Return the length of the rectangle by 
    finding the max value between the rect's width and height'''

    x, y, w, h = np.array(rect).ravel()
    length = max(w, h)

    return length

def findDistance(L, R, imsize, stereoCams, folder):
    '''Call findRectangles to locate the two largest
    handle rectangles
    @L = L img
    @R = R img

    return = computed L and R rectangles, and EDGE location
    '''
    x, y, z, p, yw, r = splitString(L)
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

    f_img = 35. #Actual Focal Length  
    m = f1 / f_img # Scaling ratio
    # Length of image on the camera sensor
    sensor = length / m 

    # Estimate the distance
    distance = obj_real_world * f_img / sensor

    # convert from mm to inches
    distance = (distance / 10.) / 2.54

    return distance


def prepareImage(images):
    '''Prepare the image for OCR. These images must have a 
    minimal amount of noise in order to attain optimal results.
    Return the image names for all handle contours'''

    handle_contour = []

    for i, name in zip(range(len(images)), images):
        # Read in image and convert to BW
        im = cv2.imread(name)
        x, y, z = name.split('_')[1:4]
        im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Apply a threshold and fill in all closed curves
        im2 = cv2.adaptiveThreshold(
            im1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)
        cv2.floodFill(im2, None, (0, 0), 255)

        im5 = cv2.erode(
            255 - im2.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)

        # Find the contours
        cnt = cv2.findContours(
            im5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        # For the mask, keep all contours larger than 800
        cnt = [c for c in cnt if (cv2.contourArea(c) > 800)]  # > 1000
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.drawContours(mask, cnt, -1, 255, -1)

        # Subtract the threshold image from the mask
        dst = mask - im2
        dst[dst[:, ] == 1] = 0

        # Keep all small contours and fill in closed curves
        cnt = cv2.findContours(
            dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50

        cv2.drawContours(dst, cnt, -1, 0, -1)
        cv2.floodFill(dst, None, (0, 0), 255)

        h, w = dst.shape

        # If there is mostly white pixels and no handle tag,
        # reparse the image to keep more contours
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

        # Save the image names 
        handle_contour.append(img_contour)
        plt.imshow(dst)
        plt.show()

        return handle_contour


def TesseractOCR(images, rect):
    '''Use pytesseract to read the handle numbers and keep the 
    rectangles (aka handle lengths) that contained handle numbers 
    that were read successfully 

    return = handle numbers and its corresponding rect estimates
    '''
    digit, rectMatch = [], []

    for i, r in zip(images, rect):
        # Read all text from the image
        digit_str = (pytesseract.image_to_string(Image.open(i)))
        print digit_str
        digit_str = digit_str.split(' ')

        # Keep only digits from the image
        for j in digit_str:
            if j.isdigit() is True:
                digit.append(int(j))
                length = calcLength(r)
                rectMatch.append(length)

    return digit, rectMatch

def smallerRectImage(img_name):
    '''Recrop the image to the smaller handle tag

    return = name of the updated handle image
    '''
    # Read in the image
    # crop out extra noise before extracting new rect
    im = cv2.imread(img_name)
    h, w, _ = im.shape
    q = h/4
    crop = im[10 + q: 3 * q, 0.2 * w :w * 0.8]

    # Save cropped result
    cv2.imwrite(img_name, crop)

    # fxn subimage needs a cv image
    img = cv.LoadImage(img_name)

    rect = compute_threshold(crop, img_name, True)

    for i, r in zip(range(len(rect)), rect):
        centre = r[0]
        w, h = np.int0(r[1])
        theta = np.radians(r[-1])

        # Crop image and save image name
        cropped = subimage(img, centre, theta, w, h)
        cL_name = img_name
        cv.SaveImage(cL_name, cropped)

        # check if the image needs to be rotated
        # if so, read in a cv2 image
        if h > w:
            rot_img = cv2.imread(cL_name)
            cv2.imwrite(cL_name, ndimage.rotate(rot_img, 90))

    return img_name

def subimage(image, centre, theta, width, height):
    '''Crop the image to fit the rotated rectangle found in smallerRectImage()

    @image = image to be cropped
    @centre = center of the rotated rectangle
    @theta = angle of rotation for the rotated rect (in radians)
    @width = width of rotated rect
    @height = height of rotated rect

    return = modified cropped /rotated image
    '''

    # Create the image using the inputted params
    output_image = cv.CreateImage((width, height), image.depth, \
        image.nChannels)

    # Create the new rotation mapping matrix
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                        [np.sin(theta), np.cos(theta), centre[1]]])

    # Crop the image
    map_matrix_cv = cv.fromarray(mapping)
    cv2.cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
    
    return output_image

def cropImage(img_name, rect, name, direc):
    '''Crop the image down to the approx. handle size
    @img_name = img to be cropped
    @rect = handle rect that contains the cropping dimensions

    return = vector of all cropped images
    '''
    img = cv2.imread(img_name)
    name_vec = []

    for i, r in zip(range(len(rect)), rect):
        x, y, w, h = r

        # Crop image and save image name
        cropped = img[y: y + h, x: x + w]
        cL_name = name + direc + '_rect' + str(i) + '.png'
        name_vec.append(cL_name)

        if h > w:
            cropped = ndimage.rotate(cropped, -90)

        # Save image
        cv2.imwrite(cL_name, cropped)

    return name_vec

def lastParseAttempt(cnt_name, rect):
    '''If the background of the handle tag is too dark 
    for Tesseract, reverse the tag's colors. This will likely make
    the background white, and the handle number black, allowing Tesseract to 
    have a higher probablity of successful detection

    return = the calculated digit and corresponding handle length
    '''
    img = cv2.imread(cnt_name)
    cv2.imwrite(cnt_name, 255 - img)

    digit, length = TesseractOCR(cnt_name, rect)
    return digit, length

def matchHandle(digit):
    '''Match the ocr-read handle number to the existing
    handle number database
    @digit = OCR detected number

    return = matched rectangle(s) if the digit was successfully matched
        using the handle database
    '''

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
        # if a match is found, save the index and handle values
        if len([i for i, x in enumerate(handles) if x[0] == d]) > 0:
            idx = [i for i, x in enumerate(handles) if x[0] == d][0]
            matches.append(np.array(handles[idx]))

    return matches

def computeAvgDistance(distanceL, distanceR):
    '''Compute the average distance to the ISS
    @distanceL = estimated distance in the left img
    @distanceR = estimated distance in the right img
    '''

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

    # If there are both successful L and R handle matches
    # utilize the appropriate scheme
    elif len(distanceR) != 0 and len(distanceL) != 0:
        if len(distanceR) > len(distanceL):
            for dR in distanceR:
                idx = []
                idx = [i for i, x in enumerate(distanceL) if x[0] == dR[0]]

                # Match found between R and L
                if len(idx) > 0:
                    print 'Handle: ', dR[0], ' Distance: ', \
                        np.mean([dR[-1], distanceL[idx[0]][-1]])

                # No Match - Print R Handle result
                else:
                    print 'Handle: ', dR[0], ' Distance: ', dR[1]

        elif len(distanceL) > len(distanceR):
            for dL in distanceL:
                idx = []
                idx = [i for i, x in enumerate(distanceR) if x[0] == dL[0]]

                # Match found between L and R
                if len(idx) > 0:
                    print 'Handle: ', dL[0], ' Distance: ', \
                        np.mean([dL[-1], distanceR[idx[0]][-1]])

                # No Match - Print L Handle result
                else:
                    print 'Handle: ', dL[0], ' Distance: ', dL[1]

        # If there are an equal number of L and R matches...
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

            # Print any remaining R handle matches
            if len(index) != len(distanceR):
                for i in range(len(distanceR)):
                    if i not in j is True:
                        print 'Handle: ', distanceR[i][0], ' Distance: ',
                        distanceR[i][1]

#######################################################################################3
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
    cropL_name = cropImage(rL_name, rectL, cL_name, 'L')
    cntL = prepareImage(cropL_name)
    # digitL, lengthL = TesseractOCR(cntL, rectL)

    # If no digit was found, crop image to approx. tag size
    if len(digitL) is 0:
        cntL_new = []
        for i in cntL:
            cnt = smallerRectImage(i)
            cntL_new.append(cnt)

        # digitL, lengthL = TesseractOCR(cntL_new, rectL)

        # If Tesseract can't recognize the number, reverse its color
        if len(digitL) is 0:
            digitL, lengthL = lastParseAttempt(cntL_new[0], rectL)

    print 'right images'
    cropR_name = cropImage(rR_name, rectR, cR_name, 'R')
    cntR = prepareImage(cropR_name)
    # digitR, lengthR = TesseractOCR(cntR, rectR)

    # If no digit was found, crop image to approx. tag size

    if len(digitR) is 0:
        cntR_new = []
        for i in cntR:
            cnt = smallerRectImage(i)
            cntR_new.append(cnt)

        # digitR, lengthR = TesseractOCR(cntR_new, rectR)

        # If Tesseract can't recognize the number, reverse its color
        if len(digitR) is 0:
            digitR, lengthR = lastParseAttempt(cntR_new[0], rectR)

    rectangleR.append(rectR)
    rectangleL.append(rectL)
    plt.close('all')
    # digitL = [1141]
    # digitR = [1141]
    # distanceL, distanceR = [], []
    # side = 0.
    # if len(digitL) is not 0:
    #     # Match the handle to info in the database
    #     handlePropertiesL = matchHandle(digitL)

    #     # Predict the distance to the handle
    #     for h, length in zip(handlePropertiesL, lengthL):
    #         n, l, w = h
    #         obj_real_world = l
    #         d = distLengthPredict(stereoCams, length, obj_real_world)
    #         distanceL.append(np.array([n, d]))

    # if len(digitR) is not 0:
    #     handlePropertiesR = matchHandle(digitR)

    #     for h, length in zip(handlePropertiesR, lengthR):
    #         n, l, w = h
    #         obj_real_world = l
    #         d = distLengthPredict(stereoCams, length, obj_real_world)
    #         distanceR.append(np.array([n, d]))

    # computeAvgDistance(distanceL, distanceR)
    # print '############################################'
    # break
