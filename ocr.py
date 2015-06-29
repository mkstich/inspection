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
from find_rectangles import *
from stereo_calibrate import *

# Import pytesseract library
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract


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
    val1 = [float(vals[-6]), float(vals[-5]), float(vals[-4]),
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
    r_imgL, r_imgR = rectifyImage(
        imgL_unrect, imgR_unrect, imsize, stereoCams, folder, right, left)

    # Create all image names needed
    h1 = imageName(folder, 'handlesL.png')
    h2 = imageName(folder, 'handlesR.png')
    hL_name = imageName(folder, 'filtered_rect_L' + loc + '.png')
    hR_name = imageName(folder, 'filtered_rect_R' + loc + '.png')

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

    if len(rect) == 3:
        x, y = rect[0]
        w, h = rect[1]

    else:
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

    f_img = 35.  # Actual Focal Length
    m = f1 / f_img  # Scaling ratio
    # Length of image on the camera sensor
    sensor = length / m

    # Estimate the distance using the pinhole projection model
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
        # Keep only the numbers and separate each number with a ' '
        digit_str = re.sub('[^0-9]', ' ', digit_str)

        if len(digit_str) > 0 and digit_str[0] == ' ':
            digit_str = digit_str[1:].split(' ')
        else:
            digit_str = digit_str.split(' ')

        # Keep only 4 - numbered handle digits from the image
        for j in digit_str:
            if j.isdigit() is True:
                if len(str(j)) == 4:
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
    q = h / 4
    crop = im[10 + q: 3. * q, 0.2 * w:w * 0.8]

    new_rect = compute_threshold(crop, img_name, False)
    
    if len(new_rect) > 0:
        if len(new_rect) > 1:
            x, y, w, h = new_rect[-1]
            if h > w:
                x, y, w, h = new_rect[0]
        else:
            x, y, w, h = new_rect[0]

        crop2 = crop[y:y + h, x + 0.1 * w: x + 0.9 * w]

        cv2.imwrite(img_name, crop2)

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
    output_image = cv.CreateImage((width, height), image.depth,
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


def lastParseAttempt(img, cv_img, img_name, rect):
    '''If the background of the handle tag is too dark 
    for Tesseract, reverse the tag's colors. This will likely make
    the background white, and the handle number black, allowing Tesseract to 
    have a higher probablity of successful detection

    return = the calculated digit and corresponding handle length
    '''
    h, w, _ = img.shape
    q = h / 4

    rect = compute_threshold(img, img_name, True)

    for i, r in zip(range(len(rect)), rect):
        centre = r[0]
        x, y = centre
        w, h = np.int0(r[1])
        theta = np.radians(r[-1])

        # Crop image and save image name
        cropped = subimage(cv_img, centre, theta, w, h)
        cL_name = img_name
        cv.SaveImage(cL_name, cropped)
            
        # check if the image needs to be rotated
        # if so, read in a cv2 image
        if h > w:
            rot_img = cv2.imread(cL_name)
            cv2.imwrite(cL_name, ndimage.rotate(rot_img, 90))

    digit, length = TesseractOCR([img_name], rect) #[(x, y, w, h)])

    if len(digit) < 1 or (len(digit) > 0 and len(str(digit[0])) < 4):
        img = cv2.imread(img_name)
        cv2.imwrite(img_name, 255 - img)

        digit, length = TesseractOCR([img_name], rect) #[(x, y, w, h)])

    if len(digit) > 0 and len(str(digit[0])) < 4:
        digit = []

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
               [1212, 8.64 * convert, w],
               [1139, 8.64 * convert, w],
               [1148, 22.7 * convert, w],
               [1154, 22.7 * convert, w],
               [1159, 22.7 * convert, w],
               [1165, 8.64 * convert, w],
               [1153, 22.7 * convert, w],
               [1138, 8.64 * convert, w],
               [1147, 22.7 * convert, w],
               ]

    matches = []
    for d in digit:
        # if a match is found, save the index and handle values
        if len([i for i, x in enumerate(handles) if x[0] == d]) > 0:
            idx = [i for i, x in enumerate(handles) if x[0] == d][0]
            matches.append(np.array(handles[idx]))

    return matches

def DigitSearch(cnt, rect):
    digit, length = TesseractOCR(cnt, rect)

    # If no digit was found, crop image to approx. tag size
    if len(digit) is 0:
        orig_img = cv2.imread(cnt[0])
        cv_img = cv.LoadImage(cnt[0])

        cnt_new = []
        for i in cnt:
            c = smallerRectImage(i)
            cnt_new.append(c)

        digit, length = TesseractOCR(cnt_new, rect)

        # If Tesseract can't recognize the number, reverse its color
        if len(digit) is 0 or len(str(digit[0])) < 4:
            digit, length = lastParseAttempt(orig_img, cv_img, \
                cnt_new[0], rect)

    return digit, length

def HandleMatchDist(digit, length_vec):
    distance = []
    if len(digit) is not 0:
        # Match the handle to info in the database
        handleProperties = matchHandle(digit)

        # Predict the distance to the handle
        for h, length in zip(handleProperties, length_vec):
            n, l, w = h # n = number, l = length, w = width 
            obj_real_world = l
            d = distLengthPredict(stereoCams, length, obj_real_world)
            distance.append(np.array([n, d]))

    return distance

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
                if len(idx) > 0:
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
                    if distanceR[i] not in enumerate(distanceL):
                        print 'Handle: ', distanceR[i][0], ' Distance: ', \
                        distanceR[i][1]


imsize, stereoCams = stereoCalibration()

imgsL = sorted(glob.glob("distance/input/left/left_*.jpeg"))
imgsR = sorted(glob.glob("distance/input/right/right_*.jpeg"))
# handle dimensions (number), (length), (width)
# unit = inches

folder = "distance/sim_images"
rectangleL, rectangleR = [], []

# for L, R, rectL, rectR in zip(imgsL, imgsR, rectangleL, rectangleR):
for L, R in zip(imgsL, imgsR):

    x, y, z, p, yw, r = splitString(L)
    loc = str(x) + '_' + str(y) + '_' + str(z)
    print loc

    rectL, rectR, loc = findDistance(L, R, imsize, stereoCams, folder)
    rL_name = imageName(folder, 'rectify_imgL' + loc + '.png')
    rR_name = imageName(folder, 'rectify_imgR' + loc + '.png')

    cL_name = imageName(folder, 'cropL_' + loc)
    cR_name = imageName(folder, 'cropR_' + loc)

    print 'left images'
    cropL_name = cropImage(rL_name, rectL, cL_name, 'L')
    cntL = prepareImage(cropL_name)
    digitL, lengthL = DigitSearch(cntL, rectL)

    print 'right images'
    cropR_name = cropImage(rR_name, rectR, cR_name, 'R')
    cntR = prepareImage(cropR_name)
    digitR, lengthR = DigitSearch(cntR, rectR)

    rectangleR.append(rectR)
    rectangleL.append(rectL)
    plt.close('all')

    distanceL = HandleMatchDist(digitL, lengthL)
    distanceR = HandleMatchDist(digitR, lengthR)

    computeAvgDistance(distanceL, distanceR)
    print '############################################'

# Filter the Predicted Distances #############################
# Compute Data Statistics ####################################
# Predicted handle distances
data = [[0, 18.67],
        [1, 20.06],
        [2, 21.58],
        [3, 21.83],
        [4, 18.92],
        [5, 22.56],
        [6, 23.64],
        [7, 21.42],
        [8, 22.58],
        [9, 18.27],
        [10, 19.08],
        [11, 30.90],
        [12, None],
        [13, 17.87],
        [14, 27.34],
        [14, 21.26],
        [15, None],
        [16, 21.48],
        [17, 17.61],
        [18, 18.57],
        [18, 20.19],
        [19, 18.65]
        ]
# # Actual handle distances
actual = [[0, 27],
        [1, 27],
        [2, 27],
        [3, 27],
        [4, 27],
        [5, 27],
        [6, 26],
        [7, 26],
        [8, 26],
        [9, 26],
        [10, 26],
        [11, 26],
        [12, 26],
        [13, 25],
        [14, 25],
        [14, 25],
        [15, 25],
        [16, 25],
        [17, 25],
        [18, 25],
        [18, 25],
        [19, 25]
        ]

d = pd.DataFrame(data)
a = pd.DataFrame(actual)
handle_depth = 2.88 # value from NASA specs
d[3] = abs(a[1]-handle_depth-d[1])

# compute error stats
error = pd.DataFrame()
error[0] = d[3]
error[0].describe()

# Create Kalman Filter
Q, R = 1., 10.
kf = KalmanFilter(Q, R)
distance = []

for i, num in zip(d[1], d[0]):
    # Update value using KF if it's finite
    if np.isfinite(i):
        kf.update(i)
        new_dist = kf.predict()
        distance.append([num, new_dist])
    else:
        distance.append([num, None])

dist = pd.DataFrame(distance)
dist[2] = abs(a[1] - handle_depth - dist[1])
error[1] = dist[2]
error[1].describe()

# Plot Error results
plt.plot(dist[0]+1, dist[2], 'ro', label = 'filtered')
plt.plot(d[0] + 1, d[3], 'bo', label = 'unfiltered')
plt.xticks(np.arange(min(d[0]), max(d[0])+2, 1.0))
plt.legend(loc = 1)
plt.ylabel('Relative Error Magnitude (inches)')
plt.xlabel('Simulation Trial Number')
plt.title('Relative Distance Error Magnitude')
plt.savefig('distance/rel_dist_error_filter.pdf')
plt.show()