import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import glob as glob
import math
import numpy.linalg as la
import simplejson
import re
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from stereo_calibrate import *

class KalmanFilter(object):
    def __init__(self, process_var, measurement_var):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.posteri_est = None
        self.posteri_error_est = 1.0

    def update(self, measurement):
        if self.posteri_est is None:
            self.posteri_est = measurement

        priori_est = self.posteri_est
        priori_error_est = self.posteri_error_est + self.process_var

        gain = priori_error_est / (priori_error_est + self.measurement_var)
        self.posteri_est = priori_est + gain * (measurement - priori_est)
        self.posteri_error_est = (1 - gain) * priori_error_est

    def predict(self):
        return self.posteri_est

class CubeSat(object):
    x, y, z = 0., 0., 0.
    p, y, r = 0., 0., 0.
    e = 0.

    def __init__(self, x, y, z, p, yw, r, e, a):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = p
        self.yaw = yw
        self.roll = r
        self.error = e
        self.abs_err = a

def drawMatches(img1, kp1, img2, kp2, matches, name):
    """
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in (matches):

    # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(
            out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    # Show the image
    # cv2.imshow('Matched Features', out)
    cv2.imwrite(name, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def featureMatching(img1, img2, name):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()  # (checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # calculate the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # plot the first 100 matches
    # drawMatches(img1, kp1, img2, kp2, good[:200], name)

    return M


def calcR(H, M):
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    # calculate the scaling factor, lda
    lda = 1. / la.norm(np.dot(la.inv(M), h1))

    # compute the components of the rotation matrix
    r1 = np.array([np.dot(lda, np.dot(np.linalg.inv(M), h1))])
    r2 = np.array([np.dot(lda, np.dot(np.linalg.inv(M), h2))])
    r3 = np.cross(r1, r2)

    R = np.hstack([r1.T, r2.T, r3.T])

    # translation vector
    t = np.dot(lda, np.dot(np.linalg.inv(M), h3))

    return R


def detAttitude(HL, ML, HR, MR, ar, ap, ay, filtered):
    ''' Gary Bradski's OpenCV Textbook method 
    http://www.cs.haifa.ac.il/~dkeren/ip/OReilly-LearningOpenCV.pdf
    Used formulas from Ch. 11 pg. 389 'What's under the hood?'

    H = Homography matrix
    M = camera matrix
    '''

    # Create CubeSat object
    satellite = CubeSat(0., 0., 0., 0., 0., 0., 0., 0.)
    # Calculate Rotation matrices
    R_L = calcR(HL, ML)
    R_R = calcR(HR, MR)

    # Compute angles
    r, p, y = calcAngles(R_R, R_L, ar, ap, ay)
    
    if filtered == True:
        kf.update(np.array([r, p, y]))
        r, p, y = kf.predict()

    satellite.roll = r
    satellite.pitch = p
    satellite.yaw = y

    # note: to estimate location you will need to SUBTRACT
    # r, p, y from the current angle to get the proper sign
    satellite.error, satellite.abs_err = error(ar, r, ap, p, ay, y)

    return satellite


def error(ar, r, ap, p, ay, y):
    '''Compute total error sum and average error'''
    e1 = np.abs(ar - r)
    e2 = np.abs(ap - p)
    e3 = np.abs(ay - y)
    sum_err = e1 + e2 + e3

    return np.mean([e1, e2, e3]), sum_err

def PYR(R):
    ''' Pitch Yaw Roll Rotation Matrix Decomposition'''
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat

    if r13 > 0.998:
        yaw = np.round(math.degrees(math.pi / 2.), 4)
        roll = 0.
        pitch = np.round(math.degrees(math.atan2(r31, r21)), 4)
    
    elif r13 < -0.998:
        yaw = np.round(math.degrees(-math.pi / 2.), 4)
        roll = 0.
        pitch = np.round(math.degrees(math.atan2(r31, r21)), 4)

    else:
        pitch = np.round(math.degrees(math.atan2(-r23, r33)), 4)
        roll = np.round(math.degrees(math.atan2(r12, r11)), 4)
        yaw = np.round(math.degrees(math.asin(-r13)), 4)

    return pitch, roll, yaw

def getMin(left, right):
    res = 0.
    if np.abs(left) < np.abs(right):
        res = left
    else:
        res = right
    
    return res

def meanAngle(rr, rl, pr, pl, yr, yl):
    '''Compute mean angle approximation'''
    if np.abs(rr - rl) >= 1.5 * min(np.abs([rr, rl])):
        r1 = getMin(rl, rr)
    else:
        r1 = np.round(np.mean([rr, rl]), 4)

    if np.abs(pr - pl) >= 1.5 * min(np.abs([pr, pl])):
        p1 = getMin(pl, pr)
    else:
        p1 = np.round(np.mean([pr, pl]), 4)
    
    if np.abs(yr - yl) >= 1.5 * min(np.abs([yr, yl])):
        y1 = getMin(yl, yr)
    else:
        y1 = np.round(np.mean([yr, yl]), 4)

    return r1, p1, y1


def calcAngles(R_R, R_L, ar, ap, ay):
    '''
    heading = theta = yaw (about y)
    bank = psi = pitch (about x)
    attitude = phi = roll (about z)

    EDGE rotation matrixes will be in the order of operations
    of pitch, yaw, roll
    '''

    '''Order of Euler Angles: pitch, yaw, roll'''
    results = []

    # print 'Pitch Yaw Roll'
    pr, rr, yr = PYR(R_R)
    pl, rl, yl = PYR(R_L)
    r2, p2, y2 = meanAngle(rr, rl, pr, pl, yr, yl)
    print 'r = ', pr, rr, yr, ' l = ', pl, rl, yl, 'a =', p2, r2, y2

    return r2, p2, y2

def splitString(name):
    vals = name.split('_')
    y = vals[-1].split('.')[0]
    val1 = [float(vals[-3]), float(vals[-2]), float(y)]

    return val1

def natural_sort(l): 
    '''Natural sort to have numbers go in human order'''
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def findBest():
    imgsL = glob.glob('attitude/left_*.jpeg')
    imgsR = glob.glob('attitude/right_*.jpeg')

    # Make sure the stereo images are in the same order
    imgsL = natural_sort(imgsL)
    imgsR = natural_sort(imgsR)

    M_camL = np.array([[1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
                       [0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    M_camR = np.array([[1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
                       [0.00000000e+00,   2.09561499e+03,   5.37271880e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    output = pd.DataFrame(columns = ('r', 'p', 'y', 'ar', 'ap', \
        'ay', 'e', 'ae'))

    for img1L_name, img1R_name in zip(imgsL, imgsR):

        img1L = cv2.imread(img1L_name, 0)
        img1R = cv2.imread(img1R_name, 0)

        val1 = splitString(img1L_name)

        for img2L_name, img2R_name in zip(imgsL, imgsR):
            img2L = cv2.imread(img2L_name, 0)
            img2R = cv2.imread(img2R_name, 0)
            val2 = splitString(img2L_name)
            ar, ap, ay = np.array(val1) - np.array(val2)

            nameL = 'attitudeResults/filtered/matchesL' + \
                str(val1) + str(val2) + '.png'
            nameR = 'attitudeResults/filtered/matchesR' + \
                str(val1) + str(val2) + '.png'

            HR = featureMatching(img1R, img2R, nameR)
            HL = featureMatching(img1L, img2L, nameL)
            satellite = detAttitude(HL, M_camL, HR, M_camR, ar, ap, ay)

            trial = pd.Series()
            trial['r'] = satellite.roll
            trial['p'] = satellite.pitch
            trial['y'] = satellite.yaw
            trial['ar'] = ar
            trial['ap'] = ap
            trial['ay'] = ay
            trial['e'] = satellite.error
            trial['ae'] = satellite.abs_err
            # print str(val1) + str(val2)

            output = output.append(trial, ignore_index = True)
        break

    output.to_csv('attitudeResults/attitude_data_filtered' + str(Q) + '_' + \
        str(R) + '.csv')

    return output

def findBestTimeSim(imgsL, imgsR, filtered, name, stereoCams):
    '''Calculate RPY angles for a full time sim
    moving 1 degree in P, Y, and/or R directions'''
    # Make sure the stereo images are in the same order
    imgsL = natural_sort(imgsL)
    imgsR = natural_sort(imgsR)

    half = len(imgsL) / 2 + 1

    # Put imgs in order from largest to smallest pitch angle
    B = imgsL[:half]
    B.reverse()
    imgsL[:half] = B

    B = imgsR[:half]
    B.reverse()
    imgsR[:half] = B

    # M_camL = np.array([[1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
    #                    [0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
    #                    [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # M_camR = np.array([[1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
    #                    [0.00000000e+00,   2.09561499e+03,   5.37271880e+02],
    #                    [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    M_camL = stereoCams.M1
    M_camR = stereoCams.M2
    # M_camR = np.array([[1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
    #                    [0.00000000e+00,   1.89292737e+03,   5.37271880e+02],
    #                    [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    output = pd.DataFrame(columns = ('r', 'p', 'y', 'ar', 'ap', \
        'ay', 'e', 'ae','Q', 'R'))
    time = len(imgsL) - 1
    # time = 5

    for dt in range(time):

        img1L = cv2.imread(imgsL[dt], 0)
        img1R = cv2.imread(imgsR[dt], 0)

        val1 = splitString(imgsL[dt])

        img2L = cv2.imread(imgsL[dt + 1], 0)
        img2R = cv2.imread(imgsR[dt + 1], 0)
        val2 = splitString(imgsL[dt + 1])
        ar, ap, ay = np.array(val1) - np.array(val2)

        nameL = 'attitudeTimeResults/matchesL' + \
            str(val1) + str(val2) + '.png'
        nameR = 'attitudeTimeResults/matchesR' + \
            str(val1) + str(val2) + '.png'

        HR = featureMatching(img1R, img2R, nameR)
        HL = featureMatching(img1L, img2L, nameL)
        satellite = detAttitude(HL, M_camL, HR, M_camR, ar, ap, ay, filtered)

        trial = pd.Series()
        trial['r'] = satellite.roll
        trial['p'] = satellite.pitch
        trial['y'] = satellite.yaw
        trial['ar'] = ar
        trial['ap'] = ap
        trial['ay'] = ay
        if filtered == True:
            trial['Q'] = Q
            trial['R'] = R
        trial['e'] = satellite.error
        trial['ae'] = satellite.abs_err
        # print str(val1) + str(val2)

        output = output.append(trial, ignore_index = True)

    # print output
    if filtered == True:
        output.to_csv('attitudeResults/attitude_data_time_filter_' + name + \
            str(np.round(Q, 3)) + '_' + str(np.round(R, 3)) + '.csv')
    else:
        output.to_csv('attitudeResults/attitude_data_time_unfiltered_' + \
            name + '.csv')

    return output


# Qest = np.linspace(0.001, 0.01, 10)#np.linspace(0.1, 0.98, 10)
# Rest = np.linspace(.1, 1.0, 10)
# Qest = np.linspace(0.1, 10, 10)
# Q, R = 1., 0.001. #15. #0.032, 1.
# kf = KalmanFilter(Q, R)

imgsL = glob.glob('attitude/pitch_yaw_roll/left_*.jpeg')
imgsR = glob.glob('attitude/pitch_yaw_roll/right_*.jpeg')

# # # for q in Qest:
# #     # Kalman Filter
# #     # Q = process var., R = measurement var

output = findBestTimeSim(imgsL, imgsR, False, "PYR", stereoCams)
print 'total abs error = ', output['ae'].sum()


# # Test All Combinations with Filter
# Q, R = 0.001, 1. #15. #0.032, 1.

# # conduct time sim from pitch = 10. to -10.
# imgsL = glob.glob('attitude/left_*.jpeg')
# imgsR = glob.glob('attitude/right_*.jpeg')

# kf = KalmanFilter(Q, R)
# output = findBestTimeSim(imgsL, imgsR, False, "P")

# output['e'].plot()
# plt.title("Filtered Error (PYR) \n Q = " + str(Q) + ", R = " + str(R))
# plt.savefig("attitudeResults/filteredPYR_" + str(Q) + "_" + \
#     str(R) + "_error.png")
# plt.show()

# output[['p', 'y', 'r']].plot()
# plt.title("Filtered Pitch, Yaw, Roll \n Q = " + str(Q) + \
#     ", R = " + str(R))
# plt.savefig("attitudeResults/filteredPYR_" + str(Q) + "_" + str(R) + \
#     "_pitch_yaw_roll.png")
# plt.show()

# output[['p','ap', 'y', 'ay', 'r', 'ar']].plot()
# plt.title("All Filtered Output (PYR) \n Q = " + str(Q) + ", R = " + str(R))
# plt.savefig("attitudeResults/filteredPYR_" + str(Q) + "_" + str(R) + \
#     "_output.png")
# plt.show()

# output['e'].plot()
# plt.title("Unfiltered Error (PYR)")
# plt.savefig("attitudeResults/unfilteredPYR_error.png")
# plt.legend(loc = 1)
# plt.show()

# output[['p', 'y', 'r']].plot()
# plt.title("Unfiltered Pitch, Yaw, Roll")
# plt.savefig("attitudeResults/unfilteredPYR_pitch_yaw_roll.png")
# plt.legend(loc = 1)
# plt.show()

# output[['p','ap', 'y', 'ay', 'r', 'ar']].plot()
# plt.title("All Unfiltered Output (PYR)")
# plt.legend(loc = 1)
# plt.savefig("attitudeResults/unfilteredPYR_output.png")
# plt.show()


def RelativeAttitude(data, name, filtered, show_plot):
    output = pd.DataFrame.from_csv(data)

    actual = CubeSat(0., 0., 0., 0., 0., 0., 0., 0.)
    estimated = CubeSat(0., 0., 0., 0., 0., 0., 0., 0.)
    new_output = pd.DataFrame(columns = ('r', 'p', 'y', 'ar', 'ap', \
        'ay', 'e', 'ae','Q', 'R'))
    
    er, ep, ey = 0., 0., 0.
    err, aerr = 0., 0.
    for i in range(len(output)):
        actual.pitch -= output['ap'].iloc[i]
        actual.roll -= output['ar'].iloc[i]
        actual.yaw -= output['ay'].iloc[i]

        p, y, r = output['p'].iloc[i], output['y'].iloc[i], \
                output['r'].iloc[i]
        p_sgn = np.sign(p)
        y_sgn = np.sign(y)
        r_sgn = np.sign(r)

        if filtered == True:
            ep += p
            ey += y
            er += r
            # print ep, ey, er

            kf.update(np.array([ep, ey, er]))
            p, y, r = kf.predict()

            if np.sign(p) is not p_sgn:
                p *= p_sgn
            if np.sign(y) is not y_sgn:
                y *= y_sgn
            if np.sign(r) is not r_sgn:
                r *= r_sgn
        estimated.pitch -= p
        estimated.yaw -= y
        estimated.roll -= r

        e, ae = error(actual.roll, estimated.roll, actual.pitch, \
            estimated.pitch, actual.yaw, estimated.yaw)

        # err += output['e'].iloc[i]
        # aerr += output['ae'].iloc[i]

        trial = pd.Series()
        trial['r'] = estimated.roll
        trial['p'] = estimated.pitch
        trial['y'] = estimated.yaw
        trial['ar'] = actual.roll
        trial['ap'] = actual.pitch
        trial['ay'] = actual.yaw
        trial['ae'] = ae
        trial['e'] = e
        trial['Q'] = Q
        trial['R'] = R
        new_output = new_output.append(trial, ignore_index = True)

    new_output[['p','ap', 'y', 'ay', 'r', 'ar', 'ae']].plot()

    if show_plot == True:
        if filtered == True:
            plt.title("All Filtered Output (" + name + ")")
            plt.axis([0, 20, -30, 30])
            plt.savefig("attitudeResults/filtered_noR_" + name + "_output.png")
        else:
            plt.title("All Unfiltered Output (" + name + ")")
            plt.axis([0, 20, -30, 30])
            plt.savefig("attitudeResults/unfiltered" + name + "_output.png")     
        plt.show()

    else:
        return new_output
    plt.close("all")
    return new_output

# Qest = np.linspace(10, 100., 10)
# Rest = np.linspace(1, 10, 10)

att = ['P', 'Y', 'R', 'PY', 'PYR']
# for r in Rest:
# # for q in Qest:
for a in att:
    Q, R = 0.0001, 1e10
    # a = 'PYR_new'
    kf = KalmanFilter(Q, R)
    data = "attitudeResults/" + a + "_unfiltered.csv"
    name = a
    output = RelativeAttitude(data, name, True, True)
    d = pd.DataFrame.from_csv(data)
    print output['ae'].iloc[-1]
    # print d.sum()
    plt.close('all')