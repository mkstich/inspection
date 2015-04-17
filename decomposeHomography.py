import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import glob as glob
import math
import numpy.linalg as la
import simplejson
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D


class CubeSat(object):
    x, y, z = 0., 0., 0.
    p, y, r = 0., 0., 0.
    e = 0.

    def __init__(self, x, y, z, p, yw, r, e):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = p
        self.yaw = yw
        self.roll = r
        self.error = e


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
        # matchesMask = mask.ravel().tolist()
        # h, w = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)

    # plot the first 100 matches
    # drawMatches(img1, kp1, img2, kp2, good[:100], name)

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


def detAttitude(HL, ML, HR, MR, ar, ap, ay):
    ''' Gary Bradski's OpenCV Textbook method 
    http://www.cs.haifa.ac.il/~dkeren/ip/OReilly-LearningOpenCV.pdf
    Used formulas from Ch. 11 pg. 389 'What's under the hood?'

    H = Homography matrix
    M = camera matrix
    '''

    # Create CubeSat object
    satellite = CubeSat(0., 0., 0., 0., 0., 0., 0.)
    # Calculate Rotation matrices
    R_L = calcR(HL, ML)
    R_R = calcR(HR, MR)

    # Compute angles
    results = calcAngles(R_R, R_L, ar, ap, ay)
    satellite.roll = results[0][0]
    satellite.pitch = results[0][1]
    satellite.yaw = results[0][2]
    satellite.error = results[0][3]

    return satellite


def error(ar, r, ap, p, ay, y):
    e1 = np.abs(ar - r)
    e2 = np.abs(ap - p)
    e3 = np.abs(ay - y)

    return np.mean([e1, e2, e3])

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
        roll = np.round(math.degrees(math.atan2(-r12, r11)), 4)
        yaw = np.round(math.degrees(math.asin(r13)), 4)

    return pitch, roll, yaw

# def RYP(R):
#     '''Rotation Matrix Decomposition for rotation angles applied
#     in the order of Roll - Yaw - Pitch '''
#     r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat

#     # if singularity at north pole
#     if -r31 > 0.998:
#         yaw = np.round(math.degrees(math.pi / 2.), 4)
#         roll = np.round(math.degrees(math.atan2(r13, -r23)), 4)
#         pitch = 0.

#     # if singularity at south pole
#     elif -r31 < -0.998:
#         yaw = np.round(math.degrees(-math.pi / 2.), 4)
#         roll = np.round(math.degrees(math.atan2(r13, -r23)), 4)
#         pitch = 0.

#     else:
#         pitch = np.round(math.degrees(math.atan2(r32, r33)), 4)
#         roll = np.round(math.degrees(math.atan2(r21, r11)), 4)
#         yaw = np.round(math.degrees(math.asin(-r31)), 4)

#     return pitch, roll, yaw


def meanAngle(rr, rl, pr, pl, yr, yl):
    r1 = np.round(np.mean([rr, rl]), 4)
    p1 = np.round(np.mean([pr, pl]), 4)
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
    e2 = error(ar, r2, ap, p2, ay, y2)
    results.append([r2, p2, y2, e2, 'PYR'])

    return results


def splitString(name):
    vals = name.split('_')
    y = vals[-1].split('.')[0]
    val1 = [float(vals[1]), float(vals[2]), float(y)]

    return val1


def findBest():
    imgsL = glob.glob('attitude/left_*.jpeg')
    imgsR = glob.glob('attitude/right_*.jpeg')

    # Make sure the stereo images are in the same order
    imgsL = sorted(imgsL)
    imgsR = sorted(imgsR)
    # values = np.array([[-146., 160., -18.],
    #                    [-146., 162., -18.],
    #                    [-143., 162., -18.],
    #                    [-143., 162., -19.],
    #                    [-143., 162., -19.],
    #                    [-143., 162., -19.]])
    M_camL = np.array([[1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
                       [0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    M_camR = np.array([[1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
                       [0.00000000e+00,   2.09561499e+03,   5.37271880e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    i = 1

    output = pd.Series()
    trial = pd.Series()

    for img1L_name, img1R_name in zip(imgsL, imgsR):

        img1L = cv2.imread(img1L_name, 0)
        img1R = cv2.imread(img1R_name, 0)
        j = 1

        val1 = splitString(img1L_name)

        for img2L_name, img2R_name in zip(imgsL, imgsR):
            img2L = cv2.imread(img2L_name, 0)
            img2R = cv2.imread(img2R_name, 0)
            val2 = splitString(img2L_name)
            ar, ap, ay = np.array(val1) - np.array(val2)

            nameL = 'attitudeResults/matchesL' + \
                str(val1) + str(val2) + '.png'
            nameR = 'attitudeResults/matchesR' + \
                str(val1) + str(val2) + '.png'

            HR = featureMatching(img1R, img2R, nameR)
            HL = featureMatching(img1L, img2L, nameL)
            satellite = detAttitude(HL, M_camL, HR, M_camR, ar, ap, ay)
            trial['r'] = satellite.roll
            trial['p'] = satellite.pitch
            trial['y'] = satellite.yaw
            trial['ar'] = ar
            trial['ap'] = ap
            trial['ay'] = ay
            trial['e'] = satellite.error
            print str(val1) + str(val2)

            output = pd.concat((trial, output), axis=1)

            break
            j += 1
        i += 1

    output.to_csv('attitude_data.csv')

    # r_arr = np.array(results)
    # names = r_arr[:,-1]
    # counts = Counter(names.tolist())
    # print counts

    # plot bar graph of Euler method used
    # df = pandas.DataFrame.from_dict(counts, orient = 'index')
    # df.plot(kind = 'bar')
    # plt.savefig('rotation_error.png')
    # plt.show()

    # write results to text file
    # f = open('attitudeTracking/small_matches/RYP.txt', 'w')
    # simplejson.dump(results, f)
    # f.close()

    return results


def main():
    M_camL = np.array([[1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
                       [0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    M_camR = np.array([[1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
                       [0.00000000e+00,   2.09561499e+03,   5.37271880e+02],
                       [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    img1L = cv2.imread('attitudeTracking/small/l1.jpeg', 0)
    img2L = cv2.imread('attitudeTracking/small/l2.jpeg', 0)
    HL = featureMatching(img1L, img2L, 'matches.png')

    img1R = cv2.imread('attitudeTracking/small/r1.jpeg', 0)  # queryImage
    img2R = cv2.imread('attitudeTracking/small/r2.jpeg', 0)  # trainImage
    HR = featureMatching(img1R, img2R, 'matchesR.jpeg')

    ar = 0.
    ap = -2.
    ay = 0.
    detAttitude(HL, M_camL, HR, M_camR, ar, ap, ay)

    img1L = cv2.imread(
        'attitudeTracking/multipleAngles/l1.jpeg', 0)  # queryImage
    img2L = cv2.imread(
        'attitudeTracking/multipleAngles/l2.jpeg', 0)  # trainImage

    # compute the feature matching, calculate the homography matrix
    HL = featureMatching(img1L, img2L, 'matchesL.jpeg')

    # M = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
    #   [  0.00000000e+00,   1.99299316e+02,   5.52500000e+02],
    #    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    # K = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
    #        [  0.00000000e+00,   1.70664233e+03,   5.52500000e+02],
    #        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # OpenCV Homography decomposition method
    print 'OpenCv Homography Decomposition'
    print 'Left'
    rl, pl, yl = detAttitude(HL, M_camL)
    print 'Right'
    rr, pr, yr = detAttitude(HR, M_camR)

    print 'Angle averages:'
    print 'roll = ', np.round(np.mean([rr, rl]), 4), \
        ' pitch = ', np.round(np.mean([pr, pl]), 4), \
        ' yaw = ', np.round(np.mean([yr, yl]), 4)
