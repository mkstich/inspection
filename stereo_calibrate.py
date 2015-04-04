import numpy as np
import cv2
import glob as glob
import numpy.linalg as la


def locateStereoPoints(imagesL, imagesR):
    '''Get the object points (this will be the same for both images) 
    and the image points (the image points will be different per camera)
    to use for stereo calibration'''

    square_size = 1.0
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points_L, img_points_R = [], []
    h, w = 0, 0

    for (lname, rname) in zip(imagesL, imagesR):
        img1 = cv2.imread(lname) #img1
        img2 = cv2.imread(rname) #img2

        imgL = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if imgL is None:
            print "Failed to load", lname
            continue

        if imgR is None:
            print "Failed to load", rname
            continue

        h, w = imgL.shape[:2]
        foundL, cornersL = cv2.findChessboardCorners(imgL, pattern_size)
        foundR, cornersR = cv2.findChessboardCorners(imgR, pattern_size)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

        if foundL:
            cv2.cornerSubPix(imgL, cornersL, (5, 5), (-1, -1), term)
            img_points_L.append(cornersL.reshape(-1, 2))

        if foundR:
            cv2.cornerSubPix(imgR, cornersR, (5, 5), (-1, -1), term)
            img_points_R.append(cornersR.reshape(-1, 2))

        obj_points.append(pattern_points)

        if not foundL:
            print 'left chessboard not found'
            continue
        if not foundR:
            print 'right chessboard not found'
            continue

    return obj_points, img_points_L, img_points_R, (w, h)


def main():
    imagesL = glob.glob('calibration_samples/l*.jpeg')
    imagesR = glob.glob('calibration_samples/r*.jpeg')

    # Perform Stereo Calibration
    obj_pts, ptsL, ptsR, imsize = locateStereoPoints(imagesL, imagesR)
    retval, cam1, dist1, cam2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, ptsL, ptsR, imsize)
    dist1 = dist1.ravel()
    dist2 = dist2.ravel()

    cam1 = np.array([[1.53216158e+03, 0., 8.96615289e+02],
                    [0., 1.43951272e+03, 5.44758043e+02],
                    [0., 0., 1.]])
    # dist1 = np.array([0.03702081, 0.85215103, 0.00343193, -0.02808688, -1.08401299])

    cam2 = np.array([[1.4409744e+03, 0., 7.74076208e+02],
                    [0., 1.42435001e+03, 5.27489490e+02],
                    [0., 0., 1.]])
    # dist2 = np.array([-0.23991173, 1.3750377, -0.03521028, -0.06381475, -3.57839555])
    # Perform Stereo Rectification
    (R1, R2, P1, P2, Q, roi1, roi2) = cv2.stereoRectify(
        cam1, dist1, cam2, dist2, imsize, R, T)

    # Undistort
    map1x, map1y = cv2.initUndistortRectifyMap(cam1, dist1, R1, P1, imsize, 5)
    map2x, map2y = cv2.initUndistortRectifyMap(cam2, dist2, R2, P2, imsize, 5)

    # Remap the Rectified Images
    # original images
    imgL = cv2.imread('calibration_samples/l1.jpeg')
    imgR = cv2.imread('calibration_samples/r1.jpeg')

    # remapped images
    r_imgL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    # x, y, w, h = roi1
    # r_imgL = r_imgL[ y: y + h, x : x + w ]

    r_imgR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    # x, y, w, h = roi2
    # r_imgR = r_imgR[ y: y + h, x : x + w ]

    # Save the Rectified Images
    cv2.imwrite("rectify_imgL.png", r_imgL)
    cv2.imwrite("rectify_imgR.png", r_imgR)
