import numpy as np
import cv2
import glob as glob
import numpy.linalg as la

class StereoCam:
    def __init__(self): # cam1, dist1, cam2, dist2, R1, P1, R2, P2, Q, R, T, K1, K2):
        self.cam1 = np.identity(3)
        self.dist1 = np.array([0., 0., 0., 0., 0.])
        self.cam2 = np.identity(3)
        self.dist2 = np.array([0., 0., 0., 0., 0.])
        self.R1 = np.zeros((3, 3))
        self.P1 = np.zeros((3, 4))
        self.R2 = np.zeros((3, 3))
        self.P2 = np.zeros((3, 4))
        self.Q = np.zeros((4, 4))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))
        self.K1 = np.zeros((3, 3))
        self.K2 = np.zeros((3, 3))
        self.E = np.zeros((3, 3))
        self.F = np.zeros((3, 3))

    def printCam(self):
        print 'cam1 = ', self.cam1
        print 'dist 1 = ', self.dist1
        print 'cam2 = ', self.cam2
        print 'dist2 = ', self.dist2
        print 'R1 = ', self.R1
        print 'R2 = ', self.R2
        print 'P1 = ', self.P1
        print 'P2 = ', self.P2
        print 'Q = ', self.Q
        print 'R = ', self.R
        print 'T = ', self.T
        print 'K1 = ', self.K1
        print'K2 = ', self.K2
        print 'F = ', self.F
        print 'E = ', self.E 

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

def calibrateImages(obj_pts, ptsL, ptsR, imsize):

    # Perform Stereo Calibration
    retval, cam1, dist1, cam2, dist2, R, T, E, F = \
        cv2.stereoCalibrate(obj_pts, ptsL, ptsR, imsize)#, cam1, None, cam2, None)
    dist1 = dist1.ravel()
    dist2 = dist2.ravel()

    cameraObj = StereoCam()
    cameraObj.cam1, cameraObj.dist1 = cam1, dist1
    cameraObj.cam2, cameraObj.dist2 = cam2, dist2

    cameraObj.R, cameraObj.T, cameraObj.E, cameraObj.F = R, T, E, F

    # cam1 = np.array([[1.53216158e+03, 0., 8.96615289e+02],
    #                 [0., 1.43951272e+03, 5.44758043e+02],
    #                 [0., 0., 1.]])
    # # dist1 = np.array([0.03702081, 0.85215103, 0.00343193, -0.02808688, -1.08401299])

    # cam2 = np.array([[1.4409744e+03, 0., 7.74076208e+02],
    #                 [0., 1.42435001e+03, 5.27489490e+02],
    #                 [0., 0., 1.]])
    # dist2 = np.array([-0.23991173, 1.3750377, -0.03521028, -0.06381475, -3.57839555])

    # Perform Stereo Rectification
    (R1, R2, P1, P2, Q, roi1, roi2) = cv2.stereoRectify(cam1, \
        dist1, cam2, dist2, imsize, R, T)

    cameraObj.R1, cameraObj.R2 = R1, R2
    cameraObj.P1, cameraObj.P2 = P1, P2
    cameraObj.Q = Q

    # Get optimal new camera matrices from the rectified projection matrices
    K_L = cv2.decomposeProjectionMatrix(P1)
    K1 = K_L[0]
    K_R = cv2.decomposeProjectionMatrix(P2) 
    K2 = K_R[0]

    cameraObj.K1, cameraObj.K2 = K1, K2

    return cameraObj

def remapImages(imgL, imgR, cameraObj, imsize, folder):
    # Undistort
    cam1, dist1, R1, K1 = cameraObj.cam1, cameraObj.dist1, \
        cameraObj.R1, cameraObj.K1

    cam2, dist2, R2, K2 = cameraObj.cam2, cameraObj.dist2, \
        cameraObj.R2, cameraObj.K2
    map1x, map1y = cv2.initUndistortRectifyMap(cam1, dist1, R1, K1, imsize, 5)
    map2x, map2y = cv2.initUndistortRectifyMap(cam2, dist2, R2, K2, imsize, 5)

    # Remap the Rectified Images
    # remapped images
    r_imgL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    r_imgR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # Save the Rectified Images
    cv2.imwrite(folder + "rectify_imgL.png", r_imgL)
    cv2.imwrite(folder + "rectify_imgR.png", r_imgR)

    return r_imgL, r_imgR

def computeDisparity(r_imgL, r_imgR, cameraObj):
    window_size = 11
    min_disp = 20
    num_disp = 52 - min_disp # 26
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=5,
                            speckleWindowSize=100,
                            speckleRange=1,
                            disp12MaxDiff=1,
                            P1=8 * 3 * window_size**2,
                            P2=32 * 3 * window_size**2,
                            fullDP = False
                            )
    print 'computing disparity...'
    disp = stereo.compute(r_imgL, r_imgR).astype(np.float32) / 16.0
    disp_value = (disp - min_disp) / num_disp

    h, w = r_imgL.shape[:2]
    points = cv2.reprojectImageTo3D(disp, cameraObj.Q)
    return disp, points

def stereoCalibration():
    imagesL = glob.glob('calibration_samples/l*.jpeg')
    imagesR = glob.glob('calibration_samples/r*.jpeg')

    obj_pts, ptsL, ptsR, imsize = locateStereoPoints(imagesL, imagesR)
    stereoCams = calibrateImages(obj_pts, ptsL, ptsR, imsize)
    
    return imsize, stereoCams

def rectifyImage(imgL, imgR, imsize, stereoCams, folder):
    # Rectified images
    r_imgL, r_imgR = remapImages(imgL, imgR, stereoCams, imsize, folder)
    disp, points = computeDisparity(r_imgL, r_imgR, stereoCams)

    return r_imgL, r_imgR, disp, points


