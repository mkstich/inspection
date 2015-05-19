import numpy as np
import cv2
import glob as glob
import numpy.linalg as la


class StereoCam:

    def __init__(self):
        '''Initialize a StereoCam class object'''
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
        self.M1 = np.zeros((3, 3))
        self.M2 = np.zeros((3, 3))

    def printCam(self):
        '''Print a StereoCam class object'''
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

    def findM(self, img_names, side):
        '''Determine the Camera Matrix and update the 
        StereoCam's values for the camera matrix, M
        @img_names = individual camera calibration images
        @side = specify which camera (1 = left, 2 = right)
        '''
        # Specify the calibration checkerboard square and pattern size
        pattern_size = (9, 6)
        square_size = 1.0
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        # locate the object and image points of the checker board in the 
        # calibration images
        obj_points = []
        img_points = []
        h, w = 0, 0
        for fn in img_names:
            img1 = cv2.imread(fn)
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            if img is None:
                print "Failed to load", fn
                continue

            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if not found:
                print 'chessboard not found'
                continue
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

        # Calibrate the camera 
        rms, camera_matrix, dist_coefs, rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

        # undistort images
        K = camera_matrix
        d = np.array([dist_coefs[0][0], dist_coefs[0][1], 0, 0, 0])

        img = cv2.imread('calibration042015/left/l_0.jpeg')
        h, w = img.shape[:2]

        # Update the StereoCam camera matrix with the optimal camera matrix 
        if side == 1:
            self.M1, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)

        if side == 2:
            self.M2, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)

def imageName(folder, name):
    '''Create an image name given img folder
    and name'''

    if len(folder) != 0:
        imgName = folder + '/' + name
    else:
        imgName = name

    return imgName
    
def locateStereoPoints(imagesL, imagesR):
    '''Get the object points (this will be the same for both images) 
    and the image points (the image points will be different per camera)
    to use for stereo calibration
    @imagesL = left calibration images
    @imagesR = right calibration images

    return obj_points, img points from L img, img points from R img, and imsize'''

    # Specify the checkerboard's square and pattern size
    square_size = 1.0
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # locate the object and img points from the calibration images
    obj_points = []
    img_points_L, img_points_R = [], []
    h, w = 0, 0

    for (lname, rname) in zip(imagesL, imagesR):
        img1 = cv2.imread(lname)
        img2 = cv2.imread(rname)

        imgL = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if imgL is None:
            print "Failed to load", lname
            continue

        if imgR is None:
            print "Failed to load", rname
            continue

        # locate the chessboards in the left and right image pair
        h, w = imgL.shape[:2]
        foundL, cornersL = cv2.findChessboardCorners(imgL, pattern_size)
        foundR, cornersR = cv2.findChessboardCorners(imgR, pattern_size)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

        # calibrate the left and right cameras
        if foundL:
            cv2.cornerSubPix(imgL, cornersL, (5, 5), (-1, -1), term)
            img_points_L.append(cornersL.reshape(-1, 2))

        if foundR:
            cv2.cornerSubPix(imgR, cornersR, (5, 5), (-1, -1), term)
            img_points_R.append(cornersR.reshape(-1, 2))

        obj_points.append(pattern_points)

        if not foundL:
            print 'left chessboard not found in', lname
            continue
        if not foundR:
            print 'right chessboard not found in', rname
            continue

    return obj_points, img_points_L, img_points_R, (w, h)


def calibrateImages(obj_pts, ptsL, ptsR, imsize):
    '''Calibrate the Stereo camera rig, given:
    @obj_pts: points that specify the calibration checkerboard pattern
    @ptsL: detected image points in the left calibration images
    @ptsR: detected image points in the right calibration images
    @imsize: the stipulated size of all calibration images
    return: updated StereoCam object
    '''
    # Perform Stereo Calibration
    retval, cam1, dist1, cam2, dist2, R, T, E, F = \
        cv2.stereoCalibrate(
            obj_pts, ptsL, ptsR, imsize)  # , cam1, None, cam2, None)
    dist1 = dist1.ravel()
    dist2 = dist2.ravel()

    cameraObj = StereoCam()
    cameraObj.cam1, cameraObj.dist1 = cam1, dist1
    cameraObj.cam2, cameraObj.dist2 = cam2, dist2

    cameraObj.R, cameraObj.T, cameraObj.E, cameraObj.F = R, T, E, F

    # cam1 = np.array([[1.53216158e+03, 0., 8.96615289e+02],
    #                 [0., 1.43951272e+03, 5.44758043e+02],
    #                 [0., 0., 1.]])
    # dist1 = np.array([0.03702081, 0.85215103, 0.00343193, -0.02808688, -1.08401299])

    # cam2 = np.array([[1.4409744e+03, 0., 7.74076208e+02],
    #                 [0., 1.42435001e+03, 5.27489490e+02],
    #                 [0., 0., 1.]])
    # dist2 = np.array([-0.23991173, 1.3750377, -0.03521028, -0.06381475, -3.57839555])

    # Perform Stereo Rectification
    (R1, R2, P1, P2, Q, roi1, roi2) = \
        cv2.stereoRectify(cam1, dist1, cam2, dist2, imsize, R, T)

    # update the left and right rotation and projection matrices
    cameraObj.R1, cameraObj.R2 = R1, R2
    cameraObj.P1, cameraObj.P2 = P1, P2

    # update the image projection (aka disparity-to-depth mapping) matrix
    cameraObj.Q = Q

    # Get optimal new camera matrices from the rectified projection matrices
    K_L = cv2.decomposeProjectionMatrix(P1)
    K1 = K_L[0]
    K_R = cv2.decomposeProjectionMatrix(P2)
    K2 = K_R[0]

    cameraObj.K1, cameraObj.K2 = K1, K2

    return cameraObj


def remapImages(imgL, imgR, cameraObj, imsize, folder, nameR, nameL):
    '''Remap the images using the calibrated stereo camera parameters
    @imgL = the left rectified images
    @imgR = the right rectified images
    @cameraObj = the updated StereoCam object
    @imsize = the size of the images
    @folder = the calibration folder
    @nameR = the name of the right images
    @nameL = the name of the left images

    return = the saved rectified left and right images
    '''
    # Undistort the images
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

    r_imgL = cv2.medianBlur(r_imgL, 3)
    r_imgR = cv2.medianBlur(r_imgR, 3)

    # Save the Rectified Images
    r_Name = imageName(folder, nameL)
    l_Name = imageName(folder, nameR)
    cv2.imwrite(r_Name, r_imgL)
    cv2.imwrite(l_Name, r_imgR)

    return r_imgL, r_imgR


def computeDisparity(r_imgL, r_imgR, cameraObj):
    '''Compute the disparity between the left and right images
    @r_imgL = rectified left image
    @r_imgR = rectified right image
    @cameraObj = StereoCam object

    return = the disparity and the reprojected to 3D points
    '''
    # Calculate the disparity between the L and R images
    h, w = r_imgL.shape[:2]
    window_size = 9
    min_disp = 0  # 20
    max_disp = w / 8  # image width / 8
    num_disp = max_disp - min_disp  # 52 - min_disp # 26
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=10,  # 5,
                            speckleWindowSize=100,
                            speckleRange=32,
                            disp12MaxDiff=1,
                            preFilterCap=63,
                            P1=8 * 3 * window_size**2,
                            P2=32 * 3 * window_size**2,
                            fullDP=False
                            )
    print 'computing disparity...'

    # Normalize the values
    disp = stereo.compute(r_imgL, r_imgR).astype(
        np.float32) / 16.  # max_disp #/52.

    # Update the Q matrix
    cx = cameraObj.M1[0][-1]
    cy = cameraObj.M1[1][-1]
    f =  np.mean([cameraObj.M1[0][0], cameraObj.M1[1][1]])
    Tx = cameraObj.T[0]
    cxp = cameraObj.M2[0][-1]
    q43 = (cx - cxp) / Tx

    Q = np.float32([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],  # turn points 180 deg around x-axis,
                    [0, 0, 0, f],  # so that y-axis looks up
                    [0, 0, -1./Tx, q43]])   
    
    # Compute the 3D World Coordinates - these points are world coordinates
    # relative to the image's specific image frame
    points = cv2.reprojectImageTo3D(disp, Q)

    return disp, points

def rectifyImage(imgL, imgR, imsize, stereoCams, folder, nameR, nameL):
    '''Rectify the images using the calibration parameters
    @imgL = left img
    @imgR = right img
    @imsize = image size

    return = rectified left and right images
    '''
    # Rectified images
    r_imgL, r_imgR = remapImages(imgL, imgR, stereoCams, imsize, folder, nameR, nameL)

    return r_imgL, r_imgR

def stereoCalibration():
    '''Calibrate a Stereo Camera rig
    Note: the stereo cameras must be displaced in either a completely horizontal 
    or a completely vertical orientation in order to attain accruate results
    '''
    # Gather calibration images
    imagesL = sorted(glob.glob('calibration_samples/l*.jpeg'))
    imagesR = sorted(glob.glob('calibration_samples/r*.jpeg'))

    # Locate Stereo Calibration points
    obj_pts, ptsL, ptsR, imsize = locateStereoPoints(imagesL, imagesR)
    if(len(ptsL) == len(ptsR)) and len(obj_pts) > len(ptsL):
        while len(obj_pts) > len(ptsL):
            del obj_pts[-1]

    # Calibrate the images
    stereoCams = calibrateImages(obj_pts, ptsL, ptsR, imsize)

    # Update the individual camera matrices
    M_imagesL = sorted(glob.glob('calibration042015/left/l_*.jpeg'))
    M_imagesR = sorted(glob.glob('calibration042015/right/r_*.jpeg'))

    stereoCams.findM(M_imagesL, 1)
    stereoCams.findM(M_imagesR, 2)

    return imsize, stereoCams



