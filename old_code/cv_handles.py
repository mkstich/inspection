import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_handles(img):
    lower_handles = np.array([50,  50,  0], dtype=np.uint8)
    upper_handles = np.array([250, 150, 25], dtype=np.uint8)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(RGB, lower_handles, upper_handles)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def pltplot(img):
    BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(BGR)
    plt.show()


def disparity(imgL, imgR):
    window_size = 11
    min_disp = 0
    num_disp = 16 - min_disp
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=5,
                            speckleWindowSize=100,
                            speckleRange=1,
                            disp12MaxDiff=1,
                            P1=8 * 3 * window_size**2,
                            P2=32 * 3 * window_size**2,
                            fullDP=True
                            )
    print 'computing disparity...'
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity = (disp - min_disp) / num_disp
    return disparity


imgL = cv2.imread('left.png')
imgR = cv2.imread('right.png')

handlesL = find_handles(imgL)
handlesR = find_handles(imgR)

disp = disparity(handlesL, handlesR)
plt.imshow(disp)