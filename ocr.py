import sys
import cv2
import skimage.morphology as morphology
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import glob
import re
import pandas as pd
import numpy.linalg as la

# try:
#     import Image
# except ImportError:
#     from PIL import Image
# import pytesseract

# im = cv2.imread('left_-855.0_230.0_153.0.jpeg')
# im = cv2.imread('right_-855.0_130.0_153.0.jpeg')
# im = cv2.imread('right_-855.0_350.0_153.0.jpeg')
# im = cv2.imread('left_-855.0_210.0_153.0.jpeg')
# im = cv2.imread('nike-rsvp-ocr-im1.png')
images = ["input/handle_-795_230_120_-35_0_0.jpeg",
          "input/handle_-795_430_120_-35_0_0.jpeg",
          "input/handle_-810_230_110_-35_0_0.jpeg",
          "input/left_-855.0_230.0_153.0.jpeg",
          "input/left_-855.0_160.0_153.0.jpeg",
          "input/right_-855.0_130.0_153.0.jpeg",
          "input/right_-855.0_350.0_153.0.jpeg",
          "input/left_-855.0_210.0_153.0.jpeg"
          # "input/50/handle_-816_137_105_-35_0_0.jpeg",
          # "input/50/handle_-816_148_105_-35_0_0.jpeg",
          # "input/50/handle_-816_164_105_-35_0_0.jpeg",
          # "input/50/handle_-816_200_105_-35_0_0.jpeg",
          # "input/50/handle_-816_252_105_-35_0_0.jpeg",
          # "input/50/handle_-816_405_105_-35_0_0.jpeg",
          # "input/50/handle_-816_445_105_-35_0_0.jpeg",
          ]

data = []
for name in images:
    im = cv2.imread(name)
    x, y, z = name.split('_')[1:4]
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im2 = cv2.adaptiveThreshold(
        im1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 0)
    cv2.floodFill(im2, None, (0, 0), 255)
    # im3 = morphology.skeletonize(im2 > 0) * 255
    # im4 = cv2.copyMakeBorder(im3, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # cv2.floodFill(im4, None, (0, 0), 255)
    im5 = cv2.erode(
        255 - im2.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)

    cnt = cv2.findContours(im5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnt = [c for c in cnt if (cv2.contourArea(c) > 1000)] #> 3000

    mask = np.zeros(im.shape[:2], np.uint8)
    cv2.drawContours(mask, cnt, -1, 255, -1)
    cv2.imshow('', mask)
    cv2.waitKey()

    # dst = im2 & mask
    dst = mask - im2
    dst[dst[:,]==1] = 0
    cnt = cv2.findContours(
        dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50
    cv2.drawContours(dst, cnt, -1, 0, -1)
    cv2.floodFill(dst, None, (0, 0), 255)
    cv2.imshow('regfill', dst)
    cv2.waitKey()
    cv2.imwrite('skel_contour_' + x +'_' + y + '_' + z + 'regfill.png', dst)
    
    # Again subtracting from 255
    dst = 255 - mask - im2
    cnt = cv2.findContours(
        dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnt = [c for c in cnt if cv2.contourArea(c) < 1]  # < 50
    cv2.drawContours(dst, cnt, -1, 0, -1)
    cv2.floodFill(dst, None, (0, 0), 255)    
    cv2.imshow('minus fill', dst)
    cv2.waitKey()
    # cv2.imwrite('skel_contour_' + x +'_' + y + '_' + z + 'minusfill.png', dst)

    # print(pytesseract.image_to_string(
    #     Image.open('skel_contour' + x + y + z + '.png')))


# api = tesseract.TessBaseAPI()
# api.Init(".", "eng", tesseract.OEM_DEFAULT)
# api.SetVariable("tessedit_char_whitelist", "#1234567890")
# api.SetPageSegMode(tesseract.PSM_SINGLE_LINE)

# image = cv.CreateImageHeader(dst.shape[:2], cv.IPL_DEPTH_8U, 1)
# cv.SetData(image, dst.tostring(), dst.dtype.itemsize * dst.shape[1])
# tesseract.SetCvImage(image, api)
# print api.GetUTF8Text().string()
