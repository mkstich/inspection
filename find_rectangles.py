import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math


def find_handles(img, name):
    '''Detect the orange handles in the image by applying the mask'''
    lower_handles = np.array([20,  20,  0], dtype=np.uint8)  # 35, 50
    upper_handles = np.array([255, 255, 20], dtype=np.uint8)  # 255, 150, 30

    # Apply the color range to the image
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(RGB, lower_handles, upper_handles)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Save the parsed image
    cv2.imwrite(name, res)
    return res

def compute_threshold(img, fname, small):
    '''Use Canny edge detection to find the contours, 
    Sort the contours into ellipses and then filter out ellipses 
    that do not have a small minor axis to major axis ratio'''

    # Get the edge map of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # Find the contours and sort the contours from large to small area
    (contours, _) = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    rectangles = []
    cnts = []
    for cnt in contours:
        try:
            if small is True:
                rectangle = cv2.minAreaRect(cnt)

            else:
                rectangle = cv2.boundingRect(cnt)

            rectangles.append(rectangle)
            # cv2.drawContours(img, [box], (0, 255, 0), 2)

        except Exception as e:
            pass

    # Drop repeats
    rectangles = list(set(rectangles))

    # Draw the elements
    # Keep the largest
    finalRect = []
    vertical = False

    # Grab the two largest rectangles
    if small is True:
        rectangles.sort(key = lambda rectangle: rectangle[1][0] * rectangle[1][1],
            reverse = True)
        r = rectangles[0:2]

        if len(r) > 0:
            x1, y1 = r[0][0]
            w1, h1 = r[0][1]

            if h1 > w1:
                vertical = True

            # Draw the largest rectangle
            box = cv2.cv.BoxPoints(r[0])
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
            finalRect.append(r[0])

            repeat = False
            # Keep the second largest, if the second is not contained
            # within the first rectangle
            x2, y2 = r[1][0]
            w2, h2 = r[1][1]

            if h2 > w2:
                vertical = True

            if (x1 - 10 <= x2 and x2 <= x1 + w1) and (y1 - 10 <= y2 and y2 <= y1 + h1):
                repeat = True

            # Draw the second rect. if not a repeat
            if repeat is False:
                box = cv2.cv.BoxPoints(r[1])
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                finalRect.append(r[1])

    else:
        rectangles.sort(
            key=lambda rectangle: rectangle[2] * rectangle[3], reverse=True)

        r = rectangles[0:2]

        if len(r) > 0:
            x1, y1, w1, h1 = r[0]

            if h1 > w1:
                vertical = True

            cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            finalRect.append(r[0])

            repeat = False
            # Keep the second largest, if the second is not contained
            # within the first rectangle
            x2, y2, w2, h2 = r[1]

            if h2 > w2:
                vertical = True

            if (x1 <= x2 and x2 <= x1 + w1) and (y1 <= y2 and y2 <= y1 + h1):
                repeat = True

            # Draw the second rect. if not a repeat
            # Also draw if the second rect is not vertical and not small
            if repeat is False:
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                finalRect.append(r[1])

        # Save the result
        cv2.imwrite(fname, img)

    return finalRect
