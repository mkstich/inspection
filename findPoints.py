import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from main_test import *
from stereo_calibrate import *
from cannyEllipse import *

def determinePoints(stereoCams, ptsL, ptsR, dL, dR):

	coordsL, coordsR = [], []
	# Find Left Image World Coordinates
	for e_1 in e1:
		x, y = round(e_1.x), round(e_1.y)

		if (x > 0. and y > 0.):
			d = dR[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_1.x], [e_1.y], [d], [1.]])
			coords = np.dot(stereoCams.Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			coordsL.append(np.array([X[0], Y[0], Z[0]]))

	# Find Right Image World Coordinates
	for e_2 in e2:
		x, y = round(e_2.x), round(e_2.y)

		if (x > 0. and y > 0.):
			d = dL[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_2.x], [e_2.y], [d], [1.]])
			coords = np.dot(stereoCams.Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			coordsR.append(np.array([X[0], Y[0], Z[0]]))

	return coordsL, coordsR


def createScatter(coordsL, coordsR, pred):
	xL, yL, zL = [], [], []
	xR, yR, zR = [], [], []
	xP, yP, zP = [], [], []

	for i in coordsL:
		xL.append(i[0])
		yL.append(i[1])
		zL.append(i[2])

	for i in coordsR:
		xR.append(i[0])
		yR.append(i[1])
		zR.append(i[2])

	for i in pred:
		xP.append(i[0])
		yP.append(i[1])
		zP.append(i[2])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(xL, yL, zL, c='r', marker = 'o')
	# ax.scatter(xR, yR, zR, c='b', marker = 'o')
	# ax.scatter(xP, yP, zP, c='c', marker = 'o')

	ax = plt.subplot(111, projection='3d')

	ax.plot(xL, yL, zL, 'o', color='r', label='Left Image')
	ax.plot(xR, yR, zR, 'o', color='b', label='Right Image')
	ax.plot(xP, yP, zP, 'o', color='c', label='Prediction')
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')

	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

	plt.show()


def dist3D(e1, e2):
	return np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)


def calcDistances(coordsL, coordsR):
	dist = []
	for i in range(len(coordsL)):
		pts = []
		for j in range(len(coordsR)):
			d = dist3D(coordsL[i], coordsR[j])
			# print d
			pts.append((d, i, j))

		min_pair = min(pts, key = lambda pts: pts[0])
		dist.append(np.array(min_pair))

	# Find the i, j pairs where dist < 200.
	d2 = [i for i in dist if i[0] <= 200.]
	cL, cR = [], []
	for i in d2:
		cL.append(coordsL[int(i[1])])
		cR.append(coordsR[int(i[2])])

	return cL, cR

def computeMeanDist(cL, cR):
	new_coords = []
	for i, j in zip(cL, cR):
		x = np.mean([i[0], j[0]])
		y = np.mean([i[1], j[1]])
		z = np.mean([i[2], j[2]])
		new_coords.append(np.array([x, y, z]))

	return new_coords


def calculate3DCloud(ptsL, ptsR, stereoCams, dL, dR):
	# determine 3D points
	coordsL, coordsR = determinePoints(stereoCams, ptsL, ptsR, dL, dR)
	cL, cR = calcDistances(coordsL, coordsR)
	
	# predict actual 3D points
	predPoints = computeMeanDist(cL, cR)

	# plot results
	createScatter(cL, cR, predPoints)

def findEllipses(stereoCams, folder):
	imgL = cv2.imread(folder + '/left.jpeg')
	imgR = cv2.imread(folder + '/right.jpeg')
	r_imgL, r_imgR, disp, points = \
		rectifyImage(imgL, imgR, imsize, stereoCams, folder + '/')

	h1 = folder + '/handlesL'
	h1_full = h1 + '.png'
	h2 = folder + '/handlesR'
	h2_full = h2 + '.png'
	hL_name = folder + '/filtered_ellipse_L'
	hR_name = folder + '/filtered_ellipse_R'
	all_name = folder + '/all_and_best'

	imgL = cv2.imread(folder + '/rectify_imgL.png')
	imgR = cv2.imread(folder + '/rectify_imgR.png')

	handlesL = find_handles(imgL)
	pltplot(handlesL, h1)
	h_img = cv2.imread(h1_full)

	handlesR = find_handles(imgR)
	pltplot(handlesR, h2)
	r_img = cv2.imread(h2_full)

	e, cts = compute_threshold(handlesL, h_img.copy(), hL_name)
	er, cts_r = compute_threshold(handlesR, r_img.copy(), hR_name)

	return e, er, h_img, r_img

def main():
    imsize, stereoCams = stereoCalibration()
    folders = ['Test '+ str(i) for i in range(1, 5)]

    for folder in folders:
        e, er, h_img, r_img, = findEllipses(stereoCams, folder)
        # dL, pointsL = computeDisparity(h_img, r_img, stereoCams)
        # dR, pointsR = computeDisparity(r_img, h_img, stereoCams)
        calculate3DCloud(e, er, stereoCams, dL, dR)

# a = stereoCams.Q[3][2]
# b = stereoCams.Q[3][3]
# f = stereoCams.Q[2][3]
# cx1 = stereoCams.P1[0][2]
# cy1 = stereoCams.P1[1][2]

# d = (f - Z * b) / (Z * a)
# ix = X * (d * a + b) + cx1
# iy = Y * (d * a + b) + cy1
