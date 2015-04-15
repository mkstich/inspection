import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import numpy.linalg as la
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from stereo_calibrate import *
from cannyEllipse import *

def determinePoints(stereoCams, ptsL, ptsR, dL, dR):
	'''Calculate the 3D points using: 
	1) the stereoCams projection matrix Q
	2) ptsL and ptsR the elllipses from the left/right images
	3) dL and dR the disparity values from the left/right images '''

	coordsL, coordsR = [], []
	# Find Left Image World Coordinates
	for e_1 in ptsL:
		x, y = round(e_1.x), round(e_1.y)

		if (x >= 0. and y >= 0.):
			d = dR[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_1.x], [e_1.y], [d], [1.]])
			# coords vector will be in the form [X, Y, Z, W].T
			coords = np.dot(stereoCams.Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			coordsL.append(np.array([X[0], Y[0], Z[0]]))

	# Find Right Image World Coordinates
	for e_2 in ptsR:
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


def createScatter(coordsL, coordsR, pred, folder):
	'''Create scatter plot to map the 3D world coordinates'''

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
	avg_corr = np.round(compute3DPointCorrelation(cL, cR), 4)


	ax.plot(xL, yL, zL, 'o', color='r', label='Left Image')
	ax.plot(xR, yR, zR, 'o', color='b', label='Right Image')
	ax.plot(xP, yP, zP, 'o', color='c', label='Prediction')
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	title = folder + ': ' + str(len(pred)) + ' Matches \
	\n Average Correlation Coefficient = ' + str(avg_corr)
	plt.title(title)

	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.savefig(folder + 'scatterPlot.png')

	plt.show()


def dist3D(e1, e2):
	'''Compute the distance between the X, Y, Z ellipse centers'''
	return np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)

def calcDistances(coordsL, coordsR):
	'''Calculate the distance between the ellipses in imgL and imgR
	using the 3D world coordinates. Keep ellipses that are less than 
	the calculated limit'''

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
	#max(dist, key = lambda dist: dist[0])
	limit = max(dist, key = lambda dist: dist[0])[0] / 2.5

	if limit > 250.:
		limit = 250.

	print limit	
	d2 = [i for i in dist if i[0] <= limit]
	cL, cR = [], []
	for i in d2:
		cL.append(coordsL[int(i[1])])
		cR.append(coordsR[int(i[2])])

	return cL, cR

def computeMeanDist(cL, cR):
	'''Predict where the actual ellipse center should be located
	by computing the mean of the left and right world coords'''
	new_coords = []
	for i, j in zip(cL, cR):
		x = np.mean([i[0], j[0]])
		y = np.mean([i[1], j[1]])
		z = np.mean([i[2], j[2]])
		new_coords.append(np.array([x, y, z]))

	return new_coords

def compute3DPointCorrelation(cL, cR):
	corr = []
	for i, j in zip(cL, cR):
		cc, p_val = stats.pearsonr(i, j)
		corr.append(cc)

	return np.array(corr).mean()

def calculate3DCloud(ptsL, ptsR, stereoCams, dL, dR, folder):
	'''Calculate the 3D world coordinate cloud'''
	# determine 3D points
	coordsL, coordsR = determinePoints(stereoCams, ptsL, ptsR, dL, dR)
	cL, cR = calcDistances(coordsL, coordsR)
	
	avg_corr = compute3DPointCorrelation(cL, cR)
	print 'Average correlation coeff between L and R image pts = ', avg_corr
	# predict actual 3D points
	predPoints = computeMeanDist(cL, cR)

	# plot results
	createScatter(cL, cR, predPoints, folder)

def findEllipses(stereoCams, folder, imsize):
	'''Locate ellipses in a set of images'''
	L_name = imageName(folder, 'left.jpeg')
	R_name = imageName(folder, 'right.jpeg')

	imgL = cv2.imread(L_name)
	imgR = cv2.imread(R_name)
	r_imgL, r_imgR = rectifyImage(imgL, imgR, imsize, stereoCams, folder)

	h1 = imageName(folder,'handlesL.png')
	h2 = imageName(folder,'handlesR.png')
	hL_name = imageName(folder, 'filtered_ellipse_L.png')
	hR_name = imageName(folder, 'filtered_ellipse_R.png')

	rL_name = imageName(folder, 'rectify_imgL.png')
	rR_name = imageName(folder, 'rectify_imgR.png')
	imgL = cv2.imread(rL_name)
	imgR = cv2.imread(rR_name)

	# Compute gradient and threshold for L and R imgs
	handlesL = find_handles(imgL)
	pltplot(handlesL, h1)
	h_img = cv2.imread(h1)

	handlesR = find_handles(imgR)
	pltplot(handlesR, h2)
	r_img = cv2.imread(h2)

	# Find ellipses from gradient images
	e, cts = compute_threshold(handlesL, h_img.copy(), hL_name)
	er, cts_r = compute_threshold(handlesR, r_img.copy(), hR_name)

	return e, er, h_img, r_img

def main():
    imsize, stereoCams = stereoCalibration()
    folders = ['Test '+ str(i) for i in range(1, 5)]

    for folder in folders:
        e, er, h_img, r_img, = findEllipses(stereoCams, folder, imsize)
        dL, pointsL = computeDisparity(h_img, r_img, stereoCams)
        dR, pointsR = computeDisparity(r_img, h_img, stereoCams)
        calculate3DCloud(e, er, stereoCams, dL, dR, folder)

def imageName(folder, name):
	if len(folder) != 0:
		imgName = folder + '/' + name
	else:
		imgName = name

	return imgName


M = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
       [  0.00000000e+00,   1.99299316e+02,   5.52500000e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
# K = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
#        [  0.00000000e+00,   1.70664233e+03,   5.52500000e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
# M_camL = np.array([[  1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
#        [  0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
