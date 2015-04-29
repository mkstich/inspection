import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import glob
import pandas as pd
import numpy.linalg as la
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from stereo_calibrate import *
from find_ellipse import *

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

def determinePoints(stereoCams, ptsL, ptsR, dL, dR, imsize):
	'''Calculate the 3D points using: 
	1) the stereoCams projection matrix Q
	2) ptsL and ptsR the elllipses from the left/right images
	3) dL and dR the disparity values from the left/right images '''
	w, h = imsize
	coordsL, coordsR = [], []
	# c2L, c2R = [], []
	# Find Left Image World Coordinates
	for e_1 in ptsL:
		x, y = round(e_1.x), round(e_1.y)

		if (x >= 0. and y >= 0.) and (x <= w and y <= h):
			d = dR[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_1.x], [e_1.y], [d], [1.]])
			# coords vector will be in the form [X, Y, Z, W].T
			coords = np.dot(stereoCams.Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			coordsL.append(np.array([X[0], Y[0], Z[0]]))
			# print "distance = ",10. * 1920. / (2 * math.tan(0.912423 / 2.) * d)
			# print (np.linalg.norm(stereoCams.T) * w) / (2 * math.tan(0.5 * 4.211) * d), Z
			# c2L.append(pointsL[y-1][x-1])

	# Find Right Image World Coordinates
	for e_2 in ptsR:
		x, y = round(e_2.x), round(e_2.y)

		if (x >= 0. and y >= 0.) and (x <= w and y <= h):
			d = dL[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_2.x], [e_2.y], [d], [1.]])
			coords = np.dot(stereoCams.Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			coordsR.append(np.array([X[0], Y[0], Z[0]]))
			# c2R.append(pointsR[y - 1][x - 1])
	return coordsL, coordsR


def createScatter(coordsL, coordsR, pred, name):
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
	avg_corr = np.round(compute3DPointCorrelation(coordsL, coordsR), 4)


	ax.plot(xL, yL, zL, 'o', color='r', label='Left Image')
	ax.plot(xR, yR, zR, 'o', color='b', label='Right Image')
	ax.plot(xP, yP, zP, 'o', color='c', label='Prediction')
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	title = name + ': ' + str(len(pred)) + ' Matches \
	\n Average Correlation Coefficient = ' + str(avg_corr)
	plt.title(title)

	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.savefig(name)

	# plt.show()


def dist3D(e1, e2):
	'''Compute the distance between the X, Y, Z ellipse centers'''
	return np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)

def calcDistances(coordsL, coordsR, eL, eR):
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
		if sum(np.isfinite(np.array(min_pair))) == 3:
			dist.append(np.array(min_pair))

	# Find the i, j pairs where dist < 200.
	#max(dist, key = lambda dist: dist[0])
	limit = max(dist, key = lambda dist: dist[0])[0] / 2.5

	# print limit

	if limit < 250.:
		limit = 250.

	print limit	
	d2 = [i for i in dist if i[0] <= limit]
	cL, cR = [], []
	areaL, areaR = [], []
	a_L, a_R = [], []
	for i in d2:
		# world coordinates
		cL.append(coordsL[int(i[1])])
		cR.append(coordsR[int(i[2])])

		# area
		areaL.append(ellipseArea(eL[int(i[1])].a, eL[int(i[1])].b))
		areaR.append(ellipseArea(eR[int(i[2])].a, eR[int(i[2])].b))

		# semi - major axis
		a_L.append(eL[int(i[1])].a)
		a_R.append(eL[int(i[1])].a)

	return cL, cR, areaL, areaR, a_L, a_R

def computeMeanDist(cL, cR, areaL, areaR, aL, aR):
	'''Predict where the actual ellipse center should be located
	by computing the mean of the left and right world coords'''
	new_coords = []
	for i, j, a1, a2, area1, area2 in zip(cL, cR, aL, aR, areaL, areaR):
		x = np.mean([i[0], j[0]])
		y = np.mean([i[1], j[1]])
		z = np.mean([i[2], j[2]])
		area = np.mean([area1, area2])
		a = np.mean([a1, a2])
		new_coords.append(np.array([x, y, z, a, area]))

	return new_coords

def compute3DPointCorrelation(cL, cR):
	corr = []
	for i, j in zip(cL, cR):
		cc, p_val = stats.pearsonr(i, j)
		corr.append(cc)

	return np.array(corr).mean()

def calculate3DCloud(ptsL, ptsR, stereoCams, dL, dR, name, imsize):
	'''Calculate the 3D world coordinate cloud'''
	# determine 3D points
	coordsL, coordsR = determinePoints(stereoCams, ptsL, ptsR, dL, dR, imsize)
	cL, cR, areaL, areaR, aL, aR = calcDistances(coordsL, coordsR, ptsL, ptsR)
	
	avg_corr = compute3DPointCorrelation(cL, cR)
	print "matches = ", len(cL)
	print 'Average correlation coeff between L and R image pts = ', avg_corr
	# predict actual 3D points
	predPoints = computeMeanDist(cL, cR, areaL, areaR, aL, aR)

	# plot results
	createScatter(cL, cR, predPoints, name)

	return predPoints

def findEllipses(stereoCams, folder, imsize, L_name, R_name, loc):
	'''Locate ellipses in a set of images'''

	imgL = cv2.imread(L_name)
	imgR = cv2.imread(R_name)
	r_imgL, r_imgR = rectifyImage(imgL, imgR, imsize, stereoCams, folder)

	h1 = imageName(folder,'handlesL.png')
	h2 = imageName(folder,'handlesR.png')
	hL_name = imageName(folder, 'filtered_ellipse_L' + loc + '.png')
	hR_name = imageName(folder, 'filtered_ellipse_R' + loc + '.png')

	rL_name = imageName(folder, 'rectify_imgL.png')
	rR_name = imageName(folder, 'rectify_imgR.png')
	imgL = cv2.imread(rL_name)
	imgR = cv2.imread(rR_name)

	# Compute gradient and threshold for L and R imgs
	handlesL = find_handles(imgL, h1)
	h_img = cv2.imread(h1)

	handlesR = find_handles(imgR, h2)
	r_img = cv2.imread(h2)

	# Find ellipses from gradient images
	e, cts = compute_threshold(handlesL, h_img.copy(), hL_name)
	er, cts_r = compute_threshold(handlesR, r_img.copy(), hR_name)

	return e, er, r_imgL, r_imgR #h_img, r_img

def buildScatterPlots():
    imsize, stereoCams = stereoCalibration()
    folders = ['Test '+ str(i) for i in range(1, 5)]

    for folder in folders:
        L_name = imageName(folder, 'left.jpeg')
        R_name = imageName(folder, 'right.jpeg')
        e, er, h_img, r_img, = findEllipses(stereoCams, folder, imsize, L_name, R_name)

        dL, pointsL = computeDisparity(h_img, r_img, stereoCams)
        dR, pointsR = computeDisparity(r_img, h_img, stereoCams)
        calculate3DCloud(e, er, stereoCams, dL, dR, folder +'/')

def imageName(folder, name):
	if len(folder) != 0:
		imgName = folder + '/' + name
	else:
		imgName = name

	return imgName

def splitString(name):
    vals = name.split('_')
    y = vals[-1].split('.')[0]
    val1 = [float(vals[-3]), float(vals[-2]), float(y)]

    return val1

def findDistance():
    imgsL = glob.glob('distance/distance/left_*.jpeg')
    imgsR = glob.glob('distance/distance/right_*.jpeg')

    imsize, stereoCams = stereoCalibration()
    folder = "distanceResults/"
    output, dist = [], []

    for L, R in zip(imgsL, imgsR):
        # L = imgsL[0]
        # R = imgsR[0]
        x, y, z = splitString(L)
        location = str(x) + '_' + str(y) + '_' + str(z)
        print location
        name = folder + location + '.png'

        eL, eR, l_img, r_img = findEllipses(stereoCams, "distanceResults", imsize, L, R, location)

        # disparity between the left and right image --> use for right 3D points
        dL, pointsL = computeDisparity(l_img, r_img, stereoCams)

        # disparity between the right and left image --> use for left 3D points
        dR, pointsR = computeDisparity(r_img, l_img, stereoCams)

        pred = calculate3DCloud(eL, eR, stereoCams, dL, dR, name, imsize)
        pred_dist = []
        e2 = np.array([-675., 20., 20.]) # EDGE origin
        for i in pred:
        	pred_dist.append(dist3D(i[0:3], e2))

        output.append(pred)
        dist.append(pred_dist)
        # break

    new_output = plotAll(output, dist)

    return output, dist, new_output

def plotAll(output, dist):
	new_output = pd.DataFrame(columns = ('x', 'y', 'z', 'd', 'm', 'a', 'area'))

	for i in range(len(output)):
		for j in range(len(output[i])):
			trial = pd.Series()
			trial['x'] = output[i][j][0]
			trial['y'] = output[i][j][1]
			trial['z'] = output[i][j][2]
			trial['d'] = dist[i][j]
			trial['a'] = output[i][j][3]
			trial['area'] = output[i][j][-1]
			trial['m'] = len(output[i])

			new_output = new_output.append(trial, ignore_index = True)

	ax = plt.subplot(111, projection = '3d')
	ax.plot(new_output['x'], new_output['y'], new_output['z'], 'o', color='r', label='Left Image')

	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	# ax.set_zlim(-200, -100)
	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.show()

	new_output.to_csv("modified_disp_data.csv")
	return new_output

def distPredict(dist):
	min_d, old_min = [], []
	mean_d, old_mean_d = [], []
	Q, R = 0.01, 1.
	kf = KalmanFilter(Q, R)
	kfm = KalmanFilter(Q, R)

	for i in dist:
		m = min(i)
		old_min.append(m)
		kf.update(m)
		m = kf.predict()
		min_d.append(m)

		mn = np.mean(i)
		kfm.update(mn)
		mn = kfm.predict()
		mean_d.append(mn)
		old_mean_d.append(np.mean(i))

	old_min = np.array(old_min) / 96.
	min_d = np.array(min_d) / 96.
	mean_d = np.array(mean_d) / 96.
	old_mean_d = np.array(old_mean_d) / 96.

	origin = []
	for i in imgsL:
		x, y, z = splitString(i)
		print x, y, z
		origin.append(dist3D(np.array([x, y, z]), np.array([-675., 20., 20.])))

	origin = np.array(origin)

def distLengthPredict(stereoCams, length):
	f1 = np.mean([stereoCams.M1[0][0], stereoCams.M2[0][0]])
	f2 = np.mean([stereoCams.M2[0][0], stereoCams.M2[1][1]])
	f = np.mean([f1, f2])
	f_img = 35. #588.4 # mm (FOV = 4.211 degrees)
	m = f / f_img
	obj_real_world = 350. #404.157 # mm
	sensor = length / m

	distance = obj_real_world * f_img / sensor

	# convert to inches
	distance = (distance / 10.) / 2.54

	# compare to actual results
	n = np.linspace(5., 300., 60.)
	origin = n[::-1]

	error = np.sum(distance - origin)
	print error
	# plot results
	plt.plot(distance)
	plt.plot(origin)
	# plt.savefig("dist_predict.png")
	plt.show()

	return distance

output = pd.DataFrame.from_csv("modified_disp_data.csv")

def extractA(output):
	a = []
	# a_temp = []
	m = output['m'].iloc[0]

	i = 0.
	last = False

	while last is False:
		a_temp = []
		m = output['m'].iloc[i]
		start = i

		if i + m > len(output) - 1:
			end = output.iloc[-1].name
			last = True
		else:
			end = i + m

		for j in range(int(m)):
			a_temp.append(output['a'].iloc[start + j] * 2.)

		a.append(a_temp)
		i = end

	return a

def getMaxA(a_output, Q, R):
	kf = KalmanFilter(Q, R)
	max_a, unfilter_a_max = [], []

	for i in a_output:
		# s = sorted(i)
		kf.update(max(i) - np.std(i))#s[-3])
		m = kf.predict()
		max_a.append(m)

		unfilter_a_max.append(max(i) - np.std(i))#s[-3])

	plt.plot(max_a, label='Filtered')
	plt.plot(unfilter_a_max, label = 'Unfiltered')
	plt.legend()
	plt.title("Q = " + str(Q) + " R = " + str(R))
	plt.savefig("max_a.png")
	plt.show()

	return np.array(max_a), np.array(unfilter_a_max)

def getMeanA(a_output, Q, R):
	kf = KalmanFilter(Q, R)
	mean_a, unfilter_a_mean = [], []

	for i in a_output:
		s = sorted(i)
		# kf.update(np.mean(i) + np.std(i))
		kf.update(np.mean(s[:-2]) + np.std(s[:-2]))
		m = kf.predict()
		mean_a.append(m)

		unfilter_a_mean.append(np.mean(s[:-2]) + np.std(s[:-2]))

	plt.plot(mean_a, label='Filtered')
	plt.plot(unfilter_a_mean, label = 'Unfiltered')
	plt.title("Q = " + str(Q) + " R = " + str(R))
	plt.legend()
	plt.show()

	return np.array(mean_a), np.array(unfilter_a_mean)

def getMinA(a_output, Q, R):
	kf = KalmanFilter(Q, R)
	min_a, unfilter_a_min = [], []

	for i in a_output:
		kf.update(min(i))
		m = kf.predict()
		min_a.append(m)

		unfilter_a_min.append(min(i))

	plt.plot(min_a, label='Filtered')
	plt.plot(unfilter_a_min, label = 'Unfiltered')
	plt.title("Q = " + str(Q) + " R = " + str(R))
	plt.savefig("min_a.png")
	plt.legend()
	plt.show()

	return np.array(min_a), np.array(unfilter_a_min)


Qest = np.linspace(.10, 10., 10)

for q in Qest:
	Q, R = 2., 200#10., 50.
	new_mean, new_u_mean = getMeanA(a, Q, R)
	d = distLengthPredict(stereoCams, new_mean)

