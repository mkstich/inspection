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

def determinePoints(stereoCams, ptsL, ptsR, dL, imsize):
	'''Calculate the 3D points using: 
	1) the stereoCams projection matrix Q
	2) ptsL and ptsR the elllipses from the left/right images
	3) dL and dR the disparity values from the left/right images '''
	w, h = imsize
	coordsL, aLength, dvec = [], [], []
	# f = 0.8 * w  # guess for focal length
	cx = stereoCams.M1[0][-1] #* np.sign(stereoCams.K1[0][-1]) #960.8659 
	cy = stereoCams.M1[1][-1] #* np.sign(stereoCams.K1[1][-1])#521.2082 
	f =  np.mean([stereoCams.M1[0][0], stereoCams.M1[1][1]]) #1485.05
	Tx = stereoCams.T[0] #-9.882
	cxp = stereoCams.M2[0][-1] #mn_949.7206
	# stereoCams.Q[0][-1] = -cx #* np.sign(stereoCams.K1[0][-1])
	# stereoCams.Q[1][-1] = -cy #* np.sign(stereoCams.K1[1][-1])
	# stereoCams.Q[3][2] = -1 / Tx
	# stereoCams.Q[3][-1] = 
	q43 = (cx - cxp) / Tx

	# # Q = stereoCams.Q
	Q = np.float32([[1, 0, 0, -cx],
	                [0, -1, 0, cy],  # turn points 180 deg around x-axis,
	                [0, 0, 0, -f],  # so that y-axis looks up
	                [0, 0, -1./Tx, 0]])	
    # f = 0.8 * w  # guess for focal length
	# Q = np.float32([[1, 0, 0, -cx],
 #                    [0, -1, 0, cy],  # turn points 180 deg around x-axis,
 #                    [0, 0, 0, -f],  # so that y-axis looks up
 #                    [0, 0, 1, 0]])	
	# Find Left Image World Coordinates
	for e_1 in ptsL:
		x, y = round(e_1.x), round(e_1.y)
		aLength.append(e_1.a)

		if (x >= 0. and y >= 0.) and (x <= w and y <= h):
			dvec.append(dL[y - 1][x - 1])
			d = dL[y - 1][x - 1]

			# # homogeneous point vector
			x_vec = np.array([[e_1.x], [e_1.y], [d], [1.]])
			# coords vector will be in the form [X, Y, Z, W].T
			# coords = np.dot(stereoCams.Q, x_vec)
			coords = np.dot(Q, x_vec)

			# convert to world coordinates
			X, Y, Z = coords[0] / coords[-1], coords[1] / coords[-1], \
				coords[2] / coords[-1]

			if sum(np.isfinite(np.array([X[0], Y[0], Z[0]]))) == 3:
				coordsL.append(np.array([X[0], Y[0], Z[0]]))
			# coordsL.append(pointsL[y - 1][x - 1])


	# Find Right Image World Coordinates
	# for e_2 in ptsR:
	# 	aLength.append(e_2.a)

	return coordsL, dvec, aLength


def createScatter(coordsL, name):
	'''Create scatter plot to map the 3D world coordinates'''

	xL, yL, zL = [], [], []
	xR, yR, zR = [], [], []
	xP, yP, zP = [], [], []

	for i in coordsL:
		xL.append(i[0])
		yL.append(i[1])
		zL.append(i[2])

	ax = plt.subplot(111, projection='3d')

	ax.plot(xL, yL, zL, 'o', color='r', label='Left Image')
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	title = name 

	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.savefig(name)

	# plt.show()


def dist3D(e1, e2):
	'''Compute the distance between the X, Y, Z ellipse centers'''
	return np.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)

def compute3DPointCorrelation(cL, cR):
	corr = []
	for i, j in zip(cL, cR):
		cc, p_val = stats.pearsonr(i, j)
		corr.append(cc)

	return np.array(corr).mean()

def calculate3DCloud(ptsL, ptsR, stereoCams, dL, name, imsize):
	'''Calculate the 3D world coordinate cloud'''
	# determine 3D points
	coordsL, d, aLength = determinePoints(stereoCams, ptsL, ptsR, dL, imsize)

	# plot results
	createScatter(coordsL, name)

	return coordsL, d, aLength

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
    output, dist, a = [], [], []

    for L, R in zip(imgsL, imgsR):
        # L = imgsL[0]
        # R = imgsR[0]
        x, y, z = splitString(L)
        location = str(x) + '_' + str(y) + '_' + str(z)
        print location
        name = folder + location + '.png'

        eL, eR, l_img, r_img = findEllipses(stereoCams, "distanceResults", imsize, L, R, location)

        # disparity between the left and right image 
        dL = computeDisparity(r_img, l_img, stereoCams)

        coordsL, d, aLength = calculate3DCloud(eL, eR, stereoCams, dL, name, imsize)

        output.append(coordsL)
        dist.append(d)
        a.append(aLength)
        # break

    # new_output = plotAll(output, dist)

    return output, dist, a #new_output, a

def plotAll(output, dist):
	new_output = pd.DataFrame(columns = ('x', 'y', 'z', 'd', 'm'))

	for i in range(len(output)):
		for j in range(len(output[i])):
			if sum(np.isfinite(output[i][j])) == 3:

				trial = pd.Series()
				trial['x'] = output[i][j][0]
				trial['y'] = output[i][j][1]
				trial['z'] = output[i][j][2]
				trial['d'] = dist[i][j]
				# trial['a'] = output[i][j][3]
				# trial['area'] = output[i][j][-1]
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

	new_output.to_csv("matlab_disp_data.csv")
	return new_output

def extractZ(output):
	z = []
	# a_temp = []
	m = output['m'].iloc[0]

	i = 0.
	last = False

	while last is False:
		z_temp = []
		m = output['m'].iloc[i]
		start = i

		if i + m > len(output) - 1:
			end = output.iloc[-1].name
			last = True
		else:
			end = i + m

		for j in range(int(m)):
			z_temp.append(output['z'].iloc[start + j])

		z.append(z_temp)
		i = end

	return z

def getMeanA(a_output, Q, R, name):
	kf = KalmanFilter(Q, R)
	mean_a, unfilter_a_mean = [], []

	for i in a_output:
		s = sorted(i)
		print np.median(s), np.mean(s[2:-2]), np.std(s)
		mn = np.median(s[:-2]) / 2.54 #np.mean(s[:-2]) / 2.54 
		kf.update(mn)
		# kf.update(np.mean(s[:-2]) + np.std(s[:-2]))
		m = kf.predict()
		mean_a.append(m)

		unfilter_a_mean.append(mn)
	
	n = np.linspace(78., 373., 60.)
	origin = n[::-1]

	error = np.sum(abs(mean_a - origin))
	print error
	plt.plot(-origin, mean_a, 'b--', label='Filtered')
	plt.plot(-origin, origin, 'g-', label = 'Actual')
	plt.plot(-origin, unfilter_a_mean, 'b-', label = 'Unfiltered')
	# plt.plot(-origin, origin - mean_a, 'r-', label = "Error")
	plt.ylabel("Calculated Target Distance (in.)")
	plt.xlabel("Distance to Target (in.)")
	plt.xlim(-373, -78)
	plt.title(name + "Q = " + str(Q) + " R = " + str(R))
	plt.legend()
	plt.savefig('case4_median.png')
	plt.show()

	return np.array(mean_a), np.array(unfilter_a_mean)

def sim():
	outM, dM, aM = findDistance()
	new_output = plotAll(outM, dM)

	z = extractZ(new_output)
	most_common = new_output['z'].value_counts().idxmax()
	new_z = []
	for i in range(len(z)):
		z_mod = [x for x in z[i] if x != most_common ]#and x > 0.]
		new_z.append(z_mod)

	Q, R = 2., 5.
	mn_z, u_mn_z = getMeanA(new_z, Q, R, 'case7 ')


d4 = pd.DataFrame.from_csv("case4_disp_data.csv")
d5 = pd.DataFrame.from_csv("case5_disp_data.csv")
d8 = pd.DataFrame.from_csv("case8_disp_data.csv")

case_name = ['case4 ', 'case8 ']
data = [d4, d8]

for name, d in zip(case_name, data):
	print name
	# data = pd.DataFrame.from_csv(name + "_disp_data.csv")
	most_common = d['z'].value_counts().idxmax()
	z = extractZ(d)
	new_z = []
	for i in range(len(z)):
		z_mod = [x for x in z[i] if x != most_common and x > 0.]
		new_z.append(z_mod)

	Q, R = 2., 50.
	mn_z, u_mn_z = getMeanA(new_z, Q, R, name)

# imgs = glob.glob('distance/distance/left_*.jpeg')
# origin = []
# for L in imgs: 
#     x, y, z = splitString(L)
#     location = str(x) + '_' + str(y) + '_' + str(z)
#     current = np.array([x, y, z])
#     actual = np.array([-748, 20., 20.])
#     d = dist3D(current, actual)
#     origin.append(d)