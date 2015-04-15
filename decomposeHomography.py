import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import glob as glob
import math
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D

def drawMatches(img1, kp1, img2, kp2, matches, name):
    """
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in (matches):

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    cv2.imwrite(name, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def featureMatching(img1, img2, name):
	MIN_MATCH_COUNT = 10

	# Initiate SIFT detector
	sift = cv2.SIFT()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict()#(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# calculate the homography matrix
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		# h, w = img1.shape
		# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		# dst = cv2.perspectiveTransform(pts,M)

	# plot the first 100 matches 
	drawMatches(img1, kp1, img2, kp2, good[:100], name)

	return M, src_pts, dst_pts

def detAttitude(H, M, ar, ap, ay):
	''' Gary Bradski's OpenCV Textbook method 
	http://www.cs.haifa.ac.il/~dkeren/ip/OReilly-LearningOpenCV.pdf
	Used formulas from Ch. 11 pg. 389 'What's under the hood?'

	H = Homography matrix
	M = camera matrix
	'''
	h1 = H[:, 0]
	h2 = H[:, 1]
	h3 = H[:, 2]
	
	# calculate the scaling factor, lda
	lda = 1. / la.norm(h1)#np.dot(la.inv(M), h1))

	# compute the components of the rotation matrix
	r1 = np.dot(lda, np.dot(np.linalg.inv(M), h1))
	r2 = np.dot(lda, np.dot(np.linalg.inv(M), h2))
	r3 = np.cross(r1, r2)

	R = np.array([r1, r2, r3])

	# U, S, Vt = la.svd(R)
	# R = np.dot(U, Vt)

	# translation vector
	t = np.dot(lda, np.dot(np.linalg.inv(M), h3))

	# r, p, y = 
	return calcAngles(R, ar, ap, ay)

	# return r, p, y, R

def error(ar, r, ap, p, ay, y):
	e1 = np.abs(ar - r)
	e2 = np.abs(ap - p)
	e3 = np.abs(ay - y)

	return np.mean([e1, e2, e3])
	# return np.mean(np.abs((ar - np.abs(r))) + np.abs((ap - np.abs(p))) + \
		# np.abs(ay - np.abs(y)))

def PRY(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.atan2(r32, r22)), 4)
	roll = np.round(math.degrees(math.asin(r13)), 4)
	yaw = np.round(math.degrees(math.atan2(r13, r11)), 4)

	return pitch, roll, yaw

def PYR(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.atan2(-r23, r33)), 4)
	roll = np.round(math.degrees(math.atan2(-r12, r11)), 4)
	yaw = np.round(math.degrees(math.asin(r13)), 4)

	return pitch, roll, yaw

def YPR(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.asin(-r23)), 4)
	roll = np.round(math.degrees(math.atan2(r21, r22)), 4)
	yaw = np.round(math.degrees(math.atan2(r13, r33)), 4)

	return pitch, roll, yaw

def YRP(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.atan2(-r23, r22)), 4)
	roll = np.round(math.degrees(math.asin(r21)),4 )
	yaw = np.round(math.degrees(math.atan2(-r31, r11)), 4)

	return pitch, roll, yaw

def RYP(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.atan2(r32, r33)), 4)
	roll = np.round(math.degrees(math.atan2(r21, r11)), 4)
	yaw = np.round(math.degrees(math.asin(-r31)), 4)

	return pitch, roll, yaw

def RPY(R):
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat
	pitch = np.round(math.degrees(math.asin(r32)), 4)
	roll = np.round(math.degrees(math.atan2(-r12, r22)), 4)
	yaw = np.round(math.degrees(math.atan2(-r31, r33)), 4)

	return pitch, roll, yaw

def calcAngles(R, ar, ap, ay):
	'''
	heading = theta = yaw (about y)
	bank = psi = pitch (about x)
	attitude = phi = roll (about z)

	EDGE rotation matrixes will be in the order of operations
	of pitch, yaw, roll
	'''

	'''Order of Euler Angles: pitch, yaw, roll'''
	# if singularity at the north pole
	# if r21 > 0.998:
	# 	yaw = np.round(math.degrees(math.atan2(r13, r33)), 4)
	# 	roll = np.round(math.degrees(math.pi / 2.), 4)
	# 	pitch = 0.0

	# # if sinularity at south pole
	# elif r21 < -0.998:
	# 	yaw = np.round(math.degrees(math.atan2(r13, r33)), 4)
	# 	roll = np.round(math.degrees(math.pi / 2.), 4)
	# 	bank = 0.

	# else:
	# 	yaw = np.round(math.degrees(math.atan2(-r31, r11)), 4)
	# 	roll = np.round(math.degrees(math.asin(r21)), 4)
	# 	pitch = np.round(math.degrees(math.atan2(-r23, r22)), 4)

		# yaw = np.round(math.degrees(math.atan2(r21, r11)), 4)
		# roll = np.round(math.degrees(math.atan2(r32, r33)), 4)
		# pitch = np.round(math.degrees(math.atan2(-r31, np.sqrt(r11**2 + r21**2))), 4)
# ###########################################################################################33
	# ar = 3.
	# ap = 0.
	# ay = 1.
	# print 'Pitch Roll Yaw *'
	results = []
	p1, r1, y1 = PRY(R)
	e1 = error(ar, r1, ap, p1, ay, y1)
	# print p1, r1, y1, e1
	results.append([r1, p1, y1, e1, 'PRY'])

	# print 'Pitch Yaw Roll'
	# p2, r2, y2 = PYR(R)
	# e2 = error(ar, r2, ap, p2, ay, y2)
	# # print p2, r2, y2, e2
	# results.append([r2, p2, y2, e2, 'PYR'])

	# print 'Yaw Pitch Roll'
	# p3, r3, y3 = YPR(R)
	# e3 = error(ar, r3, ap, p3, ay, y3)
	# # print p3, r3, y3, e3
	# results.append([r3, p3, y3, e3, 'YPR'])

	# # print 'Yaw Roll Pitch'
	# p4, r4, y4 = YPR(R)
	# e4 = error(ar, r4, ap, p4, ay, y4)
	# # print p4, r4, y4, e4
	# results.append([r4, p4, y4, e4, 'YRP'])

	# print 'Roll Yaw Pitch'
	p5, r5, y5 = RYP(R)
	e5 = error(ar, r5, ap, p5, ay, y5)
	# print p5, r5, y5, e5
	results.append([r5, p5, y5, e5, 'RYP'])

	# print 'Roll Pitch Yaw'
	p6, r6, y6 = RPY(R)
	e6 = error(ar, r6, ap, p6, ay, y6)
	# print p6, r6, y6, e6
	results.append([r6, p6, y6, e6, 'RPY'])

	results = sorted(results, key = lambda results: results[3], reverse = True)
	##################################################################
	# print 'Euler yaw =', yaw, ' pitch = ', pitch, ' roll = ', roll
	return results[-1]
	# return roll, pitch, yaw

def findBest():
	imgs = glob.glob('attitudeTracking/small/l*.jpeg')
	values = np.array([[-146., 160., -18.], 
			[-146., 162., -18.],
			[-143., 162., -18.],
			[-143., 162., -19.],
			[-143., 162., -19.],
			[-143., 162., -19.]])
	M_camL = np.array([[  1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
       [  0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	results = []
	for img1Name, val1 in zip(imgs, values):
		img1 = cv2.imread(img1Name, 0)

		for img2Name, val2 in zip(imgs, values):
			img2 = cv2.imread(img2Name, 0)
			ar, ap, ay = val1 - val2
			H, src_pts, dst_pts = featureMatching(img1, img2, 'matches.png')
			results.append(detAttitude(H, M_camL, ar, ap, ay))

	r_arr = np.array(results)
	names = r_arr[:,-1]
	counts = Counter(names.tolist())
	print counts

	df = pandas.DataFrame.from_dict(counts, orient = 'index')
	df.plot(kind = 'bar')
	plt.savefig('rotation_error.png')
	plt.show()

	return results

def main():
	img1 = cv2.imread('attitudeTracking/small/l1.jpeg', 0)
	img2 = cv2.imread('attitudeTracking/small/l2.jpeg', 0)
	H_main, src_pts, dst_pts = featureMatching(img1, img2, 'matches.png')

	img1R = cv2.imread('attitudeTracking/multipleAngles/r2.jpeg',0) # queryImage
	img2R = cv2.imread('attitudeTracking/multipleAngles/r3.jpeg',0) # trainImage
	HR, src_pts_r, dst_pts_r = featureMatching(img1R, img2R, 'matchesR.jpeg')

	img1L = cv2.imread('attitudeTracking/multipleAngles/l1.jpeg',0) # queryImage
	img2L = cv2.imread('attitudeTracking/multipleAngles/l2.jpeg',0) # trainImage

	# compute the feature matching, calculate the homography matrix
	HL, src_pts_l, dst_pts_l = featureMatching(img1L, img2L, 'matchesL.jpeg')


	# M = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
    #   [  0.00000000e+00,   1.99299316e+02,   5.52500000e+02],
    #    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	# K = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
	#        [  0.00000000e+00,   1.70664233e+03,   5.52500000e+02],
	#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	M_camL = np.array([[  1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
       [  0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	M_camR = np.array([[  1.89292737e+03,   0.00000000e+00,   9.17130175e+02],
       [  0.00000000e+00,   2.09561499e+03,   5.37271880e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	# OpenCV Homography decomposition method
	print 'OpenCv Homography Decomposition'
	print 'Left'
	rl, pl, yl = detAttitude(HL, M_camL)
	print 'Right'
	rr, pr, yr = detAttitude(HR, M_camR)

	print 'Angle averages:'
	print 'roll = ',np.round(np.mean([rr, rl]), 4), \
	' pitch = ',np.round(np.mean([pr, pl]), 4), \
	' yaw = ', np.round(np.mean([yr, yl]), 4)

