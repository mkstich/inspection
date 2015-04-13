import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from stereo_calibrate import *
from cannyEllipse import *

def determinePoints(stereoCams, ptsL, ptsR, dL, dR):

	coordsL, coordsR = [], []
	# Find Left Image World Coordinates
	for e_1 in ptsL:
		x, y = round(e_1.x), round(e_1.y)

		if (x >= 0. and y >= 0.):
			d = dR[y - 1][x - 1]

			# homogeneous point vector
			x_vec = np.array([[e_1.x], [e_1.y], [d], [1.]])
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
	title = folder + ': ' + str(len(pred)) + ' Matches'
	plt.title(title)

	plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
	plt.savefig(folder + 'scatterPlot.png')

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
	print max(dist, key = lambda dist: dist[0])
	limit = max(dist, key = lambda dist: dist[0])[0] / 2.5

	if limit > 250.:
		limit = 250.
		
	d2 = [i for i in dist if i[0] <= limit]
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


def calculate3DCloud(ptsL, ptsR, stereoCams, dL, dR, folder):
	# determine 3D points
	coordsL, coordsR = determinePoints(stereoCams, ptsL, ptsR, dL, dR)
	cL, cR = calcDistances(coordsL, coordsR)
	
	# predict actual 3D points
	predPoints = computeMeanDist(cL, cR)

	# plot results
	createScatter(cL, cR, predPoints, folder)

def findEllipses(stereoCams, folder, imsize):
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

def detAttitude(srcImg, ptsSrc, destImg, folder):
	''' 
	srcImg = The actual image
	destImg = The reference image...what the image
	is supposed to look like 
	'''
	folder = ''
	srcImg = 'rotated.png'
	destImg = 'rectify_imgL.png'

	name = imageName(folder, 'handlesDest.png')
	destName = imageName(folder, 'filtered_ellipse_dest.png')

	nameS = imageName(folder, 'handlesSrc.png')
	srcName = imageName(folder, 'filtered_ellipse_src.png')

	# read in images
	src = cv2.imread(srcImg)
	dest = cv2.imread(destImg)

	# Locate points in the destination image
	handlesDest = find_handles(dest)
	pltplot(handlesDest, name)
	hDest = cv2.imread(name)

	######################################3
	handlesSrc = find_handles(src)
	pltplot(handlesSrc, nameS)
	hSrc = cv2.imread(nameS)

	eh, ctsH = compute_threshold(handlesDest, hDest.copy(), destName)
	es, ctsS = compute_threshold(handlesSrc, hSrc.copy(), srcName)

	ptsSrc, ptsDest = [], []
	for i in eh:
		ptsDest.append(np.array((i.x, i.y)))
	for i in es:
		ptsSrc.append(np.array((i.x, i.y)))

	# make sure there is the same number of dest and src pts
	limit = min(len(ptsDest), len(ptsSrc))
	ptsDest = ptsDest[: limit]
	ptsSrc = ptsSrc[: limit]

	H, mask = cv2.findHomography(np.array(ptsSrc), np.array(ptsDest), cv2.RANSAC, 5.0)
	m = np.asarray(H)
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = m.flat

	x1 = -math.asin(r31)
	x2 = math.pi + math.asin(r31)
	y1 = math.atan2(r32 / math.cos(x1), r33 / math.cos(x1)) * 180./math.pi
	y2 = math.atan2(r32 / math.cos(x2), r33 / math.cos(x2))* 180./math.pi
	z1 = math.atan2(r21 / math.cos(x1), r11 / math.cos(x1))* 180./math.pi
	z2 = math.atan2(r21 / math.cos(x2), r11 / math.cos(x2))* 180./math.pi

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

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
    cv2.imshow('Matched Features', out)
    cv2.imwrite('matches.png', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def featureMatching():
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread('attitudeTracking/l1.jpeg',0)          # queryImage
	img2 = cv2.imread('attitudeTracking/l2.jpeg',0) # trainImage

	# Initiate SIFT detector
	sift = cv2.SIFT()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

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

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		# h, w = img1.shape
		# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		# dst = cv2.perspectiveTransform(pts,M)

		# img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

	drawMatches(img1, kp1, img2, kp2, good[:100])
	return M, src_pts, dst_pts

M = array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
       [  0.00000000e+00,   1.99299316e+02,   5.52500000e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
# K = array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
#        [  0.00000000e+00,   1.70664233e+03,   5.52500000e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
# M_camL = array([[  1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
#        [  0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
def detAttitude(H, M):
	''' OpenCV Textbook method '''
	# U, S, Vt = la.svd(HL)
	# sig2 = sorted(S)[1]

	# H = HL / sig2
	h1 = np.array([[H[0][0]], [H[1][0]], [H[2][0]]])
	h2 = np.array([[H[0][1]], [H[1][1]], [H[2][1]]])
	h3 = np.array([[H[0][2]], [H[1][2]], [H[2][2]]])
	lda = 1. / la.norm(h1)#
	# lda = 1. / np.linalg.norm(np.dot(np.linalg.inv(M), h1))
	r1 = np.dot(lda, np.dot(np.linalg.inv(M), h1))
	r2 = np.dot(lda, np.dot(np.linalg.inv(M), h2))
	r3 = np.cross(r1.ravel(), r2.ravel())
	t = np.dot(lda, np.dot(np.linalg.inv(M), h3))

	r3 = np.array([[r3[0]], [r3[1]], [r3[2]]])

	r2[0] = -r2[0]
	r3[0] = -r3[0]
	r1[1], r1[2] = -r1[1], -r1[2]
	R = np.hstack([r1, r2, r3])

	U, S, Vt = np.linalg.svd(R)

	if round(la.det(U), 4) == -1.:
		U = -U
		Vt = -Vt

	D = np.identity(len(S))

	R2 = np.dot(U, np.dot(D, Vt))

	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R2.flat

	x1 = -math.asin(r31)
	x2 = math.pi + math.asin(r31)
	y1 = math.atan2(r32 / math.cos(x1), r33 / math.cos(x1))
	y2 = math.atan2(r32 / math.cos(x2), r33 / math.cos(x2))
	z1 = math.atan2(r21 / math.cos(x1), r11 / math.cos(x1))
	z2 = math.atan2(r21 / math.cos(x2), r11 / math.cos(x2))

	print 'option 1 = ', math.degrees(x1), math.degrees(y1), math.degrees(z1)
	print 'option 2 = ', math.degrees(x2), math.degrees(y2), math.degrees(z2)
	print 't = ', t

def newAttitude(HL):
	'''UCLA Method (Ch. 5)'''

	U, S, Vt = la.svd(HL)
	sig2 = sorted(S)[1]

	H = HL / sig2
	V, S, Vt = la.svd(np.dot(H.T, H))

	if(la.det(V) == -1.):
		V = -V
		Vt = -Vt

	sig1_sq, sig3_sq = S[0], S[2]
	v1 = np.array([[V[0][0]], [V[1][0]], [V[2][0]]])
	v2 = np.array([[V[0][1]], [V[1][1]], [V[2][1]]])
	v3 = np.array([[V[0][2]], [V[1][2]], [V[2][2]]])

	u1_n = np.dot(v1, np.sqrt(1. - sig3_sq)) + np.dot(v3, np.sqrt(sig1_sq - 1.))
	u1_d = np.sqrt(sig1_sq - sig3_sq)
	u1 = u1_n / u1_d

	u2_n = np.dot(v1, np.sqrt(1. - sig3_sq)) - np.dot(v3, np.sqrt(sig1_sq - 1.))
	u2_d = u1_d
	u2 = u2_n / u2_d

	v2_hat = v2 / la.norm(v2)
	new_u13 = np.cross(v2_hat.ravel(), u1.ravel())
	new_u23 = np.cross(v2_hat.ravel(), u2.ravel())

	U1 = np.hstack([v2, u1, np.array([[new_u13[0]], [new_u13[1]], [new_u13[2]]])])
	U2 = np.hstack([v2, u2, np.array([[new_u23[0]], [new_u23[1]], [new_u23[2]]])])

	new_W13 = np.cross(np.dot(H, v2_hat).ravel(), np.dot(H, u1).ravel())
	new_W23 = np.cross(np.dot(H, v2_hat).ravel(), np.dot(H, u2).ravel())
	
	W1 = np.hstack([np.dot(H, v2), np.dot(H, u1), \
		np.array([[new_W13[0]], [new_W13[1]], [new_W13[2]]])])
	W2 = np.hstack([np.dot(H, v2), np.dot(H, u2), \
		np.array([[new_W23[0]], [new_W23[1]], [new_W23[2]]])])

	R1 = np.dot(W1, U1.T)
	R2 = np.dot(W2, U2.T)

	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R1.flat

	x11 = -math.asin(r31)
	x12 = math.pi + math.asin(r31)
	y11 = math.atan2(r32 / math.cos(x11), r33 / math.cos(x11))
	y12 = math.atan2(r32 / math.cos(x12), r33 / math.cos(x12))
	z11 = math.atan2(r21 / math.cos(x11), r11 / math.cos(x11))
	z12 = math.atan2(r21 / math.cos(x12), r11 / math.cos(x12))
	print 'R1 = ', R1
	print 'option 1 = ', math.degrees(x11), math.degrees(y11), math.degrees(z11)
	print 'option 2 = ', math.degrees(x12), math.degrees(y12), math.degrees(z12)

	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R2.flat

	x21 = -math.asin(r31)
	x22 = math.pi + math.asin(r31)
	y21 = math.atan2(r32 / math.cos(x21), r33 / math.cos(x21))
	y22 = math.atan2(r32 / math.cos(x22), r33 / math.cos(x22))
	z21 = math.atan2(r21 / math.cos(x21), r11 / math.cos(x21))
	z22 = math.atan2(r21 / math.cos(x22), r11 / math.cos(x22))
	print 'R2 = ', R2
	print 'option 1 = ', math.degrees(x21), math.degrees(y21), math.degrees(z21)
	print 'option 2 = ', math.degrees(x22), math.degrees(y22), math.degrees(z22)


def paperAttitude(H):
	'''method from paper'''
	V, S, Vt = la.svd(np.dot(H.T, H))
	v1 = np.array([[V[0][0]], [V[1][0]], [V[2][0]]])
	v2 = np.array([[V[0][1]], [V[1][1]], [V[2][1]]])
	v3 = np.array([[V[0][2]], [V[1][2]], [V[2][2]]])
	sig1, sig3 = np.sqrt(S[0]), np.sqrt(S[2])

	lda1 = (1. / (2. * sig1 * sig3)) * (-1. + np.sqrt(1. + 4.*(sig1 * sig3) / ((sig1 - sig3)**2)))
	lda3 = (1. / (2. * sig1 * sig3)) * (-1. - np.sqrt(1. + 4.*(sig1 * sig3) / ((sig1 - sig3)**2)))

	v1_norm = np.sqrt(lda1 * lda1 * (sig1 - sig3)**2 + 2 * lda1 * (sig1 * sig3 - 1.) + 1.)
	v3_norm = np.sqrt(lda3 * lda3 * (sig1 - sig3)**2 + 2 * lda3 * (sig1 * sig3 - 1.) + 1.)

	v1_p = np.dot(v1_norm, v1)
	v3_p = np.dot(v3_norm, v3)

	t_star = (v1_p + v3_p) / (lda1 - lda3)
	n = (np.dot(lda1, v3_p) + np.dot(lda3, v1_p)) / (lda1 - lda3)

	R = np.dot(H, la.inv(np.identity(3) + np.dot(t_star, n.T)))

	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat

	x1 = -math.asin(r31)
	x2 = math.pi + math.asin(r31)
	y1 = math.atan2(r32 / math.cos(x1), r33 / math.cos(x1))
	y2 = math.atan2(r32 / math.cos(x2), r33 / math.cos(x2))
	z1 = math.atan2(r21 / math.cos(x1), r11 / math.cos(x1))
	z2 = math.atan2(r21 / math.cos(x2), r11 / math.cos(x2))

	print 'option 1 = ', math.degrees(x1), math.degrees(y1), math.degrees(z1)
	print 'option 2 = ', math.degrees(x2), math.degrees(y2), math.degrees(z2)