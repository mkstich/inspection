import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import math
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D

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

def featureMatching(img1, img2):
	MIN_MATCH_COUNT = 10

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

		# calculate the homography matrix
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

	# plot the first 100 matches 
	drawMatches(img1, kp1, img2, kp2, good[:100])

	return M, src_pts, dst_pts

def detAttitude(H, M):
	''' Gary Bradski's OpenCV Textbook method 
	http://www.cs.haifa.ac.il/~dkeren/ip/OReilly-LearningOpenCV.pdf
	Used formulas from Ch. 11 pg. 389 'What's under the hood?'

	H = Homography matrix
	M = camera matrix
	'''

	h1 = np.array([[H[0][0]], [H[1][0]], [H[2][0]]])
	h2 = np.array([[H[0][1]], [H[1][1]], [H[2][1]]])
	h3 = np.array([[H[0][2]], [H[1][2]], [H[2][2]]])
	
	# calculate the scaling factor, lda
	b11, b12, b13, b12, b22, b23, b13, b23, b33 = np.dot(la.inv(M.T), la.inv(M)).flat
	cy = (b12 * b13 - b11 * b23) / (b11*b22 - b12**2)
	lda = b33 - (b13**2 + cy * (b12 * b13 - b11 * b23)) / b11

	# compute the components of the rotation matrix
	r1 = np.dot(lda, np.dot(np.linalg.inv(M), h1))
	r2 = np.dot(lda, np.dot(np.linalg.inv(M), h2))
	r3 = np.cross(r1.ravel(), r2.ravel())

	r3 = np.array([[r3[0]], [r3[1]], [r3[2]]])

	R = np.hstack([r1, r2, r3])

	# translation vector
	t = np.dot(lda, np.dot(np.linalg.inv(M), h3))

	# U, S, Vt = np.linalg.svd(R)

	# # if round(la.det(U), 4) == -1.:
	# # 	U = -U
	# # 	Vt = -Vt

	# D = np.identity(len(S))

	# R2 = np.dot(U, np.dot(D, Vt))
	r, p, y = calcAngles(R)



def newAttitude(HL):
	'''UCLA Method (Ch. 5)
	http://vision.ucla.edu//MASKS/MASKS-ch5.pdf
	Used Ch.5 pg. 131 - 138
	'''

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

	calcAngles(R1)
	calcAngles(R2)



def paperAttitude(H):
	'''Method from 'Deeper Understanding of the homography decomposition
	for vision based control' 
	Used the Zhang SVD - Based composition method on pg. 9
	https://hal.archives-ouvertes.fr/file/index/docid/174739/filename/RR-6303.pdf
	'''
	V, S, Vt = la.svd(np.dot(H.T, H))
	v1 = np.array([[V[0][0]], [V[1][0]], [V[2][0]]])
	v2 = np.array([[V[0][1]], [V[1][1]], [V[2][1]]])
	v3 = np.array([[V[0][2]], [V[1][2]], [V[2][2]]])
	sig1, sig3 = np.sqrt(S[0]), np.sqrt(S[2])

	lda1 = (1. / (2. * sig1 * sig3)) * (-1. + np.sqrt(1. + 4.*(sig1 * sig3) / ((sig1 - sig3)**2)))
	lda3 = (1. / (2. * sig1 * sig3)) * (-1. - np.sqrt(1. + 4.*(sig1 * sig3) / ((sig1 - sig3)**2)))

	v1_norm = np.sqrt(lda1 * lda1 * ((sig1 - sig3)**2) + 2 * lda1 * (sig1 * sig3 - 1.) + 1.)
	v3_norm = np.sqrt(lda3 * lda3 * ((sig1 - sig3)**2) + 2 * lda3 * (sig1 * sig3 - 1.) + 1.)

	v1_p = np.dot(v1_norm, v1)
	v3_p = np.dot(v3_norm, v3)

	t_star = (v1_p + v3_p) / (lda1 - lda3)
	n = (np.dot(lda1, v3_p) + np.dot(lda3, v1_p)) / (lda1 - lda3)

	R = np.dot(H, la.inv(np.identity(3) + np.dot(t_star, n.T)))

	calcAngles(R)


def calcAngles(R):
	'''
	heading = theta = yaw (about y)
	bank = psi = pitch (about x)
	attitude = phi = roll (about z)
	'''
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = R.flat

	# heading = math.atan2(-r31, r11)
	# bank = math.atan2(-r23, r22)
	# attitude = math.asin(r21)

	# heading = math.asin(-r31)
	# cy = math.cos(heading)

	# if cy > 0.:
	# 	attitude = math.atan2(r21, r11) #(r11, r21)
	# 	bank = math.atan2(r32, r33) #(r33, r32)
	# else:
	# 	attitude = math.atan2(-r11, -r21)
	# 	bank = math.atan2(-r33, -r32)

	# yaw = np.round(math.degrees(heading), 5)
	# pitch = np.round(math.degrees(bank), 5)
	# roll = np.round(math.degrees(attitude), 5)
	# print 'r = ', roll, 'p = ', pitch, 'y = ', yaw

	'''Order of Euler Angles: heading, attitude, bank'''
	# if singularity at the north pole
	if r21 > 0.998:
		yaw = np.round(math.degrees(math.atan2(r13, r33)), 4)
		roll = np.round(math.degrees(math.pi / 2.), 4)
		pitch = 0.0

	# if sinularity at south pole
	elif r21 < -0.998:
		yaw = np.round(math.degrees(math.atan2(r13, r33)), 4)
		roll = np.round(math.degrees(math.pi / 2.), 4)
		bank = 0.

	else:
		yaw = np.round(math.degrees(math.atan2(-r31, r11)), 4)
		roll = np.round(math.degrees(math.asin(r21)), 4)
		pitch = np.round(math.degrees(math.atan2(-r23, r22)), 4)

	print 'Euler yaw =', yaw, ' pitch = ', pitch, ' roll = ', roll
	return roll, pitch, yaw


def firstMethod(H):
	# Get a symmetric matrix
	S = np.dot(H.T, H) - np.identity(3)
	s11, s12, s13, s12, s22, s23, s13, s23, s33 = S.flat

	Ms11 = la.det(np.array([[s22, s23], [s23, s33]]))
	Ms11 *= np.sign(Ms11)
	Ms22 = la.det(np.array([[s11, s13], [s13, s33]]))
	Ms22 *= np.sign(Ms22)
	Ms33 = la.det(np.array([[s11, s12], [s12, s22]]))
	Ms33 *= np.sign(Ms33)

	Ms12 = - la.det(np.array([[s12, s13], [s23, s33]]))
	Ms13 = - la.det(np.array([[s12, s13], [s22, s23]]))
	Ms23 = - la.det(np.array([[s11, s13], [s12, s23]]))
	Ms21 = Ms12
	Ms31 = Ms13
	Ms32 = Ms23

	naP_s11 = np.array([[s11], [s12 + np.sqrt(Ms33)], [s13 + np.sign(Ms23) * np.sqrt(Ms22)]])
	naP_s22 = np.array([[s12 + np.sqrt(Ms33)], [s22], [s23 - np.sign(Ms13) * np.sqrt(Ms11)]])
	naP_s33 = np.array([[s13 + np.sign(Ms12) * np.sqrt(Ms22)], [s23 + np.sqrt(Ms11)], [s33]])

	nbP_s11 = np.array([[s11], [s12 - np.sqrt(Ms33)], [s13 - np.sign(Ms23) * np.sqrt(Ms22)]])
	nbP_s22 = np.array([[s12 - np.sqrt(Ms33)], [s22], [s23 + np.sign(Ms13) * np.sqrt(Ms11)]])
	nbP_s33 = np.array([[s13 - np.sign(Ms12) * np.sqrt(Ms22)], [s23 - np.sqrt(Ms11)], [s33]])

	# calculate the normals
	na_s11 = naP_s11 / la.norm(naP_s11)
	na_s22 = naP_s22 / la.norm(naP_s22)
	na_s33 = naP_s33 / la.norm(naP_s33)

	nb_s11 = nbP_s11 / la.norm(nbP_s11)
	nb_s22 = nbP_s22 / la.norm(nbP_s22)
	nb_s33 = nbP_s33 / la.norm(nbP_s33)

	nu = 2. * np.sqrt(1 + np.trace(S) - Ms11 - Ms22 - Ms33)
	rho = np.sqrt(2 + np.trace(S) + nu)
	te_norm = np.sqrt(2 + np.trace(S) - nu)

	# calculate translation vectors
	ta_s11 = 0.5 * te_norm * (np.sign(s11) * np.dot(rho, nb_s11) - np.dot(te_norm, na_s11))
	ta_s22 = 0.5 * te_norm * (np.sign(s22) * np.dot(rho, nb_s22) - np.dot(te_norm, na_s22))
	ta_s33 = 0.5 * te_norm * (np.sign(s33) * np.dot(rho, nb_s33) - np.dot(te_norm, na_s33))

	tb_s11 = 0.5 * te_norm * (np.sign(s11) * np.dot(rho, na_s11) - np.dot(te_norm, nb_s11))
	tb_s22 = 0.5 * te_norm * (np.sign(s22) * np.dot(rho, na_s22) - np.dot(te_norm, nb_s22))
	tb_s33 = 0.5 * te_norm * (np.sign(s33) * np.dot(rho, na_s33) - np.dot(te_norm, nb_s33))

	Ra_s11 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(ta_s11, na_s11.T)))
	Ra_s22 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(ta_s22, na_s22.T)))
	Ra_s33 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(ta_s33, na_s33.T)))

	Rb_s11 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(tb_s11, nb_s11.T)))
	Rb_s22 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(tb_s22, nb_s22.T)))
	Rb_s33 = np.dot(H, (np.identity(3) - (2. / rho) * np.dot(tb_s33, nb_s33.T)))

def main():
	# img1 was manually placed to have a Pitch, Yaw, Roll of (158, -25., -145.) degrees
	# img2 has a Pitch, Yaw, Roll of (158. -20., -145.) degrees
	# both images have the same X, Y, Z coordinates of (384, 277, 360) inches
	# thus the only difference between the points in img1 to img2 should be
	# 5 degrees in the yaw direction

	img1 = cv2.imread('attitudeTracking/multipleChanges/l1.jpeg',0) # queryImage
	img2 = cv2.imread('attitudeTracking/multipleChanges/l2.jpeg',0) # trainImage

	# compute the feature matching, calculate the homography matrix
	H, src_pts, dst_pts = featureMatching(img1, img2)

	# camera matrices attained through an external calibration script
	# M_camL was attained through the calibration script but it only has 
	# semi-decent performance, so I was experiementing with modified 
	# camera matrices M and K. These get better results but since they were 
	# manually altered, there performance is not universal to all images

	# M = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
    #   [  0.00000000e+00,   1.99299316e+02,   5.52500000e+02],
    #    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	# K = np.array([[  1.78586597e+03,   0.00000000e+00,   9.60000000e+02],
	#        [  0.00000000e+00,   1.70664233e+03,   5.52500000e+02],
	#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	M_camL = np.array([[  1.78586597e+03,   0.00000000e+00,   9.15499196e+02],
       [  0.00000000e+00,   1.70664233e+03,   5.45250506e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	# OpenCV Homography decomposition method
	print 'OpenCv Homography Decomposition'
	detAttitude(H, M_camL)
	print '\n'
	# UCLA Method
	print 'UCLA Method'
	newAttitude(H)
	print '\n'
	print 'Paper on Homography Decomposition'
	# Paper method
	paperAttitude(H)

	# again the calculated angles should have degree magnitudes about equal to 0 pitch, 0 roll, 5 yaw