import cv2
import numpy as np

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

############################# testing part  #########################

im = cv2.imread('skel_contour230.png')#'pi.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)
test_cnts = cnts[0 : int(len(cnts) * .04)]
rects = []

for cnt in test_cnts:
    # if cv2.contourArea(cnt) > 3000:# and cv2.contourArea(cnt) < 500: # > 50
    [x,y,w,h] = cv2.boundingRect(cnt)
    rects.append(np.array([x, y, w, h, np.float32(w) / h]))
    if  h>28:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        roi = thresh[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
        roismall = roismall.reshape((1,100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
        string = str(int((results[0][0])))
        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

r = sorted(rects, key = lambda rects: rects[-1], reverse = False)
x1, y1, w1, h1, _ = r[0]
x2, y2, w2, h2, _ = r[-1]

cv2.rectangle(im,(int(x1),int(y1)),(int(x1+w1),int(y1+h1)),(0,255,0),2)
cv2.rectangle(im,(int(x2),int(y2)),(int(x2+w2),int(y2+h2)),(0,255,0),2)

cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)