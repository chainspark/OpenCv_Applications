#Better than connerHarris

import cv2
import numpy as np

corner = cv2.imread('Corner3.jpg')

corner =cv2.resize(corner,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
gray =cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)

good_corner = cv2.goodFeaturesToTrack(gray,maxCorners=7,qualityLevel=.05,minDistance=35)

for item in good_corner:
	x, y =item[0]
	cv2.circle(corner,(x,y),5,255,-1)

cv2.imshow("Top 'k' features", corner)
cv2.waitKey()
