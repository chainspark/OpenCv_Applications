import cv2
import numpy as np

feed = cv2.imread('Gokarna.jpg')
gray = cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)

#initialteORB
orb =cv2.ORB_create()

keypoints = orb.detect(gray,None)

keypoints,descriptors = orb.compute(gray,keypoints)

cv2.drawKeypoints(feed,keypoints,feed,color=(0,255,0))

cv2.imshow('ORB keypoints',feed)
cv2.imwrite('ORB keypoints.jpg',feed)

cv2.waitKey()
