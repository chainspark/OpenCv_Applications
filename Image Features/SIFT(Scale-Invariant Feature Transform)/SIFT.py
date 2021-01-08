import cv2
import numpy as np

feed = cv2.imread('Gokarna.jpg')
gray = cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)

sift =cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray,None)

cv2.drawKeypoints(feed,keypoints,feed,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT features',feed)
cv2.imwrite('SIFT festures.jpg',feed)
cv2.waitKey()
