import cv2
import numpy as np

feed = cv2.imread('Gokarna.jpg')
gray = cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)

surf=cv2.xfeatures2d.SURF_create()

surf.setHessianThreshold(4000)

keypoints,descriptors=surf.detectAndCompute(gray,None)

cv2.drawKeypoints(feed,keypoints,feed,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SURF features',feed)
cv2.imwrite('SURF features.jpg',feed)
cv2.waitKey()
