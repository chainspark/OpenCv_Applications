import cv2
import numpy as np

feed = cv2.imread('Gokarna.jpg')
gray=cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()

keypoints = fast.detect(gray,None)
print("Number of keypoints with non max suppression:", len(keypoints))

keypoints_image_nonmax=feed.copy()

cv2.drawKeypoints(feed,keypoints,keypoints_image_nonmax,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST keypoiny- with non max supression',keypoints_image_nonmax)

fast.setNonmaxSuppression(False)


keypoints = fast.detect(gray,None)
print("Number of keypoints with non max suppression:", len(keypoints))

keypoints_image_without_nonmax=feed.copy()

cv2.drawKeypoints(feed,keypoints,keypoints_image_without_nonmax,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST keypoiny- without non max supression',keypoints_image_without_nonmax)

cv2.waitKey()
cv2.imwrite('FAST keypoiny- with non max supression.jpg',keypoints_image_nonmax)

cv2.imwrite('FAST keypoiny- without non max supression.jpg',keypoints_image_without_nonmax)



