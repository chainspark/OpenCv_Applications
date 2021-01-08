import cv2
import numpy as np

feed=cv2.imread('Gokarna.jpg')
gray=cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)

#initiate fast
fast=cv2.FastFeatureDetector_create()

#initiate BRIEF
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints=fast.detect(gray,None)

keypoints,descriptors=brief.compute(gray,keypoints)

cv2.drawKeypoints(feed,keypoints,feed,color=(0,255,0))

cv2.imshow('BRIEF keypoints',feed)
cv2.imwrite('BRIEF keypoints.jpg',feed)
cv2.waitKey()





