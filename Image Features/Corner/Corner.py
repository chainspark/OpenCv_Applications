import cv2
import numpy as np

corner = cv2.imread('Corner4.jpg')
raw_gray =cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)

raw_corner =cv2.cornerHarris(raw_gray,blockSize=4,ksize=5,k=0.04)

raw_corner=cv2.dilate(raw_corner, None)

corner[raw_corner > 0.01*raw_corner.max()] =[0,0,0]

cv2.imshow('Harris Corners(sharp)',corner)

raw_corner_soft =cv2.cornerHarris(raw_gray,blockSize=14,ksize=5,k=0.04)

raw_corner_soft=cv2.dilate(raw_corner_soft, None)

corner[raw_corner_soft > 0.01*raw_corner_soft.max()] =[0,0,0]
cv2.imshow('Harris Corners(soft)',corner)

cv2.waitKey()


