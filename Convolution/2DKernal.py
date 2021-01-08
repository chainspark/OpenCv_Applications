import cv2
import numpy as np

starry=cv2.imread('starry_night.jpg')

kernel_identity=np.array([[0,0,0],[0,1,0],[0,0,0]])

kernel3x3=np.ones((3,3))/9.0

kernel5x5=np.ones((5,5))/25.0

blur3x3 = cv2.filter2D(starry,-1,kernel3x3)

blur5x5 = cv2.filter2D(starry,-1,kernel5x5)

#cv2.imshow('3x3 Blur',blur3x3)
#cv2.imshow('5x5 Blue',blur5x5)
cv2.imshow('Original',starry)

#Motion Blur (Vertical)

size =20  
Vblur = np.zeros((size,size))
Vblur[:,int((size-1)/2)] = np.ones(size)
Vblur = Vblur/size

blur_vertical = cv2.filter2D(starry,-1,Vblur)

#cv2.imshow('Vertical Blur',blur_vertical)


#Sharpening 

sharpen1 =np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

sharpen2 =np.array([[1,1,1],[1,-7,1],[1,1,1]])

sharpen3 =np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0


sharp_out1=cv2.filter2D(starry,-1,sharpen1)

sharp_out2=cv2.filter2D(starry,-1,sharpen2)

sharp_out3=cv2.filter2D(starry,-1,sharpen3)

#cv2.imshow('Sharpen1',sharp_out1)
#cv2.imshow('Sharpen2',sharp_out2)
#cv2.imshow('Sharpen3',sharp_out3)

#Embossing

starry_gray=cv2.cvtColor(starry,cv2.COLOR_BGR2GRAY)

emboss1 =np.array([[0,-1,-1],[1,0,-1],[1,1,0]])

emboss2 =np.array([[-1,-1,0],[-1,0,1],[0,-1,-1]])

emboss3 =np.array([[1,0,0],[0,0,0],[0,0,-1]])

emboss_out1=cv2.filter2D(starry_gray,-1,emboss1)

emboss_out2=cv2.filter2D(starry_gray,-1,emboss2)

emboss_out3=cv2.filter2D(starry_gray,-1,emboss3)

#cv2.imshow('Emboss1',emboss_out1)
#cv2.imshow('Emboss2',emboss_out2)
#cv2.imshow('Emboss3',emboss_out3)

#Edge Detection-Sobel,Laplacian,Canny

star_edge_sobel=cv2.Sobel(starry_gray,cv2.CV_64F,1,1,ksize=3)

star_edge_laplacian=cv2.Laplacian(starry_gray,cv2.CV_64F)

star_edge_canny =cv2.Canny(starry_gray,230,250)

cv2.imshow('Original Gray',starry_gray)
#cv2.imshow('Edge Deteection',star_edge_sobel)
#cv2.imshow('Edge Laplacian',star_edge_laplacian)
#cv2.imshow('Edge Canny',star_edge_canny)

#Erosion and Dilation-Binary and gray images

k5=np.ones((5,5))
eroded_night=cv2.erode(starry_gray,k5,iterations=1 )
dilated_night=cv2.dilate(starry_gray,k5,iterations=2)

cv2.imshow('Erode',eroded_night)
cv2.imshow('Dilate',dilated_night)

cv2.waitKey()

cv2.destroyAllwindows()

