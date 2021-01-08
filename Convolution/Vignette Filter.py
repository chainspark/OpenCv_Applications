import cv2
import numpy as np

boat = cv2.imread('FishingBoat.jpg')

rows, cols =boat.shape[:2]

#Vignette Filter 

#Kernal Mask
kernel_x=cv2.getGaussianKernel(int(1.4*cols),400)
kernel_y=cv2.getGaussianKernel(int(1.4*rows),250)
kernel = kernel_y*kernel_x.T
mask = 255*kernel / np.linalg.norm(kernel)
mask=mask[int(0*rows):int(1.0*rows),int(0.4*cols):]

output_vignette =np.copy(boat)

#Appyling mask to all channels of image
for i in range(3):
	output_vignette[:,:,i] = output_vignette[:,:,i]*mask
cv2.imshow('Original',boat)
cv2.imshow('Vignette',output_vignette)

cv2.imwrite('Focus_on_boat.jpg',output_vignette)

#Contrast equilizer-Grayscale

output_vignette_gray=cv2.cvtColor(output_vignette,cv2.COLOR_BGR2GRAY)

histeq = cv2.equalizeHist(output_vignette_gray)

cv2.imshow('Gray-vignette',output_vignette_gray)
cv2.imshow('Vignette Equalised',histeq)


cv2.imwrite('Focus_on_boat_equalised_gray.jpg',histeq )

#Contrast equilizer - YUV

boat_yuv=cv2.cvtColor(boat,cv2.COLOR_BGR2YUV)

#Equlaize intensitu=y value
boat_yuv[:,:,0] = cv2.equalizeHist(boat_yuv[:,:,0])

output_equalized_boat =cv2.cvtColor(boat_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Equalized Vignette Boat',output_equalized_boat)
cv2.imwrite('Equalized_Vignette_Boat.jpg',output_equalized_boat)

cv2.waitKey(0)

cv2.destroyAllwindows()
