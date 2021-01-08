import cv2
import numpy as np

def compute_energy(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)

	sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

	abs_sobel_x = cv2.convertScaleAbs(sobel_x)
	abs_sobel_y = cv2.convertScaleAbs(sobel_y)

	energy=cv2.addWeighted(abs_sobel_x,0.5,abs_sobel_y,0.5,0)

	return energy

def find_vertical_seam(img,energy):

	rows, cols = img.shape[:2]

	#Empty seam vector along vertical direction
	seam = np.zeros(img.shape[0])

	dist_to = np.zeros(img.shape[:2])+float('inf')
	dist_to[0,:] = np.zeros(img.shape[1])
	edge_to = np.zeros(img.shape[:2])

	#Compute paths of lowest energy

	for row in range(rows-1):
		for col in range(cols):
			if col !=0 and dist_to[row+1,col-1] > dist_to[row, col]+energy[row+1,col-1]:
				dist_to[row+1,col-1] = dist_to[row, col]+energy[row+1,col-1]
				edge_to[row+1,col-1]=1

			if dist_to[row+1,col] >dist_to[row,col]+energy[row+1,col]:
				dist_to[row+1,col] = dist_to[row,col] +energy[row+1,col]
				edge_to[row+1,col]=0

			if col != cols-1 and dist_to[row+1,col+1]>dist_to[row,col]+energy[row+1, col+1]:

				dist_to[row+1, col+1] = dist_to[row,col]+energy[row+1,col+1]
				edge_to[row+1,col+1] = -1

	#Retrace the path
	seam[row-1] = np.argmin(dist_to[row-1,:])
	for i in (x for x in reversed(range(rows)) if x>0):
		seam[i-1] = seam[i] + edge_to[i,int(seam[i])]

	return seam
			


def find_horizontal_seam(img,energy):

	rows, cols = img.shape[:2]

	#Empty seam vector along horizontal direction
	seam = np.zeros(img.shape[1])

	dist_to = np.zeros(img.shape[:2])+float('inf')
	dist_to[:,0] = np.zeros(img.shape[0])
	edge_to = np.zeros(img.shape[:2])



	#Compute paths of lowest energy

	for col in range(cols-1):
		for row in range(rows):
			if row !=0 and dist_to[row-1,col+1] > dist_to[row, col]+energy[row-1,col+1]:
				dist_to[row-1,col+1] = dist_to[row, col]+energy[row-1,col+1]
				edge_to[row-1,col+1]=1

			if dist_to[row,col+1] >dist_to[row,col]+energy[row,col+1]:
				dist_to[row,col+1] = dist_to[row,col+1] +energy[row,col+1]
				edge_to[row,col+1]=0

			if row != rows-1 and dist_to[row+1,col+1]>dist_to[row,col]+energy[row+1, col+1]:

				dist_to[row+1, col+1] = dist_to[row,col]+energy[row+1,col+1]
				edge_to[row+1,col+1] = -1

	#Retrace the path
	seam[col-1] = np.argmin(dist_to[:,col-1])
	for i in (y for y in reversed(range(cols)) if y>0):
		seam[i-1] = seam[i] + edge_to[int(seam[i]),i]

	return seam

def overlay_vertical_seam(img, seam):

	img_seam_overlay = np.copy(img)

	x_coords,y_coords = np.transpose([(i,int(j)) for i,j in enumerate(seam)])

	img_seam_overlay[x_coords,y_coords] = (0,255,0)

	return img_seam_overlay

def overlay_horizontal_seam(img, seam):

	img_seam_overlay = np.copy(img)

	y_coords,x_coords = np.transpose([(int(j),int(i)) for j,i in enumerate(seam)])

	img_seam_overlay[x_coords,y_coords] = (255,0,0)

	return img_seam_overlay

def remove_vertical_seam(img, seam):
	rows, cols =img.shape[:2]

	for row in range(rows):
		for col in range(int(seam[row]),cols-1):
			img[row,col] = img[row,col+1]

	img = img[:,0:cols-1]
	return img

def remove_horizontal_seam(img, seam):
	rows, cols =img.shape[:2]

	for col in range(cols):
		for row in range(int(seam[col]),row-1):
			img[row,col] = img[row+1,col]

	img = img[0:row-1,:]
	return img


if __name__ =='__main__':

	boat_raw=cv2.imread('FishingBoat.jpg')

	num_seams_vertical = 1
	num_seams_horizontal = 1

	boat = np.copy(boat_raw)
	boat_overlay = np.copy(boat_raw)
	
	energy=compute_energy(boat)

	for i in range(num_seams_vertical):
		vertical_seam = find_vertical_seam(boat,energy)
		boat_overlay = overlay_vertical_seam(boat_overlay,vertical_seam)

	for i in range(num_seams_horizontal):
		horizontal_seam = find_horizontal_seam(boat,energy)

		boat_overlay = overlay_horizontal_seam(boat_overlay,horizontal_seam)

	cv2.imshow('Raw',boat_raw)
	cv2.imshow('Seams',boat_overlay)
	
	cv2.waitKey()

	cv.destroyAllwindows

	
