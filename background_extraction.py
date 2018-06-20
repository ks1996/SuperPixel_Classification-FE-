import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from skimage.io import imsave, imread
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import glob


path = "/nfs001/kavya/inputData/train/M1"

d = 0 
for file in sorted(glob.glob("/nfs001/kavya/inputData/train/*.jpg")):

	img = cv2.imread(file)
	#cv2.imshow('image1', img)
	#print('yes')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imwrite('/home/user1/Desktop/featureEx/newImageMask.jpg',thresh)
        #cv2.imwrite('{!s}/{!s}.JPEG'.format(folder, count), e_size)
	#print('yes')
	# Open the input image as numpy array
	npImage=np.array(Image.open(file))
	#print('yes')
	# Open the mask image as numpy array
	npMask=np.array(Image.open('/home/user1/Desktop/featureEx/newImageMask.jpg').convert("RGB"))

	# Make a binary array identifying where the mask is black
	cond = npMask<128
	#print('yes')
	# Select image or mask according to condition array
	pixels=np.where(cond, npImage, npMask)

	# Save resulting image
	result=Image.fromarray(pixels)
   
	segments_slic = slic(result, n_segments=1700, compactness=0.01, sigma=1)
	im = mark_boundaries(result, segments_slic)
	imsave(path+'%d_B.png'%d, im)	
	d-=1

	#result.save(path + 'Back%d.jpg'%d, 'PNG', quality=90)
	#d+=1
	#result.save('/home/user1/Desktop/featureEx/ISIC_0000000Mask11.png')

