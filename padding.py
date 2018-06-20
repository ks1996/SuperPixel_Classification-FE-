import numpy as np
import cv2
import glob

path = '/home/user1/Desktop/featureEx/Kaggle-Carvana-Image-Masking-Challenge-master/input/data/train_new/NN1_newSP/*.png'

for file in sorted(glob.glob(path)):
	#print(file)
	im = cv2.imread(file)
	row, col= im.shape[:2]
	bottom= im[row-2:row, 0:col]
	mean= cv2.mean(bottom)[0]

	bordersize=50
	border=cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	cv2.imwrite(file,border)

