from PIL import Image
import glob as glob
import skimage as sk

#from skimage.io import imsave
#from skimage.io as io
from resizeimage import resizeimage
#from skimage.io import imsave, imread
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# Open the input image as numpy array
import cv2
import os, sys
import warnings
with warnings.catch_warnings():
     warnings.simplefilter("ignore")
import numpy as np
np.set_printoptions(threshold=np.inf)


def sp_idx(s, index = True):
     u = np.unique(s)
 
     return [np.where(s == i) for i in u]

path = '/nfs001/kavya/inputData/B1_newSP/'

d = 0
for file in sorted(glob.glob("/nfs001/kavya/inputData/B/*.png")):
	img = cv2.imread(file) 
	
	segments_slic = slic(img, n_segments=1750, compactness=0.01, sigma=1) 
	#print(segments_slic.shape)
	a = np.max(segments_slic)
	#print(a)
	
	superpixel_list = sp_idx(segments_slic)
	superpixel      = [img[idx] for idx in superpixel_list]


	

	for i in range(0,a):
	     #print(superpixel[i])
	     superpixel[i] = np.resize(superpixel[i],(56, 56,3))
	     if np.all(superpixel[i]==255): 
                print ("Image is white")
	     else:
                cv2.imwrite(path+'%d_backgrd.png'%d, superpixel[i])
                print(superpixel[i])
	     #imsave(path+'%d_globules.png'%d, superpixel[i])
	     d+=1


