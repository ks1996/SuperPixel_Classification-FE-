
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import glob as glob

np.set_printoptions(threshold=np.inf)
# construct the argument parser and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())'''
numSegments = 996
# load the image and apply SLIC and extract (approximately)
# the supplied number of segments

#for fn in glob.glob('/home/user1/Desktop/featureEx/data/ISIC2018_Task1-2_Training_Input/*.jpg'):
image = cv2.imread('/home/user1/Desktop/featureEx/ISIC_0000000.jpg', 1)
segments = slic(image,compactness=11, n_segments = 1004, sigma = 0)
cv2.imshow(mark_boundaries(image, segments))
   
'''
for (counter, seg) in enumerate(segments):
        print(seg)
       
        print('+++++++++++++++++++++++++++++++++++++++++++')
	


# show the output of SLIC
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.show()

# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	print(segVal)
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i))
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255
 
	# show the masked region
	cv2.imshow("Mask", mask)
	cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
	cv2.waitKey(0)
'''

