from PIL import Image
import fileinput
import numpy as np
import glob
import fileinput
#from resizeimage import resizeimage
#from skimage.io import imsave, imread
#from skimage.segmentation import slic
#from skimage.segmentation import mark_boundaries
#from skimage.util import img_as_float
#import matplotlib.pyplot as plt
# Open the input image as numpy array
import cv2
import os, sys
import warnings
with warnings.catch_warnings():
     warnings.simplefilter("ignore")
import numpy as np
np.set_printoptions(threshold=np.inf)


path = "/nfs001/kavya/inputData/NN/*.png"
path1 = "/nfs001/kavya/inputData/NN1/"


for main_image in sorted(glob.glob('/nfs001/kavya/inputData/train/*.jpg')):

    npImage=np.array(Image.open(main_image))
    pathname, imagename = os.path.split(main_image)
    imagenameO , ext = os.path.splitext(imagename)

    for f in sorted(glob.glob(path)):
         print(f)
         pathnameM, imagenameM = os.path.split(f)
         print(imagenameM)
         imagenameM1 = imagenameM[0:12]
         print(imagenameM1)
         if imagenameO == imagenameM1:
             print('yes')
             npMask=np.array(Image.open(f))
             cond = npMask>128
             pixels=np.where(cond, npImage, npMask)
             result=Image.fromarray(pixels)
             f, e = os.path.splitext(path1 + imagenameM)   
             result.save(f + '.png', 'PNG', quality=90)
             break
         else:
             continue



