from PIL import Image
import glob
import os

path = "/home/user1/Desktop/featureEx/Kaggle-Carvana-Image-Masking-Challenge-master/input/data/train_new/S1/"
path1 = "/home/user1/Desktop/featureEx/Kaggle-Carvana-Image-Masking-Challenge-master/input/data/train_new/S1_new/"
dirs = os.listdir( path )

for image in dirs: #Iterates through each picture
         print(image)
         if os.path.isfile(path+image):
            im=Image.open(path+image)
            extrema = im.convert("L").getextrema()
            if extrema != (0, 0):
               f, e = os.path.splitext(path1+image)   
               im.save(f + '.png', 'PNG', quality=90)
            # not all black
            '''elif extrema == (1, 1):
             f, e = os.path.splitext(path1+image)   
       img.save(f + '.jpg', 'PNG', quality=90)
            im = Image.open(path+image)#.convert('L')
            
            img = im.resize((250,250), Image.ANTIALIAS)
            f, e = os.path.splitext(path1+image)   
            img.save(f + '.jpg', 'PNG', quality=90)'''


