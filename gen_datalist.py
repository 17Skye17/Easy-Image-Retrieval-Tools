import os
import sys
from tqdm import tqdm
from PIL import Image

from io import BytesIO

VALID_EXTENSIONS = ['jpg','png','JPG','PNG']

img_dir = sys.argv[1]
save_file = sys.argv[2]
images = os.listdir(img_dir)

def check_valid(image):
    try:
        pil_image = Image.open(image)
        return True
    except:
        print ("Warning: Failed to parse image{}".format(image))
        return False

print ("number of images: {}".format(len(images)))
images = [img for img in images if img.split('.')[1] in VALID_EXTENSIONS]

print ("saved images: {}".format(len(images)))
image_path = [os.path.join(img_dir,image) for image in images]

f = open(save_file,'w')
for image in tqdm(image_path):
    flag = check_valid(image)
    if flag == True:
        f.write(image+'\n')
f.close()

