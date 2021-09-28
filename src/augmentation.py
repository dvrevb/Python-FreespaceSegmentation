#author: Burak Cevik

from torchvision import transforms
import cv2
import numpy as np
import os
import glob
import tqdm
from PIL import Image
from PIL import ImageOps

valid_size = 0.15
test_size  = 0.15

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
###############################

image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))


# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TRAIN DATASET FROM THE WHOLE DATASET FOR AUGMENTATION
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]


for image in tqdm.tqdm(train_input_path_list):
    #AUGMENTATION
    img=Image.open(image)
    new_img = transforms.ColorJitter(brightness=0.5, contrast=0.1)(img) #random brightness and contrast
    new_mir_img=ImageOps.mirror(new_img)  #mirror image
    #SAVE
    new_path=image[:-4]+"_1"+".jpg"
    new_path=new_path.replace('images', 'augmented_images')
    new_mir_img=np.array(new_mir_img)
    cv2.imwrite(new_path,new_mir_img)

for mask in tqdm.tqdm(train_label_path_list):
    #AUGMENTATION
    msk=Image.open(mask)
    new_mask=ImageOps.mirror(msk)
    #SAVE
    newm_path=mask[:-4]+"_1"+".png"
    newm_path=newm_path.replace('masks', 'augmentation_masks')
    new_mask=np.array(new_mask)
    cv2.imwrite(newm_path,new_mask)
