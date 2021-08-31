from torchvision import transforms
import cv2
import numpy as np
import os
import glob
import tqdm
from PIL import Image

valid_size = 0.15
test_size  = 0.15
train_size = 1- (valid_size+test_size)

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

# DEFINE TRAIN INDICES
train_ind = int(len(indices) * train_size)

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[:train_ind]
train_label_path_list = mask_path_list[:train_ind]


for image in tqdm.tqdm(train_input_path_list):
    img=Image.open(image)
    new_img = transforms.functional.adjust_brightness(img,brightness_factor=0.5)
    #new_img = transforms.functional.adjust_hue(new_img,hue_factor=0.1)
    

    new_path=image[:-4]+"-1"+".jpg"
    new_path=new_path.replace('images', 'augmented_images')
    new_img=np.array(new_img)
    cv2.imwrite(new_path,new_img)
  
for mask in tqdm.tqdm(train_label_path_list):
    msk=cv2.imread(mask)
    new_mask=msk
    newm_path=mask[:-4]+"-1"+".jpg"
    newm_path=newm_path.replace('masks', 'augmentation_masks')
    cv2.imwrite(newm_path,new_mask)
