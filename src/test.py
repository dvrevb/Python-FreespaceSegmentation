#author: Burak Cevik

import torch
import glob
import tqdm
input_shape = (224, 224)
n_classes=2
import numpy as np
import cv2
import os
from preprocess import tensorize_image

cuda = True

#LOAD MODEL
model = torch.load('UnetModel.pth')
if cuda:
  model=model.cuda()
model.eval()
input_shape = (224, 224)#What size will the image resize
n_classes = 2


######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'test_data')
###############################

image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()


def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)          
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        #SAVE PREDICTED IMAGE
        predict_path=predict_name.replace('test_data', 'predict_data')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))
#MAKE PREDICT
predict(image_path_list)






