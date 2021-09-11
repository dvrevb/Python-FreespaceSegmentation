from modelU import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
from utils import draw_graph
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2

######### PARAMETERS ##########
valid_size = 0.15
test_size  = 0.15
train_size = 1-(valid_size+test_size)
batch_size = 16
epochs = 30
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, '/content/sample_data/images')
MASK_DIR = os.path.join(DATA_DIR, '/content/sample_data/masks')
AUGM_IMAGE=os.path.join(DATA_DIR,'/content/sample_data/augmented_images')
AUGM_MASK=os.path.join(DATA_DIR,'/content/sample_data/augmentation_masks')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()


# PREPARE AUGMENTED IMAGE AND MASK LISTS
aug_path_list = glob.glob(os.path.join(AUGM_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUGM_MASK, '*'))
aug_mask_path_list.sort()



# DATA CHECK
image_mask_check(image_path_list, mask_path_list)
image_mask_check(aug_path_list, aug_mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TRAIN AND TEST INDICES
train_ind = int(len(indices) * train_size)
valid_ind = int(train_ind + len(indices) * valid_size)


# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[:train_ind]
train_label_path_list = mask_path_list[:train_ind]


# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[train_ind:valid_ind]
valid_label_path_list = mask_path_list[train_ind:valid_ind]

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[valid_ind:]
test_label_path_list = mask_path_list[valid_ind:]


train_input_path_list=aug_path_list+train_input_path_list
train_label_path_list=aug_mask_path_list+train_label_path_list


# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = UNet(n_channels=3, n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses=[]
train_losses=[]
# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    pair_IM=list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object=zip(*pair_IM)
    zipped_list=list(unzipped_object)
    train_input_path_list=list(zipped_list[0])
    train_label_path_list=list(zipped_list[1])
    running_loss = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        optimizer.zero_grad()
        
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(ind)
        if ind == steps_per_epoch-1:
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                    outputs = model(batch_input)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()
                    
                    #break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            model.train()
            val_losses.append(val_loss)
            torch.save(model, 'UnetModel.pth')
torch.save(model, 'UnetModel.pth')
print("Model Saved!")

model = torch.load('UnetModel.pth')
if cuda:
  model=model.cuda()
model.eval()


draw_graph(val_losses,train_losses,epochs)

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
        predict_path=predict_name.replace('images', 'predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)