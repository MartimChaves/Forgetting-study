import numpy as np
import utils.models as mod
from utils.utils import * 
from PIL import Image, ImageFilter
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

def process_image(img):
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    pil_im = Image.fromarray(img)
    
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
        
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    
    return im_as_var

def recreate_image(im_as_var):
    
    reverse_mean = [-0.4914, -0.4822, -0.4465]
    reverse_std = [1/0.2023, 1/0.1994, 1/0.2010]
    
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

np.random.seed(1)

model_base = "cifar10_epoch_152_alldata_lr_0.1.pth"
model_forget = "cifar10_epoch_152_alldata_lr_0.1_layer1.pth"

model = mod.PreActResNet18(num_classes_first_stage=10,num_classes_second_stage=10)
model.load_state_dict(torch.load(model_forget))


for j in range(0,64,8):
    random_img = np.uint8(np.random.uniform(150, 180, (32, 32, 3)))

    plt.imshow(random_img)
    plt.show()

    processed_image = process_image(random_img)

    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)

    for i in range(1,11):
        x = processed_image
        
        out = model.conv1(x)
        out = model.bn1(out)
        out = F.relu(out)
        
        # out = model.layer1[0].bn1(out)
        # out = F.relu(out)
        # out = model.layer1[0].conv1(out)
        
        out = model.layer1(out)
        #out = model.layer2(out)
        #out = model.layer3(out)
        
        conv_output = out[0,j]
        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        loss = -torch.mean(conv_output)
        # Backward
        loss.backward()
        # Update image
        optimizer.step()
        
        if i % 5 == 0:
            # Recreate image
            result = recreate_image(x)
            plt.imshow(result)
            plt.show()
            
