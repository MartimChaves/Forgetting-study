import os
import pickle
import torchvision as tv
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed

import matplotlib.pyplot as plt

def get_dataset(args,root,noise_type,subset,noise_ratio,transform_train, transform_test):
    
    #subset = [2,4,6,20,89]
    cifar_train = Cifar100(args,root,subset,noise_ratio,train=True,transform=transform_train)
    
    if noise_type == "random_in_noise":
        cifar_train.random_in_noise()
        for idx,i in enumerate(cifar_train.labels):
            cifar_train.soft_labels[idx,i] = 1
    else:
        print(noise_type + " not added yet. No changes made.")
        
    testset = Cifar100(args,root,subset,noise_ratio,train=False,transform=transform_train)
    cifar_train.labels = np.asarray(cifar_train.labels, dtype=np.long)
    cifar_train.labelsNoisyOriginal = cifar_train.labels.copy()
    
    return cifar_train, testset, cifar_train.clean_labels, cifar_train.noisy_indexes

#class Cifar100Test(tv.datasets.CIFAR100):
    

class Cifar100(tv.datasets.CIFAR100):
        
    def __init__(self, args, path, subset, noise_ratio,train=True,transform=None):
        
        self.args = args
        
        #if transform is given, we transoform data using
        if train:
            data_path = 'train'
            self.trainOrTest = True
        else:
            data_path = 'test'
            self.trainOrTest = False
            
        with open(os.path.join(path, data_path), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform
        
        if subset:
            subdataset, sublabelset = self.get_subset(subset)
            self.labels = sublabelset
            self.train = subdataset
            self.num_classes = len(subset)
            self.clean_labels = self.labels.copy()
        else:
            self.labels = self.data['fine_labels'.encode()]
            self.train = self.data['data'.encode()]
            self.num_classes = 100 
            self.clean_labels = self.labels.copy()
            
        self.noise_ratio = noise_ratio
        self._num = int(len(self.labels) * self.noise_ratio)
        self.noisy_indexes = []
        
        self.soft_labels = np.zeros((len(self.labels), self.num_classes), dtype=np.float32)
        self.neighbour_labels = np.zeros(self.soft_labels.shape)
        self.labelsNoisyOriginal = [] 
            
    def __len__(self):
        return len(self.labels)#data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.labels[index] #self.data['fine_labels'.encode()][index]
        r = self.train[index, :1024].reshape(32, 32) #self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.train[index, 1024:2048].reshape(32, 32) #self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.train[index, 2048:].reshape(32, 32) #self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
            
        if self.trainOrTest:
            originalNoisyLabel = self.labelsNoisyOriginal[index]
            return image, label, self.soft_labels[index], index, originalNoisyLabel, 0, 0 #label is repeated to fit with rest of the code
        else:
            return image, label
        
    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed)
        idxes = np.random.permutation(len(self.labels))

        noisy_indexes = idxes[0:self._num]
        self.noisy_indexes = noisy_indexes
        
        for i in range(len(idxes)):
            if i < self._num:
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.labels[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.labels[idxes[i]] = label_sym

    def get_subset(self, class_list):
        
        trainset = self.data['data'.encode()]
        labels = self.data['fine_labels'.encode()]
        labels_arr = np.array(labels)
        
        counter = 0
        for class_number in class_list:
            if counter == 0:
                class_indx = np.where(labels_arr == class_number)[0]
            else:
                class_indx = np.concatenate((class_indx,np.where(labels_arr == class_number)[0]),axis=None)
            
            counter += 1
            
        subtrainset = trainset[class_indx]
        sublabelset = labels_arr[class_indx]
        
        # shuffle the arrays
        perm = np.random.permutation(subtrainset.shape[0])
        
        subtrainset = subtrainset[perm]
        sublabelset = sublabelset[perm]
        
        available_labels = np.unique(sublabelset)
        labels_amount = available_labels.shape[0]
        for i in range(labels_amount):
            indxs = np.where(sublabelset==available_labels[i]) # avaiable labels are ordered
            sublabelset[indxs] = i
        
        sublabelset = list(sublabelset)
        
        return subtrainset, sublabelset

    def update_labels(self, result):
        self.soft_labels = result#self.result
        self.labels = self.soft_labels.argmax(axis = 1).astype(np.int64)
        #self._count += 1


# os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
# os.chdir("./datasets/cifar100")
# subset = [2,4,6,20,89]
# cifar_train = Cifar100Train('./data',subset,0.4)
# cifar_train.random_in_noise()
# for image in cifar_train:
#     print(image[1])
#     imgplot = plt.imshow(image[0])
#     plt.show()
# os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")