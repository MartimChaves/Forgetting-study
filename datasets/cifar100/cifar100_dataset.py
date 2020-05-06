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

def get_ssl_dataset(args, subset, transform_train, transform_test, metrics, bmm_th=0.05,th=0.20):

    if not subset:
        num_classes = 100
    else:
        num_classes = len(subset)
    
    cifar_train = Cifar100(args,args.root,subset,args.noise_ratio,train=True,transform=transform_train,ssl=True)

    if args.noise_type == "random_in_noise":
        cifar_train.random_in_noise()
        for idx,i in enumerate(cifar_train.labels):
            cifar_train.soft_labels[idx,i] = 1
    else:
        print(args.noise_type + " not added yet. No changes made.")
    
    temp_clean_indexes = []
    
    for metric in metrics:
        # organize by ascendent order 
        sorted_indxs_metric = np.argsort(metric)
        if args.balanced_set and not args.use_bmm:
            # select th number of samples per class
            for sample_class in range(num_classes):
                # add to clean set
                class_indxs = np.where(cifar_train.labels == sample_class)[0]
                c_idx_in_sorted = np.isin(sorted_indxs_metric,class_indxs) # where the class indexs are in the sorted indexes
                sorted_class_indxs = sorted_indxs_metric[np.where(c_idx_in_sorted==True)]
                n = round(len(sorted_class_indxs)*th)
                temp_clean_indexes.extend(sorted_class_indxs[:n])
        else:
            if not args.use_bmm:
                n = round(len(cifar_train.labels)*th)
                temp_clean_indexes.extend(sorted_indxs_metric[:n])
            else:
                temp_clean_indexes.extend(list(np.where(np.array(metric)<bmm_th)[0]))
    
    metrics_arr = np.array(metrics)
    if metrics_arr.shape[0] > 1: # pylint: disable=unsubscriptable-object
        if args.agree_on_clean:
            temp_clean_indx_arr = np.array(temp_clean_indexes)
            values, count = np.unique(temp_clean_indx_arr,return_counts=True) 
            train_clean_indexes = values[count>metrics_arr.shape[0]-1] #pylint: disable=unsubscriptable-object
        else:
            train_clean_indexes = np.array(list(set(temp_clean_indexes)))
    else:
        train_clean_indexes = np.array(temp_clean_indexes)
    
    # everything else is noisy
    all_indxs = np.array(list(range(cifar_train.labels.shape[0])))       
    train_noisy_indexes = np.where(np.isin(all_indxs,train_clean_indexes)==False)[0]
    
    true_clean = np.isin(train_clean_indexes,cifar_train.clean_indexes)
    boolval, count = np.unique(true_clean, return_counts=True)
    
    percent_clean = round(count[1]/(count[0]+count[1]),7)
    nImgs = count[0]+count[1]
    
    return cifar_train, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs


def get_parallel_datasets():
    
    return



class Cifar100(tv.datasets.CIFAR100):
        
    def __init__(self, args, path, subset, noise_ratio,train=True,transform=None,ssl=False):
        
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
        self.clean_indexes = []
        
        self.soft_labels = np.zeros((len(self.labels), self.num_classes), dtype=np.float32)
        self.neighbour_labels = np.zeros(self.soft_labels.shape)
        self.labelsNoisyOriginal = [] 
        
        self.ssl = ssl
        self._count = 0
        
        if self.ssl:
            self.original_labels = np.copy(self.labels)
            self.prediction = np.zeros((self.args.epoch_update, len(self.train), self.num_classes), dtype=np.float32)
            
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
            if not self.ssl:
                originalNoisyLabel = self.labelsNoisyOriginal[index]
                return image, label, self.soft_labels[index], index, originalNoisyLabel, 0, 0 #label is repeated to fit with rest of the code
            else:
                return image, 0, label, self.soft_labels[index], index
        else:
            return image, label
        
    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed)
        idxes = np.random.permutation(len(self.labels))

        noisy_indexes = idxes[0:self._num]
        self.noisy_indexes = noisy_indexes
        self.clean_indexes = idxes[self._num::]
        
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

    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):
        
        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio) # = 0
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(train_noisy_indexes)))
            idx_rand = []

        if self._count >= self.args.epoch_begin:

            relabel_indexes = list(train_noisy_indexes[idx_relab])
            self.soft_labels[relabel_indexes] = result[relabel_indexes]

            self.labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)

            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.labels[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)

            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))
            
        self._count += 1
        
        return
    
    def update_labels(self, result):
        self.soft_labels = result
        self.labels = self.soft_labels.argmax(axis = 1).astype(np.int64)

