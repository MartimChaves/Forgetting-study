import os
import pickle
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed

def get_dataset(args, transform_train, transform_test, num_classes, noise_ratio, noise_type, first_stage, subset):
    # prepare datasets

    #################################### Train set #############################################
    cifar_train = Cifar10Train(args, noise_ratio, num_classes, first_stage, train=True, transform=transform_train, pslab_transform = transform_test,subset=subset)

    #################################### Noise corruption ######################################
    if noise_type == 'random_in_noise':
        if subset:
            sublabelset = np.array(cifar_train.labels)
            if np.min(np.asarray(sublabelset)) == 5:
                sublabelset = sublabelset - 5
            sublabelset = list(sublabelset)
            cifar_train.clean_labels = sublabelset.copy()
            cifar_train.labels = sublabelset.copy()
            
        cifar_train.random_in_noise()
        
        if subset:
            temp_soft_label = cifar_train.soft_labels[::,0:5] 
            cifar_train.soft_labels = temp_soft_label
            cifar_train.labels = np.asarray(cifar_train.labels, dtype=np.long)
            cifar_train.neighbour_labels = np.zeros(cifar_train.soft_labels.shape)
        
    elif noise_type == 'real_in_noise':
        cifar_train.real_in_noise(subset)
        if subset:
            sublabelset = np.array(cifar_train.labels)
            available_labels = np.unique(sublabelset)
            labels_amount = available_labels.shape[0]
            for i in range(labels_amount):
                indxs = np.where(sublabelset==available_labels[i])
                sublabelset[indxs] = i
                
                soft_indxs = np.where(cifar_train.soft_labels[::,available_labels[i]]==1)[0]
                cifar_train.soft_labels[soft_indxs,available_labels[i]] = 0
                cifar_train.soft_labels[soft_indxs,i] = 1
                
                indxs_clean = np.where(cifar_train.clean_labels==available_labels[i])
                cifar_train.clean_labels[indxs_clean] = i
                
            temp_soft_labels = np.zeros([cifar_train.soft_labels.shape[0],labels_amount])
            temp_soft_labels = cifar_train.soft_labels[::,0:labels_amount]
            cifar_train.soft_labels = temp_soft_labels
            
            sublabelset = list(sublabelset)
            cifar_train.labels = sublabelset.copy()
            cifar_train.labels = np.asarray(cifar_train.labels, dtype=np.long)
            cifar_train.neighbour_labels = np.zeros(cifar_train.soft_labels.shape)
            
    else:
        print ('No noise')

    cifar_train.labelsNoisyOriginal = cifar_train.labels.copy()

    #################################### Test set #############################################
    #testset = tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testset = Cifar10Test(root='./data', train=False, transform=transform_test, subset=subset)
    ###########################################################################################

    return cifar_train, testset, cifar_train.clean_labels, cifar_train.noisy_labels, cifar_train.noisy_indexes,  cifar_train.labelsNoisyOriginal


def get_parallel_datasets(args, transform_train, transform_test, noise_ratio,parallel):
    first_stage = True
    num_classes = 10
    #parallel = [0] or [1]
    cifar_train = Cifar10Train(args, noise_ratio, num_classes, first_stage, train=True, transform=transform_train, pslab_transform = transform_test,parallel=parallel)
    
    cifar_train.random_in_noise()
    
    cifar_train.labelsNoisyOriginal = cifar_train.labels.copy()
    
    testset = Cifar10Test(root='./data', train=False, transform=transform_test)
    
    return cifar_train, testset, cifar_train.clean_labels, cifar_train.noisy_labels, cifar_train.noisy_indexes,  cifar_train.labelsNoisyOriginal

class Cifar10Test(tv.datasets.CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False, subset=[]):
        super(Cifar10Test, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if not subset: # Use all Data
            self.data = self.test_data # From super
            self.labels = self.test_labels # From super
        else:
            print("Using a subset of the data.")
            subtestset, sublabelset = self.get_subset(subset)
            self.data = subtestset
            self.labels = sublabelset


    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels
    
    def __len__(self):
        return len(self.data)
    
    def get_subset(self, class_list):
        
        testset = self.test_data # From super
        labels = self.test_labels # From super
        labels_arr = np.array(labels)
        
        counter = 0
        for class_number in class_list:
            if counter == 0:
                class_indx = np.where(labels_arr == class_number)[0]
            else:
                class_indx = np.concatenate((class_indx,np.where(labels_arr == class_number)[0]),axis=None)
            
            counter += 1
            
        subtestset = testset[class_indx]
        sublabelset = labels_arr[class_indx]
        
        # shuffle the arrays
        perm = np.random.permutation(subtestset.shape[0])
        
        subtestset = subtestset[perm]
        sublabelset = sublabelset[perm]
        
        if np.min(sublabelset) == 5:
            sublabelset -= 5
        else:
            available_labels = np.unique(sublabelset)
            labels_amount = available_labels.shape[0]
            for i in range(labels_amount):
                indxs = np.where(sublabelset==available_labels[i])
                sublabelset[indxs] = i
        
        sublabelset = list(sublabelset)
        
        return subtestset, sublabelset

class Cifar10Train(tv.datasets.CIFAR10):
    # including hard labels & soft labels
    def __init__(self, args, noise_ratio, num_classes, first_stage, train=True, transform=None, target_transform=None, pslab_transform=None, download=False, subset=[], parallel=''):
        super(Cifar10Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        ####### New
        self.first_stage = first_stage
        self.noise_ratio = noise_ratio
        self.args = args
        self.num_classes = num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 0
        #self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []
        
        if parallel:
            indx1 = int(parallel[0]*0.5*len(self.train_data))
            indx2 = int(indx1 + 0.5*len(self.train_data))
            self.data = self.train_data[indx1:indx2]
            self.labels = self.train_labels[indx1:indx2]
        elif not parallel:
            if not subset: # Use all Data
                print("Using all of the cifar10 data.")
                self.data = self.train_data # From super
                self.labels = self.train_labels # From super
            else:
                print("Using a subset of the data.")
                subtrainset, sublabelset = self.get_subset(subset)
                self.data = subtrainset
                self.labels = sublabelset
        
        self.alpha = 0.6
        self.Z_exp_labels = np.zeros((len(self.train_labels), 10), dtype=np.float32)

        self.gaus_noise = False
        self.pslab_transform = pslab_transform

        self.neighbour_labels = []
        self.training_pseudolabels = []

        # From in ou split function:
        self.soft_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        if self.args.epoch_update:
            self.prediction = np.zeros((self.args.epoch_update, len(self.data), self.num_classes), dtype=np.float32)
        else:
            self.prediction = []
            
        self._num = int(len(self.labels) * self.noise_ratio)

        self.neighbour_labels = np.zeros(self.soft_labels.shape)
        self.training_pseudolabels = np.zeros(self.soft_labels.shape)
        self.tracking_pseudolabels = np.zeros(self.soft_labels.shape)

        self.current_clean_bin_idx = []
        self.current_clean_idx = []


    ################# Random in-distribution noise #########################
    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed)
        idxes = np.random.permutation(len(self.labels))
        clean_labels = np.copy(self.labels)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # labels[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.labels[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.labels[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1

        self.labels = np.asarray(self.labels, dtype=np.long)
        self.noisy_labels = np.copy(self.labels)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.num_classes, self.num_classes)) - np.identity(self.num_classes))\
                                    *(self.noise_ratio/(self.num_classes -1)) + \
                                    np.identity(self.num_classes)*(1 - self.noise_ratio)


    ##########################################################################
    ################# Real in-distribution noise #########################

    def real_in_noise(self,subset):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed)

        ##### Create te confusion matrix #####
        #if self.num_classes == 10:
        self.confusion_matrix_in = np.identity(10) # cifar10 has 10 classes

        if not subset or (9 and 1 in subset):
            # truck -> automobile
            self.confusion_matrix_in[9, 9] = 1 - self.noise_ratio
            self.confusion_matrix_in[9, 1] = self.noise_ratio

        if not subset or (2 and 0 in subset):
            # bird -> airplane
            self.confusion_matrix_in[2, 2] = 1 - self.noise_ratio
            self.confusion_matrix_in[2, 0] = self.noise_ratio

        if not subset or (5 and 3 in subset):
            # cat -> dog
            self.confusion_matrix_in[3, 3] = 1 - self.noise_ratio
            self.confusion_matrix_in[3, 5] = self.noise_ratio

            # dog -> cat
            self.confusion_matrix_in[5, 5] = 1 - self.noise_ratio
            self.confusion_matrix_in[5, 3] = self.noise_ratio

        if not subset or (7 and 4 in subset):
            # deer -> horse
            self.confusion_matrix_in[4, 4] = 1 - self.noise_ratio
            self.confusion_matrix_in[4, 7] = self.noise_ratio

        idxes = np.random.permutation(len(self.labels))
        clean_labels = np.copy(self.labels)

        for i in range(len(idxes)):
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 0  ## Remove soft-label created during label mapping
            current_label = self.labels[idxes[i]]
            if self._num > 0:
                # current_label = self.labels[idxes[i]]
                conf_vec = self.confusion_matrix_in[current_label,:]
                label_sym = np.random.choice(np.arange(0, 10), p=conf_vec.transpose())
                self.labels[idxes[i]] = label_sym
            else:
                label_sym = current_label

            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1

            if label_sym == current_label:
                self.clean_indexes.append(idxes[i])
            else:
                self.noisy_indexes.append(idxes[i])

        self.labels = np.asarray(self.labels, dtype=np.long)
        self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
        self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
        self.noisy_labels = self.labels
        self.clean_labels = clean_labels

    ##########################################################################

    def __getitem__(self, index):
        img, labels, soft_labels, noisy_labels = self.data[index], self.labels[index], self.soft_labels[index], self.labelsNoisyOriginal[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels, soft_labels, index, noisy_labels, 0, 0
    
    def __len__(self):
        return len(self.data)
    
    def get_subset(self, class_list):
        
        trainset = self.train_data # From super
        labels = self.train_labels # From super
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
        
        # available_labels = np.unique(sublabelset)
        # labels_amount = available_labels.shape[0]
        # for i in range(labels_amount):
        #     indxs = np.where(sublabelset==available_labels[i])
        #     sublabelset[indxs] = i
        
        
        sublabelset = list(sublabelset)
        
        return subtrainset, sublabelset
    
    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):

        idx = (self._count % self.args.epoch_update).astype(np.uint8)
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio)
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

            self.train_labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)


            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.train_labels[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)


            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))

        self.Z_exp_labels = self.alpha * self.Z_exp_labels + (1. - self.alpha) * self.prediction[idx,:]
        self.z_exp_labels =  self.Z_exp_labels * (1. / (1. - self.alpha ** (self._count + 1)))

        self._count += 1

    
    
    
    def update_labels(self, result):
        self.soft_labels = result#self.result
        self.labels = self.soft_labels.argmax(axis = 1).astype(np.int64)
        self._count += 1
