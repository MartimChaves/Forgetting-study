import torchvision as tv
import numpy as np
from PIL import Image
from IPython import embed
import time


def get_dataset(args, transform_train, transform_val, noise_ratio, num_classes):
    # prepare datasets
    # cifar10_train_val = tv.datasets.CIFAR10(args.train_root, train=True, download=args.download)
    svhn_train_val = tv.datasets.SVHN(args.train_root, split="train", download=False)
    # here train_labels is labels, train_data is data...

    # get train/val dataset
    train_indexes, _ = train_val_split(args, svhn_train_val.labels)
    train = svhnTrain(args, noise_ratio, num_classes, train_indexes, split="train", transform=transform_train, pslab_transform = transform_val)
    validation = 1 #svhnTrain(args, val_indexes, split="train", transform=transform_val, pslab_transform = transform_val)

    clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_for_semiSup()

    return train, clean_labels, noisy_labels, noisy_indexes, clean_indexes, validation


def train_val_split(args, train_val):

    np.random.seed(args.seed) #seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    # val_num = int(args.val_samples / args.num_classes)

    # for id in range(args.num_classes):
    #     indexes = np.where(train_val == id)[0]
    #     np.random.shuffle(indexes)
    #     val_indexes.extend(indexes[:val_num])
    #     train_indexes.extend(indexes[val_num:])
    train_indexes = np.arange(train_val.shape[0]) # pylint:disable=unsubscriptable-object
    np.random.shuffle(train_indexes)
    #np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class svhnTrain(tv.datasets.SVHN):
    # including hard labels & soft labels
    def __init__(self, args, noise_ratio, num_classes, train_indexes=None, split="train", transform=None, target_transform=None, pslab_transform=None, download=False):
        super(svhnTrain, self).__init__(args.train_root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        
        if train_indexes is not None:
            self.data = self.data[train_indexes]
            self.labels = np.array(self.labels)[train_indexes]
        self.data = np.transpose(self.data, (0,2,3,1)) ## dimensions from (11301, 3, 32, 32) to (11301, 32, 32, 3)
        self.soft_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        #self.prediction = np.zeros((self.args.epoch_update, len(self.data), 10), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self._num = int(len(self.labels) * self.noise_ratio)
        self._count = 0
        self.alpha = 0.6
        self.original_labels = np.copy(self.labels)
        self.pslab_transform = pslab_transform
        # fix for the tracking:
        self.train_labels = np.copy(self.labels)

    def symmetric_noise_for_semiSup(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed) #seed)

        original_labels = np.copy(self.labels)
        noisy_indexes = [] # initialize the vector 
        clean_indexes = []


        num_unlab_samples = self._num
        num_clean_samples = len(self.labels) - num_unlab_samples

        clean_per_class = int(num_clean_samples / self.num_classes)
        unlab_per_class = int(num_unlab_samples / self.num_classes)

        for id in range(self.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            for i in range(len(indexes)):
                if i < unlab_per_class:
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)

                    # while(label_sym == original_labels[indexes[i]]):
                    #     label_sym = np.random.randint(self.args.num_classes, dtype=np.int32)
                    self.labels[indexes[i]] = label_sym

                self.soft_labels[indexes[i]][self.labels[indexes[i]]] = 1

            noisy_indexes.extend(indexes[:unlab_per_class])
            clean_indexes.extend(indexes[unlab_per_class:])

        # fix for the tracking:
        self.train_labels = np.copy(self.labels)
        # print("Training with {0} labeled samples ({1} unlabeled samples)".format(num_clean_samples, num_unlab_samples))
        return original_labels, self.labels,  np.asarray(noisy_indexes),  np.asarray(clean_indexes)

    def __getitem__(self, index):
        img, labels, soft_labels, z_exp_labels = self.data[index], self.labels[index], self.soft_labels[index], self.z_exp_labels[index]
        # doing this so that it is consistent with all other datasets.
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels, soft_labels, index, 0, 0, 0



class svhnVal(tv.datasets.SVHN):
    def __init__(self, root, val_indexes, train=True, transform=None, target_transform=None, download=False):
        super(svhnVal, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # self.labels & self.data are the attrs from tv.datasets.CIFAR10
        self.val_labels = np.array(self.labels)[val_indexes]
        self.val_data = self.data[val_indexes]
