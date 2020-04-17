import faiss
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms

import numpy as np
import random
import sys
import argparse
import logging
import os
import time

from datasets.cifar10.cifar10_dataset import get_dataset
from utils.ssl_networks import CNN as MT_Net
from utils.TwoSampler import *
from utils.utils_ssl import *

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset_type', default='semiSup', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--epoch_begin', default=0, type=int, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=1, type=int, help='#epoch to average to update soft labels')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='number of labeled samples')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=0.8, help='Hyper param for loss')
    parser.add_argument('--beta', type=float, default=0.4, help='Hyper param for loss')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='MT_Net', help='the backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--label_noise', type=float, default = 0.0,help='ratio of labeles to relabel randomly')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='the loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--relab', type=str, default='unifRelab', help='choose how to relabel the random samples from the unlabeled set')
    parser.add_argument('--num_classes', type=int, default=10, help='beta parameter for the EMA in the soft labels')
    parser.add_argument('--gausTF', type=bool, default=False, help='apply gaussian noise')
    parser.add_argument('--dropout', type=float, default=0.0, help='cnn dropout')
    parser.add_argument('--initial_epoch', type=int, default=0, help='#images in each mini-batch')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='set to 1 to choose the second gpu')
    parser.add_argument('--save_checkpoint', type=str, default= "False", help='save checkpoints for ensembles')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Daraset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=350, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=-0.01, help='LR')
    parser.add_argument('--labeled_batch_size', default=16, type=int, metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation (from the training set))')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--DA', type=str, default='standard', help='Choose the type of DA')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test):#, dst_folder):

    args.val_samples = 0

    num_classes = 10
    noise_ratio = 0.4
    noise_type = 'ramdon_in'
    subset = []
    first_stage = True #irrelevant - to be removed
    
    trainset, clean_labels, noisy_labels, train_noisy_indexes, train_clean_indexes, valset = get_dataset(args, transform_train, transform_test, num_classes, noise_ratio, noise_type, first_stage, subset)

    batch_sampler = TwoStreamBatchSampler(train_noisy_indexes, train_clean_indexes, args.batch_size, args.labeled_batch_size) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, train_noisy_indexes

def main(args):#, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.cuda_dev == 0:
        torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data loader
    train_loader, test_loader, train_noisy_indexes = data_config(args, transform_train, transform_test)#,  dst_folder)

    model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    new_labels = []

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()
        # train for one epoch
        print(args.experiment_name, args.labeled_samples)

        loss_per_epoch, top_5_train_ac, top1_train_acc_original_labels, \
        top1_train_ac, train_time = train_CrossEntropy_partialRelab(\
                                                        args, model, device, \
                                                        train_loader, optimizer, \
                                                        epoch, train_noisy_indexes)


        loss_train_epoch += [loss_per_epoch]

        loss_per_epoch, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i

if __name__ == "__main__":
    args = parse_args()
    #logging.info(args)
    # record params
    #dst_folder = record_params(args)
    # train
    main(args)#, dst_folder)
