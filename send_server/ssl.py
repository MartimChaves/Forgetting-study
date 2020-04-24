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

from datasets.cifar10.cifar10_dataset import get_ssl_dataset, get_dataset as get_cifar10_dataset
from utils.ssl_networks import CNN as MT_Net
from utils.TwoSampler import *
from utils.utils_ssl import *
from utils.bmm_model import * 

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=1, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset_type', default='semiSup', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./datasets/cifar10/data', help='root for train data')
    parser.add_argument('--epoch_begin', default=2, type=int, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=1, type=int, help='#epoch to average to update soft labels')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='number of labeled samples')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=0.8, help='Hyper param for loss')
    parser.add_argument('--beta', type=float, default=0.4, help='Hyper param for loss')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='MT_Net', help='the backbone of the network')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--seed_val', type=int, default=42, help='seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--label_noise', type=float, default = 0.0,help='ratio of labeles to relabel randomly')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='the loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--relab', type=str, default='unifRelab', help='choose how to relabel the random samples from the unlabeled set')
    parser.add_argument('--num_classes', type=int, default=10, help='beta parameter for the EMA in the soft labels')
    parser.add_argument('--gausTF', type=bool, default=False, help='apply gaussian noise')
    parser.add_argument('--dropout', type=float, default=0.1, help='cnn dropout')
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

    parser.add_argument('--threshold', type=int, default=0.20, help='Percentage of samples to consider clean')
    parser.add_argument('--agree_on_clean', dest='agree_on_clean', default=False, action='store_true', help='if true, indexes of clean samples must be present in all metric vectors')
    parser.add_argument('--balanced_set', dest='balanced_set', default=False, action='store_true', help='if true, consider x percentage of clean(labeled) samples from all classes')
    parser.add_argument('--forget', dest='forget', default=False, action='store_true', help='if true, use forget results')
    parser.add_argument('--relabel', dest='relabel', default=True, action='store_true', help='if true, use relabel results')
    parser.add_argument('--parallel', dest='parallel', default=False, action='store_true', help='if true, use parallel results')
    parser.add_argument('--use_bmm', dest='use_bmm', default=True, action='store_true', help='if true, create sets based on a bmm model')
    parser.add_argument('--double_run', dest='double_run', default=True, action='store_true', help='if true, run experiment twice')
    
    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test,device):
    args.val_samples = 0
   
    #trainset, testset, clean_labels, noisy_labels, train_noisy_indexes, train_clean_indexes, valset = get_dataset(args, transform_train, transform_test, num_classes, noise_ratio, noise_type, first_stage, subset)

    # load data of meticrs results and select clean and noisy samples
    
    # We need: trainset (aka dataset itself); noisy_indexes; clean_indexes
    # update_labels_randRelab to work
    metrics = []
    if args.forget:
        forget_arr = np.load("accuracy_measures/forget.npy")
        metrics.append(forget_arr)
    
    if args.relabel:
        relabel_arr = np.load("accuracy_measures/relabel.npy")
        metrics.append(relabel_arr)
    
    if args.parallel:
        parallel_arr = np.load("accuracy_measures/parallel.npy")
        metrics.append(parallel_arr)
        
    if args.use_bmm:
        # fit bmm and calculate probs
        for idx,metric in enumerate(metrics):
            # change metrics
            # fit bmm
            all_index = np.array(range(len(metric)))
            B_sorted = bmm_probs(metric,all_index,device,indx_np=True)
            metrics[idx] = B_sorted
        
        
    
    trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs = get_ssl_dataset(args, transform_train, transform_test, metrics)
    
    #train_clean_indexes = trainset.clean_indexes
    batch_sampler = TwoStreamBatchSampler(train_noisy_indexes, train_clean_indexes, args.batch_size, args.labeled_batch_size) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    testset = datasets.CIFAR10(root='./datasets/cifar10/data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, train_noisy_indexes, percent_clean, nImgs

def main(args):#, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.cuda_dev == 0 or args.cuda_dev == 1:
        torch.cuda.set_device(args.cuda_dev)
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
    train_loader, test_loader, train_noisy_indexes, percent_clean, nImgs = data_config(args, transform_train, transform_test,device)#,  dst_folder)

    # model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=0.1)

    # loss_train_epoch = []
    # loss_val_epoch = []
    # acc_train_per_epoch = []
    # acc_val_per_epoch = []
    # new_labels = []

    # for epoch in range(1, args.epoch + 1):
    #     st = time.time()
    #     scheduler.step()
    #     # train for one epoch
    #     print(args.experiment_name, args.labeled_samples)

    #     loss_per_epoch, top_5_train_ac, top1_train_acc_original_labels, \
    #     top1_train_ac, train_time = train_CrossEntropy_partialRelab(\
    #                                                     args, model, device, \
    #                                                     train_loader, optimizer, \
    #                                                     epoch, train_noisy_indexes)
    #     loss_train_epoch += [loss_per_epoch]

    #     loss_per_epoch, acc_val_per_epoch_i = testing(args, model, device, test_loader)

    #     loss_val_epoch += loss_per_epoch
    #     acc_train_per_epoch += [top1_train_ac]
    #     acc_val_per_epoch += acc_val_per_epoch_i
    
    model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch = train_stage(args,train_loader,device,train_noisy_indexes,test_loader)
    
    if args.double_run:
        # re-calculate measure according to original labels
        num_classes = 10
        noise_ratio = 0.40
        noise_type = "random_in_noise"
        first_stage = True
        subset = []
        trainset_measure, _, _, _, _, _ = get_cifar10_dataset(args, transform_train, transform_test, num_classes, noise_ratio, noise_type, first_stage, subset,ssl=True)
        train_loader_measure = torch.utils.data.DataLoader(trainset_measure, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        loss = track_wrt_original(model,train_loader_measure,device)
        # re-fit bmm and calculate probs
        all_index = np.array(range(len(loss)))
        B_sorted = bmm_probs(loss,all_index,device,indx_np=True)
        # re-select sets
        metrics = [B_sorted]
        trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs = get_ssl_dataset(args, transform_train, transform_test, metrics, bmm_th=0.5)
        # re-train
        model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch = train_stage(args,train_loader,device,train_noisy_indexes,test_loader)
        
    
    # write to a text file accuracy values
    save_info = "th_"
    # % of chosen images
    save_info = save_info + str(args.threshold) + "_percentClean_" + str(percent_clean) + "_noImages_" + str(nImgs) + "_"
    # agree
    if args.agree_on_clean:
        save_info = save_info + "agreeOnClean" + "_"
    # balanced set
    if args.balanced_set:
        save_info = save_info + "balancedSet" + "_"
    # forget
    if args.forget:
        save_info = save_info + "forget" + "_"
    # relabel
    if args.relabel:
        save_info = save_info + "relabel" + "_"
    # parallel
    if args.parallel:
        save_info = save_info + "parallel" + "_"
    
    save_info = save_info + "accRes_" + str(round(top1_train_ac,5)) 
    
    path_file = open("accuracy_measures/acc_results.txt","a") 
    path_file.write(save_info + "\n")
    path_file.close()
    
    # Plot accuracy graph
    acc_train = np.asarray(acc_train_per_epoch)
    acc_val = np.asarray(acc_val_per_epoch)
    
    graph_accuracy(args,acc_train,acc_val)
    
    # Plot loss graph
    plt.plot(loss_train_epoch)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss per epoch - ssl')
    plt.savefig(args.experiment_name + '.png', dpi = 150)
    plt.close()        

def train_stage(args,train_loader,device,train_noisy_indexes,test_loader):
    
    model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=0.1)

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
        
    return model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch
    
    
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
