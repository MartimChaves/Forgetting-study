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

from datasets.cifar10.cifar10_dataset import get_ssl_dataset as get_ssl_cifar10_dataset, get_dataset as get_cifar10_dataset
from datasets.cifar100.cifar100_dataset import get_ssl_dataset as get_ssl_cifar100_dataset, get_dataset as get_cifar100_dataset
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
    
    parser.add_argument('--epoch_begin', default=1, type=int, help='the epoch to begin update labels')
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
    
    parser.add_argument('--labeled_batch_size', default=0, type=int, metavar='N', help="labeled examples per minibatch (default: no constrain)")
    
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation (from the training set))')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--DA', type=str, default='standard', help='Choose the type of DA')

    parser.add_argument('--subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--threshold', type=float, default=0.20, help='Percentage of samples to consider clean')
    parser.add_argument('--bmm_th', type=float, default=0.05, help='Probability threshold to consider samples when using bmm')
    parser.add_argument('--threshold_2nd', type=float, default=0.20, help='Percentage of samples to consider clean - when in 2nd stage')
    parser.add_argument('--bmm_th_2nd', type=float, default=0.50, help='Probability threshold to consider samples when using bmm - when in 2nd stage')
    parser.add_argument('--agree_on_clean', dest='agree_on_clean', default=False, action='store_true', help='if true, indexes of clean samples must be present in all metric vectors')
    parser.add_argument('--balanced_set', dest='balanced_set', default=False, action='store_true', help='if true, consider x percentage of clean(labeled) samples from all classes')
    parser.add_argument('--forget', dest='forget', default=False, action='store_true', help='if true, use forget results')
    parser.add_argument('--f_l', dest='f_l', default=False, action='store_true', help='if true, use loss forget results (instead of CE')
    parser.add_argument('--f_FND', dest='f_FND', default=False, action='store_true', help='Use forgetting data as false negative (false clean) detection')
    parser.add_argument('--fn_th', type=float, default=0.05, help='Probability threshold for false negatives')
    parser.add_argument('--relabel', dest='relabel', default=False, action='store_true', help='if true, use relabel results')
    parser.add_argument('--parallel', dest='parallel', default=False, action='store_true', help='if true, use parallel results')
    parser.add_argument('--use_bmm', dest='use_bmm', default=False, action='store_true', help='if true, create sets based on a bmm model')
    parser.add_argument('--double_run', dest='double_run', default=False, action='store_true', help='if true, run experiment twice')
    parser.add_argument('--plot_loss', dest='plot_loss', default=True, action='store_true', help='Plot loss graphs (debugging)')
    
    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test,device):
    args.val_samples = 0
   
    metrics = []
    if args.forget:
        
        forget_arr_name = "forget_" + str(args.noise_ratio) + "_" + str(args.noise_type) + "_" + str(args.dataset)
        forget_arr = np.load("accuracy_measures/" + forget_arr_name + ".npy")
        if not args.f_FND:
            if args.f_l:
                metrics.append(forget_arr[1])
            else:
                metrics.append(forget_arr[0])
        else:
            f_CE = forget_arr[0]
            all_index = np.array(range(len(f_CE)))
            B_sorted = bmm_probs(f_CE,all_index,device,indx_np=True)
            args.fnd_fCE = B_sorted
            
    if args.relabel:
        relabel_arr_name = "relabel_" + str(args.noise_ratio) + "_" + str(args.noise_type) + "_" + str(args.dataset)
        relabel_arr = np.load("accuracy_measures/" + relabel_arr_name + ".npy")
        metrics.append(relabel_arr)
    
    if args.parallel:
        parallel_arr_name = "parallel_" + str(args.noise_ratio) + "_" + str(args.noise_type) + "_" + str(args.dataset)
        parallel_arr = np.load("accuracy_measures/" + parallel_arr_name + ".npy")
        metrics.append(parallel_arr)
        
    if args.use_bmm:
        # fit bmm and calculate probs
        for idx,metric in enumerate(metrics):
            # change metrics
            # fit bmm
            all_index = np.array(range(len(metric)))
            B_sorted = bmm_probs(metric,all_index,device,indx_np=True)
            metrics[idx] = B_sorted
    
    # bins = np.linspace(0, 1, 100)
    # plt.hist(B_sorted, bins, alpha=0.5, label='bmm')
    # plt.legend(loc='upper right')
    # #plt.yscale('log')
    # plt.show()
    
    if args.dataset == "cifar10":
        trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs = get_ssl_cifar10_dataset(args, transform_train, transform_test, metrics,bmm_th=args.bmm_th,th=args.threshold)
        testset = datasets.CIFAR10(root='./datasets/cifar10/data', train=False, download=False, transform=transform_test)
    elif args.dataset == "cifar100":
        trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs, testset = get_ssl_cifar100_dataset(args, transform_train, transform_test, metrics,bmm_th=args.bmm_th,th=args.threshold)
        
    #train_clean_indexes = trainset.clean_indexes
    if args.labeled_batch_size > 0:
        batch_sampler = TwoStreamBatchSampler(train_noisy_indexes, train_clean_indexes, args.batch_size, args.labeled_batch_size) 
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, train_noisy_indexes, percent_clean, nImgs

def main(args):#, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    save_info = ""
    
    if args.dataset == "cifar10":
        args.train_root = "./datasets/cifar10/data"
    elif args.dataset == "cifar100":
        args.train_root = "./datasets/cifar100/data"    
    else:
        print("Unknown dataset.")
        return 
    
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

    # Data loader
    train_loader, test_loader, train_noisy_indexes, percent_clean, nImgs = data_config(args, transform_train, transform_test,device)#,  dst_folder)
    
    # Data loader for tracking (noisy indexes known)
    if args.dataset == "cifar10":
        first_stage = True
        subset = []
        trainset_measure, _, _, _, noisy_labels_idx, _ = get_cifar10_dataset(args, transform_train, transform_test, args.num_classes, args.noise_ratio, args.noise_type, first_stage, subset,ssl=True)
        train_loader_measure = torch.utils.data.DataLoader(trainset_measure, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    elif args.dataset == "cifar100":
        trainset_measure, _, _, noisy_labels_idx = get_cifar100_dataset(args,args.train_root,args.noise_type,args.subset,args.noise_ratio,transform_train, transform_test,ssl=True)      
        train_loader_measure = torch.utils.data.DataLoader(trainset_measure, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    
    # Train model
    model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch, loss_per_epoch_train = train_stage(args,train_loader,device,train_noisy_indexes,test_loader,train_loader_measure)
    
    save_info = save_info_update(args,save_info,percent_clean,nImgs,top1_train_ac)
    
    if args.plot_loss:
        # noisy clean graph 
        loss_tr = np.asarray(loss_per_epoch_train)
        loss_tr_t = np.transpose(loss_tr)
        noisy_labels = np.zeros(shape = loss_tr_t.shape[0])
        noisy_labels[noisy_labels_idx] = 1
        
        exp_name = [args.dataset + "_ssl",args.noise_type,args.subset,"_","_","_"]
        clean_measures, noisy_measures, _ = process_measures(loss_tr_t,noisy_labels)
        graph_measures(exp_name,'Epoch','Loss',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_1st_ssl')
        
    
    if args.double_run:
        # re-calculate measure according to original labels
               
        loss = track_wrt_original(args,model,train_loader_measure,device)
        if args.use_bmm:
            # re-fit bmm and calculate probs
            all_index = np.array(range(len(loss)))
            B_sorted_l = bmm_probs(loss,all_index,device,indx_np=True)
            # re-select sets
            metrics = [B_sorted_l]
        else:
            metrics = [loss]
            
        # create save info function
        if args.dataset == "cifar10":
            trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs = get_ssl_cifar10_dataset(args, transform_train, transform_test, metrics, bmm_th=args.bmm_th_2nd,th = args.threshold_2nd)
        elif args.dataset == "cifar100":
            trainset, train_noisy_indexes, train_clean_indexes, percent_clean, nImgs, _ = get_ssl_cifar100_dataset(args, transform_train, transform_test, metrics, bmm_th=args.bmm_th_2nd,th = args.threshold_2nd)
        
        # re-train
        model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch, loss_per_epoch_train = train_stage(args,train_loader,device,train_noisy_indexes,test_loader,train_loader_measure)

        if args.plot_loss:
            # noisy clean graph 
            loss_tr = np.asarray(loss_per_epoch_train)
            loss_tr_t = np.transpose(loss_tr)
            noisy_labels = np.zeros(shape = loss_tr_t.shape[0])
            noisy_labels[noisy_labels_idx] = 1
            
            exp_name = [args.dataset + "_ssl",args.noise_type,args.subset,"_","_","_"]
            clean_measures, noisy_measures, _ = process_measures(loss_tr_t,noisy_labels)
            graph_measures(exp_name,'Epoch','Loss',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_2nd_ssl')
                
        save_info = save_info_update(args,save_info,percent_clean,nImgs,top1_train_ac)
    
    #plot roc curve here
    
    loss = track_wrt_original(args,model,train_loader_measure,device)
    fpr, tpr, _ = roc_curve(noisy_labels, loss)
    ce_roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % ce_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(args.experiment_name + '_roc_curve' + '.png', dpi = 150)
    plt.close()
    
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

def train_stage(args,train_loader,device,train_noisy_indexes,test_loader,train_loader_measure):
    
    model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=0.1)

    loss_train_epoch = []
    loss_per_epoch_train = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []

    train_counter = 1
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

        if args.plot_loss:
            if train_counter % 15 == 0 or train_counter == args.epoch:
                loss = track_wrt_original(args,model,train_loader_measure,device)
                loss_per_epoch_train.append(loss)
        
        loss_per_epoch, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i
        
        train_counter += 1
        
    return model, top1_train_ac, acc_train_per_epoch, acc_val_per_epoch, loss_train_epoch, loss_per_epoch_train

def save_info_update(args,save_info,percent_clean,nImgs,top1_train_ac):
    
    if not save_info:
        save_info = args.experiment_name + "_th_"
        
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
    
    return save_info


if __name__ == "__main__":
    args = parse_args()
    main(args)
