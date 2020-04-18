import faiss
import os
os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")

import numpy as np
import argparse
import random
from sklearn.metrics import roc_curve, auc

from datasets.cifar10.cifar10_dataset import get_parallel_datasets as get_cifar10_parallel_datasets
from datasets.svhn.svhn import get_dataset as get_svhn_dataset
from datasets.cifar100.cifar100_dataset import get_dataset as get_cifar100_dataset
from torchvision import datasets as torch_datasets
from utils.utils import * 
import utils.models as mod

import time
def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_2nd', type=float, default=0.1, help='learning rate 2nd stage')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma_scheduler', default=0.1, type=float, help='Value to decay the learning rate')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--M_2nd', action='append', type=int, default=[], help="Milestones for the LR sheduler for the second stage of training")
    
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    
    parser.add_argument('--epoch_1st', type=int, default=1, help='training epoches for the 1st stage')
    parser.add_argument('--epoch_2nd', type=int, default=11, help='training epoches for the 2nd stage')
    
    parser.add_argument('--first_stage_num_classes', type=int, default=10, help='number of classes for the first stage of training')
    parser.add_argument('--first_stage_noise_ration', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--first_stage_noise_type', default='random_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--first_stage_data_name', type=str, default='cifar10', help='Dataset to use in the first stage of model training')
    parser.add_argument('--first_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--second_stage_num_classes', type=int, default=10, help='number of classes for the first stage of training')
    parser.add_argument('--second_stage_noise_ration', type=float, default=0.0, help='noise ratio for the first stage of training')
    parser.add_argument('--second_stage_noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--second_stage_data_name', type=str, default='svhn', help='Dataset to use in the first stage of model training')
    parser.add_argument('--second_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--unfreeze_secondStage', type=int, default=11, help='Step/epoch at which models inner layers are set to not frozen')
    parser.add_argument('--freeze_epochWise', dest='freeze_epochWise', default=False, action='store_true', help='if true, inner layers are frozen for the duration of epochs')
    parser.add_argument('--freeze_earlySecondStage', dest='freeze_earlySecondStage', default=False, action='store_true', help='if true, for the first steps in second stage, inner layers of model are frozen')
    parser.add_argument('--batch_eval_preUnfreeze_only', dest='batch_eval_preUnfreeze_only', default=False, action='store_true', help='if true, batch norm will not be set to eval in second stage')
    
    parser.add_argument('--save_best_AUC_model', dest='save_best_AUC_model', default=False, action='store_true', help='if true, measure AUC after tracking and save model for best AUC')
    parser.add_argument('--track_CE', dest='track_CE', default=True, action='store_true', help='if true, track CE')
    
    parser.add_argument('--second_stg_max_median_loss', type=int, default=1500, help='First stage data loss when retraining maximum median loss - after that point, training is alted')
    parser.add_argument('--step_number', type=int, default=16, help='number of steps')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for replicability (default: 42)')
    
    parser.add_argument('--train_root', default='./data', help='root for train data')
    
    parser.add_argument('--experiment_name', type=str, default = 'test_CE',help='name of the experiment (for the output files)')
    
    parser.add_argument('--relearn1st_in2nd', dest='relearn1st_in2nd', default=False, action='store_true', help='Re-learn when in 2nd stage using first stage data')
    parser.add_argument('--relearn_freq', type=int, default=2, help='Frequency of training with 1st stage data when in 2nd stage (2=every 2 epochs, train 1 epoch with 1st stage data)')
    
    parser.add_argument('--NN_k', type=int, default=100, help='Number of neighbours to consider in the LOF computation')
    parser.add_argument('--repeat', type=str, default='True', help='Rerun model training')
    args = parser.parse_args()
    return args

def data_config(data_name, parallel):
    
    
    # Auxiliary function - returns data transforms used for data augmentation
    def data_augmentation_transforms(mean,std):
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        return transform_train, transform_test
    
    num_classes = args.first_stage_num_classes
    noise_ratio = args.first_stage_noise_ration
    noise_type = args.first_stage_noise_type
    subset = args.first_stage_subset
    
    if data_name == 'cifar10':
        print("Loading cifar10 data.")
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        transform_train, transform_test = data_augmentation_transforms(mean,std)
        
        os.chdir("./datasets/cifar10")
        
        trainset, testset, clean_labels, noisy_labels, noisy_indexes, all_labels = get_cifar10_parallel_datasets(args, transform_train, transform_test, noise_ratio,parallel)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
        print("Cifar10 data loaded.")
                    
    else:
        print("Dataset not recognized. Only the cifar10 dataset is available at the moment.")
        return -1
    
    return train_loader, test_loader, noisy_indexes, clean_labels

def main(args):
    
    print("Hello! ", args.freeze_earlySecondStage, " ", args.freeze_epochWise," ",args.relearn1st_in2nd," ",args.first_stage_noise_type)
        
    # variable initialization 
    acc_train_per_epoch_1 = []
    acc_val_per_epoch_1 = []
    acc_train_per_epoch_2 = []
    acc_val_per_epoch_2 = []
    
    print("First stage subset: ",str(args.first_stage_subset))
    
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(seed=args.seed)
    
    #################### Preparing seeds for replicability ########################
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    ################################################################################

    # 1st model path
    subset1_list = "complete"
    model_1st_path = args.first_stage_data_name + '_parallel_1_epoch_' + str(args.epoch_1st) + '_alldata_' + 'lr_' + str(args.lr) + '.pth'
    model_2nd_path = args.first_stage_data_name + '_parallel_2_epoch_' + str(args.epoch_1st) + '_alldata_' + 'lr_' + str(args.lr) + '.pth'
        
    # load data - 1st stage
    parallel_1 = [0]
    train_loader_1, test_loader_1, noisy_indexes_1, clean_labels_1 = data_config(args.first_stage_data_name, parallel_1)

    # load data - 2nd stage
    parallel_2 = [1]
    train_loader_2, test_loader_2, noisy_indexes_2, clean_labels_2 = data_config(args.first_stage_data_name, parallel_2)
    
    print("First stage model path is: ", model_1st_path)
    
    if not os.path.isfile(model_1st_path) and not os.path.isfile(model_2nd_path) or args.repeat == "True":
        
        loss_per_epoch_train_1st = []
        loss_per_epoch_train_2nd = []
        cross_loss_per_epoch_train_1 = []
        cross_loss_per_epoch_train_2 = []
        cE_track_1 = []
        cE_track_2 = []
        cross_cE_track_1 = []
        cross_cE_track_2 = []
        
        if args.first_stage_data_name == 'cifar10' or args.first_stage_data_name == 'svhn' or args.first_stage_data_name == 'cifar100':
            model1 = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
            model2 = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
        
        # optimizer
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.M, gamma=args.gamma_scheduler)
        
        optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.M, gamma=args.gamma_scheduler)
        
        for epoch in range(1,args.epoch_1st+1):
            
            # train model
            loss_per_epoch, top_5_train_ac, top1_train_ac_1, train_time = train_CrossEntropy(args, model1, device, train_loader_1, optimizer1, epoch)
            loss_per_epoch, top_5_train_ac, top1_train_ac_2, train_time = train_CrossEntropy(args, model2, device, train_loader_2, optimizer2, epoch)
            
            # track training data
            epoch_losses_train1 = track_training_loss_plus(args, model1, device, train_loader_1, epoch, fixed_last_layer=True,second_stage_CE_track=cE_track_1)
            epoch_losses_train2 = track_training_loss_plus(args, model2, device, train_loader_2, epoch, fixed_last_layer=True,second_stage_CE_track=cE_track_2)
            
            # cross training 
            cross_epoch_losses_train1 = track_training_loss_plus(args, model2, device, train_loader_1, epoch, fixed_last_layer=True,second_stage_CE_track=cross_cE_track_1)
            cross_epoch_losses_train2 = track_training_loss_plus(args, model1, device, train_loader_2, epoch, fixed_last_layer=True,second_stage_CE_track=cross_cE_track_2)
            
            loss_per_epoch_train_1st.append(epoch_losses_train1)
            loss_per_epoch_train_2nd.append(epoch_losses_train2)
            
            cross_loss_per_epoch_train_1.append(cross_epoch_losses_train1)
            cross_loss_per_epoch_train_2.append(cross_epoch_losses_train2)
            
            # track test data
            loss_per_epoch_1, acc_val_per_epoch_i_1 = test_cleaning(args, model1, device, test_loader_1)
            loss_per_epoch_2, acc_val_per_epoch_i_2 = test_cleaning(args, model2, device, test_loader_2)
            
            scheduler1.step()
            scheduler2.step()
            
            acc_train_per_epoch_1 += [top1_train_ac_1]
            acc_val_per_epoch_1 += acc_val_per_epoch_i_1
            
            acc_train_per_epoch_2 += [top1_train_ac_2]
            acc_val_per_epoch_2 += acc_val_per_epoch_i_2
            
        torch.save(model1.state_dict(), model_1st_path)
        torch.save(model2.state_dict(), model_2nd_path)
        
        args.experiment_name = 'parallel_1'
        plot_loss_CE_acc(args,loss_per_epoch_train_1st,cE_track_1,noisy_indexes_1,clean_labels_1,acc_train_per_epoch_1,acc_val_per_epoch_1)
        args.experiment_name = 'parallel_2'
        plot_loss_CE_acc(args,loss_per_epoch_train_2nd,cE_track_2,noisy_indexes_2,clean_labels_2,acc_train_per_epoch_2,acc_val_per_epoch_2)
        
        args.experiment_name = 'parallel_1_cross'
        plot_loss_CE_acc(args,cross_loss_per_epoch_train_1,cross_cE_track_1,noisy_indexes_1,clean_labels_1,acc_train_per_epoch_1,acc_val_per_epoch_1)
        
        args.experiment_name = 'parallel_2_cross'
        plot_loss_CE_acc(args,cross_loss_per_epoch_train_2,cross_cE_track_2,noisy_indexes_2,clean_labels_2,acc_train_per_epoch_2,acc_val_per_epoch_2)

        ce_concat = np.concatenate((cross_cE_track_1[-1],cross_cE_track_2[-1]))
        label_concat = np.concatenate((train_loader_1.dataset.labels,train_loader_2.dataset.labels))
        ce_and_labels = np.array(ce_concat)
        np.save("accuracy_measures/parallel.npy",ce_and_labels)
        
def plot_loss_CE_acc(args,loss_per_epoch_train,cE_track,noisy_indexes,clean_labels,acc_train_per_epoch,acc_val_per_epoch):
    
    # Prep for graphs plots
    loss_tr = np.asarray(loss_per_epoch_train)
    loss_tr_t = np.transpose(loss_tr)
    
    ce_tr = np.asarray(cE_track)
    ce_tr_t = np.transpose(ce_tr)
    
    noisy_labels_idx = noisy_indexes
    noisy_labels = np.zeros(shape = loss_tr_t.shape[0])

    noisy_labels[noisy_labels_idx] = 1

    original_labels = np.asarray(clean_labels)
    original_labels = np.transpose(original_labels)

    # Plot loss graph
    exp_name = [args.first_stage_data_name + "parallel_loss",args.first_stage_noise_type,"[]","_","_","_"]
    clean_measures, noisy_measures, _ = process_measures(loss_tr_t,noisy_labels)
    graph_measures(exp_name,'Epoch','Loss',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_Loss_Modelplot')
    
    exp_name = [args.first_stage_data_name + "parallel_CE",args.first_stage_noise_type,"[]","_","_","_"]
    clean_measures, noisy_measures, _ = process_measures(ce_tr_t,noisy_labels)
    graph_measures(exp_name,'Epoch','CE',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_CE_Modelplot')
    
    # Plot accuracy graph
    if 'cross' not in args.experiment_name:
        acc_train = np.asarray(acc_train_per_epoch)
        acc_val = np.asarray(acc_val_per_epoch)
        
        graph_accuracy(args,acc_train,acc_val)


if __name__ == "__main__":
    
    args = parse_args()
    main(args)  