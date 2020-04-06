import faiss
import os
os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")

import numpy as np
import argparse
import random
from sklearn.metrics import roc_curve, auc

from datasets.cifar10.cifar10_dataset import get_dataset as get_cifar10_dataset
from datasets.svhn.svhn import get_dataset as get_svhn_dataset
from datasets.cifar100.cifar100_dataset import get_dataset as get_cifar100_dataset
from torchvision import datasets as torch_datasets
from utils.utils_relabel import * 
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
    
    parser.add_argument('--epoch_1st', type=int, default=3, help='training epoches for the 1st stage')
    parser.add_argument('--epoch_2nd', type=int, default=1, help='training epoches for the 2nd stage')
    
    parser.add_argument('--first_stage_num_classes', type=int, default=10, help='number of classes for the first stage of training')
    parser.add_argument('--first_stage_noise_ration', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--first_stage_noise_type', default='random_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--first_stage_data_name', type=str, default='cifar100', help='Dataset to use in the first stage of model training')
    parser.add_argument('--first_stage_subset', nargs='+', type=int, default=[2,14,23,35,48,51,69,74,87,90], help='Classes of dataset to use as subset')
    
    parser.add_argument('--reg_term1', type=float, default=0.8, help='alpha hyperparameter for regularization component Lp (relabeling)')
    parser.add_argument('--reg_term2', type=float, default=0.4, help='beta hyperparameter for regularization component Le (relabeling)')
    
    parser.add_argument('--second_stage_num_classes', type=int, default=4, help='number of classes for the first stage of training')
    parser.add_argument('--second_stage_noise_ration', type=float, default=0.0, help='noise ratio for the first stage of training')
    parser.add_argument('--second_stage_noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--second_stage_data_name', type=str, default='cifar100', help='Dataset to use in the first stage of model training')
    parser.add_argument('--second_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--unfreeze_secondStage', type=int, default=10, help='Step/epoch at which models inner layers are set to not frozen')
    parser.add_argument('--freeze_epochWise', dest='freeze_epochWise', default=False, action='store_true', help='if true, inner layers are frozen for the duration of epochs')
    parser.add_argument('--freeze_earlySecondStage', dest='freeze_earlySecondStage', default=False, action='store_true', help='if true, for the first steps in second stage, inner layers of model are frozen')
    parser.add_argument('--batch_eval_preUnfreeze_only', dest='batch_eval_preUnfreeze_only', default=False, action='store_true', help='if true, batch norm will not be set to eval in second stage')
    
    parser.add_argument('--save_best_AUC_model', dest='save_best_AUC_model', default=False, action='store_true', help='if true, measure AUC after tracking and save model for best AUC')
    parser.add_argument('--track_CE', dest='track_CE', default=False, action='store_true', help='if true, track CE')
    
    parser.add_argument('--second_stg_max_median_loss', type=int, default=10, help='First stage data loss when retraining maximum median loss - after that point, training is alted')
    parser.add_argument('--step_number', type=int, default=1, help='number of steps')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for replicability (default: 42)')
    
    parser.add_argument('--train_root', default='./data', help='root for train data')
    
    parser.add_argument('--experiment_name', type=str, default = 'test6',help='name of the experiment (for the output files)')
    
    parser.add_argument('--relearn1st_in2nd', dest='relearn1st_in2nd', default=False, action='store_true', help='Re-learn when in 2nd stage using first stage data')
    parser.add_argument('--relearn_freq', type=int, default=2, help='Frequency of training with 1st stage data when in 2nd stage (2=every 2 epochs, train 1 epoch with 1st stage data)')
    
    parser.add_argument('--NN_k', type=int, default=100, help='Number of neighbours to consider in the LOF computation')
    parser.add_argument('--warmup_e', type=int, default=2, help='Number of warmup epochs.')

    args = parser.parse_args()
    return args

def data_config(data_name, first_stage = True):
    
    
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
    
    if first_stage:
        num_classes = args.first_stage_num_classes
        noise_ratio = args.first_stage_noise_ration
        noise_type = args.first_stage_noise_type
        subset = args.first_stage_subset
    else:
        num_classes = args.second_stage_num_classes
        noise_ratio = args.second_stage_noise_ration
        noise_type = args.second_stage_noise_type
        subset = args.second_stage_subset
    
    if data_name == 'cifar10':
        print("Loading cifar10 data.")
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        transform_train, transform_test = data_augmentation_transforms(mean,std)
        
        os.chdir("./datasets/cifar10")
        
            
        trainset, testset, clean_labels, noisy_labels, noisy_indexes, all_labels = get_cifar10_dataset(args, transform_train, transform_test, num_classes, noise_ratio, noise_type, first_stage, subset)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
        print("Cifar10 data loaded.")
                
    
    elif data_name == 'svhn':
        print("Loading svhn data.")
        
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        
        transform_train, transform_test = data_augmentation_transforms(mean,std)
        
        os.chdir("./datasets/svhn")
        
        trainset, clean_labels, noisy_labels, noisy_indexes, clean_indexes, _  = get_svhn_dataset(args, transform_train, transform_test, noise_ratio, num_classes)#,noise_type, first_stage, subset)
        testset = torch_datasets.SVHN(root='./data', split='test', download=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
    
    elif data_name == 'cifar100':
        print("Loading cifar100 data.")
        
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
        transform_train, transform_test = data_augmentation_transforms(mean,std)
        
        os.chdir("./datasets/cifar100")
        trainset, testset, clean_labels, noisy_indexes = get_cifar100_dataset(args.train_root,noise_type,subset,noise_ratio,transform_train, transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
    
    else:
        print("Dataset not recognized. Please chose between cifar10 and svhn datasets")
        return -1
    
    return train_loader, test_loader, noisy_indexes, clean_labels


def main(args):
    
    print("Hello! ", args.freeze_earlySecondStage, " ", args.freeze_epochWise," ",args.relearn1st_in2nd," ",args.first_stage_noise_type)
        
    # variable initialization 
    acc_train_per_epoch = []
    acc_val_per_epoch = []
    
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
    if args.first_stage_subset:
        subset1_list = str(args.first_stage_subset[0]) + '_'
        for i in range(1,len(args.first_stage_subset)):
            subset1_list = subset1_list + str(args.first_stage_subset[i]) + '_' 
        model_1st_path = args.first_stage_data_name + '_epoch_' + str(args.epoch_1st) + '_subset_' + subset1_list + 'lr_' + str(args.lr) + '.pth'
    else:
        subset1_list = "complete"
        model_1st_path = args.first_stage_data_name + '_epoch_' + str(args.epoch_1st) + '_alldata_' + 'lr_' + str(args.lr) + '.pth'
        
    # load data - 1st stage
    first_stage_train_loader, first_stage_test_loader, noisy_indexes, clean_labels = data_config(args.first_stage_data_name)
    
    print("First stage model path is: ", model_1st_path)
    
    loss_per_epoch_train_1st = []
    # load model # first_stage_num_classes
    if args.first_stage_data_name == 'cifar10' or args.first_stage_data_name == 'svhn' or args.first_stage_data_name == 'cifar100':
        model = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=args.gamma_scheduler)
    
    for epoch in range(1,args.epoch_1st+1):
        
        # train model
        loss_per_epoch, top_5_train_ac, top1_train_ac, train_time = train_CrossEntropy(args, model, device,
                                                                                    first_stage_train_loader, optimizer, epoch,
                                                                                    num_classes=args.first_stage_num_classes)

        # track training data
        epoch_losses_train = track_training_loss_plus(args, model, device, first_stage_train_loader, epoch)
        loss_per_epoch_train_1st.append(epoch_losses_train)
        # track test data
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, first_stage_test_loader)
        
        scheduler.step()
        
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i
        
    torch.save(model.state_dict(), model_1st_path)
    
    # update noisy_labels
    
    loss_tr = np.asarray(loss_per_epoch_train_1st)
    loss_tr_t = np.transpose(loss_tr)
    noisy_labels_idx = noisy_indexes
    noisy_labels = np.zeros(shape = loss_tr_t.shape[0])

    noisy_labels[noisy_labels_idx] = 1

    original_labels = np.asarray(clean_labels)
    original_labels = np.transpose(original_labels)
    exp_name = [args.first_stage_data_name + "relabel",args.first_stage_noise_type,subset1_list,"_","_","_"]
    clean_measures, noisy_measures, _ = process_measures(loss_tr_t,noisy_labels)
    graph_measures(exp_name,'Epoch','Loss',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_1stModelplot')
    
    acc_train = np.asarray(acc_train_per_epoch)
    acc_val = np.asarray(acc_val_per_epoch)
    
    epochs = range(len(acc_train))
    
    fig3 = plt.figure(3) 
    ax3 = fig3.add_subplot(str(1)+str(1)+str(1))
    ax3.plot(epochs, acc_val, label = 'Max val acc: ' + str(np.max(acc_val)))
    ax3.plot(epochs, acc_train, label = 'Max train acc: ' + str(np.max(acc_train)))
    ax3.set_ylabel('Acc')
    ax3.set_xlabel('Epoch')
    ax3.legend(loc='lower right', prop={'size': 10})
    ax3.grid(True)
    fig3.savefig(args.experiment_name + '_accuracy.png', dpi = 150)
    plt.close()
    
    auc_name = args.first_stage_data_name + '_' + args.first_stage_noise_type
    forget_auc_arr = np.load("relabel_compare/" + auc_name + ".npy", allow_pickle=True)
    
    final_loss = loss_per_epoch_train_1st[-1]
    
    n = round(len(final_loss)*0.2)#.astype(np.uint8)
    
    sorted_indxs_measure = np.argsort(final_loss)
    
    idx_highest = sorted_indxs_measure[-n:]
    idx_lowest = sorted_indxs_measure[:n] 
    
    # calculate % of noisy samples with those indices
    noisy_in_high_loss = np.isin(idx_highest,noisy_indexes)
    val, amount_noisy_in_HiLoss = np.unique(noisy_in_high_loss,return_counts=True)
    percent_noisy_in_HiLoss = round(amount_noisy_in_HiLoss[1]/np.sum(amount_noisy_in_HiLoss),2)
    # calculate % of clean samples of 
    clean_in_low_loss = np.isin(idx_lowest,noisy_indexes)
    val, amount_clean_in_LoLoss = np.unique(clean_in_low_loss,return_counts=True)
    percent_clean_in_LoLoss = round(amount_clean_in_LoLoss[0]/np.sum(amount_clean_in_LoLoss),2)
      
    percent_sim_in_HiLoss, best_case_HL = calc_sim_indxs(idx_highest,forget_auc_arr[0],noisy_indexes)
    percent_sim_in_LoLoss, best_case_LL = calc_sim_indxs(idx_lowest,forget_auc_arr[1],noisy_indexes,noisy=False)
    percent_sim_in_HiCE, best_case_HC  = calc_sim_indxs(idx_highest,forget_auc_arr[2],noisy_indexes)
    percent_sim_in_LoCE, best_case_LC = calc_sim_indxs(idx_lowest,forget_auc_arr[3],noisy_indexes,noisy=False)
    
    # Loss histogram with info
    bins = np.linspace(np.min(final_loss), np.max(final_loss), 60)
    plt.hist(final_loss[noisy_labels == 0], bins, alpha=0.5, label='Clean')
    plt.hist(final_loss[noisy_labels == 1], bins, alpha=0.5, label='Noisy')
    plt.legend(loc='upper right')
    plt.xlabel("Loss")
    plt.ylabel("Number of samples")
    plt.title("Loss histogram - Relabel method")
    plt.figtext(.01,0.97,'noisy samples in 20% highest loss: {} %'.format(percent_noisy_in_HiLoss), fontsize=8, ha='left')
    plt.figtext(.01,0.93,'clean samples in 20% lowest loss: {} %'.format(percent_clean_in_LoLoss), fontsize=8, ha='left')
    plt.figtext(.01,0.01,'index sim to loss in 20% highest loss: {} % ({}%)'.format(percent_sim_in_HiLoss,best_case_HL), fontsize=8, ha='left')
    plt.figtext(.01,0.04,'index sim to loss in 20% lowest loss: {} % ({}%)'.format(percent_sim_in_LoLoss,best_case_LL), fontsize=8, ha='left')
    plt.figtext(.99,0.01,'index sim to CE in 20% highest loss: {} % ({}%)'.format(percent_sim_in_HiCE,best_case_HC), fontsize=8, ha='right')
    plt.figtext(.99,0.04,'index sim to CE in 20% lowest loss: {} % ({}%)'.format(percent_sim_in_LoCE,best_case_LC), fontsize=8, ha='right')
    #plt.show()
    auc_name = args.first_stage_data_name + '_' + args.first_stage_noise_type
    plt.savefig("relabel_compare/" + auc_name + '_relabel.png', dpi = 150)
    plt.close()
    
def calc_sim_indxs(arr1,arr2,noisy_indexes,noisy=True):
    # arr1 = arr1[arr1.argsort()]
    # idx = np.searchsorted(arr1,arr2)
    # idx[idx==len(arr1)] = 0
    # mask = arr1[idx]==arr2
    # out = np.bincount(idx[mask])
    
    overlap = np.isin(arr1,arr2)
    val, amount_overlap = np.unique(overlap,return_counts=True)
    percent_overlap = round(amount_overlap[1]/np.sum(amount_overlap)*100,2)
    
    overlap_noisy = np.isin(arr1[overlap==True],noisy_indexes)
    _, amount_overlap_noisy = np.unique(overlap_noisy,return_counts=True)
    if noisy:
        best_case = round(amount_overlap_noisy[1]/np.sum(amount_overlap_noisy)*100,2)
    else:
        best_case = round(amount_overlap_noisy[0]/np.sum(amount_overlap_noisy)*100,2)
        
    return percent_overlap, best_case #round((np.count_nonzero(out)/len(out))*100,2) 
    
if __name__ == "__main__":
    
    args = parse_args()
    main(args)  