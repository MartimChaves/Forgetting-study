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
from utils.utils import * 
import utils.models as mod

import time
def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_2nd', type=float, default=0.001, help='learning rate 2nd stage')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma_scheduler', default=0.1, type=float, help='Value to decay the learning rate')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--M_2nd', action='append', type=int, default=[], help="Milestones for the LR sheduler for the second stage of training")
    
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    
    parser.add_argument('--epoch_1st', type=int, default=1, help='training epoches for the 1st stage')
    parser.add_argument('--epoch_2nd', type=int, default=1, help='training epoches for the 2nd stage')
    
    parser.add_argument('--first_stage_num_classes', type=int, default=6, help='number of classes for the first stage of training')
    parser.add_argument('--first_stage_noise_ration', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--first_stage_noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--first_stage_data_name', type=str, default='cifar10', help='Dataset to use in the first stage of model training')
    parser.add_argument('--first_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--second_stage_num_classes', type=int, default=4, help='number of classes for the first stage of training')
    parser.add_argument('--second_stage_noise_ration', type=float, default=0.0, help='noise ratio for the first stage of training')
    parser.add_argument('--second_stage_noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--second_stage_data_name', type=str, default='cifar10', help='Dataset to use in the first stage of model training')
    parser.add_argument('--second_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--unfreeze_secondStage', type=int, default=10, help='Step/epoch at which models inner layers are set to not frozen')
    parser.add_argument('--freeze_epochWise', dest='freeze_epochWise', default=False, action='store_true', help='if true, inner layers are frozen for the duration of epochs')
    parser.add_argument('--freeze_earlySecondStage', dest='freeze_earlySecondStage', default=False, action='store_true', help='if true, for the first steps in second stage, inner layers of model are frozen')
    parser.add_argument('--batch_eval_preUnfreeze_only', dest='batch_eval_preUnfreeze_only', default=False, action='store_true', help='if true, batch norm will not be set to eval in second stage')
    
    parser.add_argument('--save_best_AUC_model', dest='save_best_AUC_model', default=False, action='store_true', help='if true, measure AUC after tracking and save model for best AUC')
    parser.add_argument('--track_CE', dest='track_CE', default=False, action='store_true', help='if true, track CE')
    
    parser.add_argument('--second_stg_max_median_loss', type=int, default=15, help='First stage data loss when retraining maximum median loss - after that point, training is alted')
    parser.add_argument('--step_number', type=int, default=2, help='number of steps')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for replicability (default: 42)')
    
    parser.add_argument('--train_root', default='./data', help='root for train data')
    
    parser.add_argument('--experiment_name', type=str, default = 'test_CE',help='name of the experiment (for the output files)')
    
    parser.add_argument('--relearn1st_in2nd', dest='relearn1st_in2nd', default=False, action='store_true', help='Re-learn when in 2nd stage using first stage data')
    parser.add_argument('--relearn_freq', type=int, default=2, help='Frequency of training with 1st stage data when in 2nd stage (2=every 2 epochs, train 1 epoch with 1st stage data)')
    
    parser.add_argument('--NN_k', type=int, default=100, help='Number of neighbours to consider in the LOF computation')
    
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
    
    if not os.path.isfile(model_1st_path):
        print("Training first stage of model!")
        loss_per_epoch_train_1st = []
        # load model # first_stage_num_classes
        if args.first_stage_data_name == 'cifar10' or args.first_stage_data_name == 'svhn' or args.first_stage_data_name == 'cifar100':
            model = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
        
        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=args.gamma_scheduler)
        
        for epoch in range(1,args.epoch_1st+1):
            
            # train model
            loss_per_epoch, top_5_train_ac, top1_train_ac, train_time = train_CrossEntropy(args, model, device, first_stage_train_loader, optimizer, epoch)

            # track training data
            epoch_losses_train = track_training_loss_plus(args, model, device, first_stage_train_loader, epoch)
            loss_per_epoch_train_1st.append(epoch_losses_train)
            # track test data
            loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, first_stage_test_loader)
            
            scheduler.step()
            
            acc_train_per_epoch += [top1_train_ac]
            acc_val_per_epoch += acc_val_per_epoch_i
            
        torch.save(model.state_dict(), model_1st_path)

        # Prep for graphs plots
        loss_tr = np.asarray(loss_per_epoch_train_1st)
        loss_tr_t = np.transpose(loss_tr)
        noisy_labels_idx = noisy_indexes
        noisy_labels = np.zeros(shape = loss_tr_t.shape[0])

        noisy_labels[noisy_labels_idx] = 1

        original_labels = np.asarray(clean_labels)
        original_labels = np.transpose(original_labels)

        # Plot loss graph
        exp_name = [args.first_stage_data_name + "standard",args.first_stage_noise_type,subset1_list,"_","_","_"]
        clean_measures, noisy_measures, _ = process_measures(loss_tr_t,noisy_labels)
        graph_measures(exp_name,'Epoch','Loss',clean_measures,noisy_measures,noisy_labels,args.experiment_name + '_1stModelplot')
        
        # Plot accuracy graph
        acc_train = np.asarray(acc_train_per_epoch)
        acc_val = np.asarray(acc_val_per_epoch)
        
        graph_accuracy(args,acc_train,acc_val)
               
    else:
        
        # load model
        if args.first_stage_data_name == 'cifar10' or args.first_stage_data_name == 'cifar100': #second_stage_num_classes
            model = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
            
        model.load_state_dict(torch.load(model_1st_path))
        model.eval()
        print("Loaded model from path: ", model_1st_path)
    
    # load data - 2nd stage
    second_stage_train_loader, second_stage_test_loader, _, _ = data_config(args.second_stage_data_name, first_stage=False)
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr_2nd, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M_2nd, gamma=args.gamma_scheduler)
    
    # Re-initialize variables
    noisy_labels = np.zeros(shape = len(clean_labels))#loss_tr_t.shape[0])

    noisy_labels[noisy_indexes] = 1##

    original_labels = np.asarray(clean_labels)
    original_labels = np.transpose(original_labels)
    
    if args.save_best_AUC_model:
        best_AUC_model = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
        best_AUC_model.load_state_dict(model.state_dict()) # copy params
        highest_auc_loss = []
    
    loss_per_epoch_train = []
    
    if args.track_CE:
        second_stage_CE_track = []
        highest_auc_ce = []
        
    # data needed for second stage
    scd_stg_data = {
        'track_old_data_batchWise':True, # indicate the need to track data from first stage
        'train_loader_cifar':first_stage_train_loader, # first stage data
        'loss_per_epoch_train':loss_per_epoch_train, # list to which info is appended to
        'fixed_last_layer':True, # use second head for second stage data
        'noisy_labels':noisy_labels        
    }     
        
    for epoch in range(1, args.epoch_2nd + 1):
        
        if args.track_CE:
            _, _, _, _ = train_CrossEntropy(args, model, device, second_stage_train_loader, optimizer, epoch, # pylint:disable=undefined-variable  
                                track_old_data_batchWise=True, train_loader_cifar=first_stage_train_loader,
                                loss_per_epoch_train=loss_per_epoch_train, fixed_last_layer=True,
                                noisy_labels=noisy_labels,highest_auc_loss=highest_auc_loss,second_stage_CE_track=second_stage_CE_track,
                                highest_auc_ce=highest_auc_ce) 

        else:
            _, _, _, _ = train_CrossEntropy(args, model, device, second_stage_train_loader, optimizer, epoch, # pylint:disable=undefined-variable  
                                            track_old_data_batchWise=True, train_loader_cifar=first_stage_train_loader,
                                            loss_per_epoch_train=loss_per_epoch_train, fixed_last_layer=True,
                                            noisy_labels=noisy_labels,highest_auc_loss=highest_auc_loss) 

        # if args.relearn1st_in2nd and epoch >= args.unfreeze_secondStage:
            
        #     if epoch % args.relearn_freq == 0:
        #         _, _, _, _ = train_CrossEntropy(args, model, device, first_stage_train_loader, optimizer, epoch,
        #                                         track_old_data_batchWise=True, train_loader_cifar=first_stage_train_loader,
        #                                         loss_per_epoch_train=loss_per_epoch_train)
            
        #     scheduler.step()
        
        scheduler.step()
        
    loss_tr = np.asarray(loss_per_epoch_train)
    loss_tr_t = np.transpose(loss_tr)
    
    if args.track_CE:
        ce_tr = np.asarray(second_stage_CE_track)
        ce_tr_t = np.transpose(ce_tr)
    
    # Plot titles and lables (readibility purposes)
    xlabel_name = 'Step'
    
    if args.second_stage_subset:
        subset2_list = str(args.second_stage_subset[0]) + '_'
        for i in range(1,len(args.second_stage_subset)):
            subset2_list = subset2_list + str(args.second_stage_subset[i]) + '_'
    else:
        subset2_list = "complete"
        
    # Organize freezing info for plot titles (easier readibility)
    if args.freeze_earlySecondStage:
        if args.freeze_epochWise:
            if args.M_2nd:
                milestones = '_' + str(args.M_2nd[0]) + '_'
                for m in range(1,len(args.M_2nd)):
                    milestones = milestones + str(args.M_2nd[m]) + '_'
            else:
                 milestones = "_noMilestones_"
            freeze_method = "freeze_" + str(args.epoch_2nd) + "totalE" + milestones + "lr2nd_" + str(args.lr_2nd) + "_cutoff_val" + str(args.second_stg_max_median_loss) + "_unfreezeE_" + str(args.unfreeze_secondStage)
        else:
            freeze_method = "freeze_" + str(args.freeze_method) + "steps"
    else:
        freeze_method = "noFreeze_lr2nd_" + str(args.lr_2nd) + "_cutoff_val" + str(args.second_stg_max_median_loss)

    experiment_info = [args.first_stage_data_name,args.first_stage_noise_type,subset1_list,args.second_stage_data_name,subset2_list,freeze_method] 
    save_file_name = '_'.join(experiment_info)
    
    # Loss, CE and H
    measure_info = {
            'Loss':{
                'measure_arr':loss_tr_t,##
                'title': experiment_info,
                'xlabel':xlabel_name,'ylabel':'Loss',
                'plot_name': args.experiment_name + '_loss_cifar' # without .png
                }
            }
    
    if args.track_CE:
        measure_info['CE'] = {
            'measure_arr':ce_tr_t,##
            'title': experiment_info,
            'xlabel':xlabel_name,'ylabel':'CE',
            'plot_name': args.experiment_name + '_CE_cifar' # without .png
        }
    
    for measure in measure_info:
        measure_arr = measure_info[measure]['measure_arr']
        title = measure_info[measure]['title']
        xlabel = measure_info[measure]['xlabel']
        ylabel = measure_info[measure]['ylabel']
        plot_name = measure_info[measure]['plot_name']
        clean_measures, noisy_measures, auc_val = process_measures(measure_arr,noisy_labels,get_auc=True)
        graph_measures(title,xlabel,ylabel,clean_measures,noisy_measures,noisy_labels,plot_name,auc=auc_val)
    
    np.save("results/" + save_file_name + ".npy",np.array(loss_tr_t))
    np.save("results/" + save_file_name + "_noisy_indx.npy",np.array(noisy_labels))

    # plot histogram
    # get index of 20% of highest losses
    if args.save_best_AUC_model:
        auc_metrics = [[],[],[],[]]
        
        auc_name = args.first_stage_data_name + '_' + args.first_stage_noise_type
        idx_highest, idx_lowest = graph_hist(highest_auc_loss[-1],noisy_indexes,noisy_labels,auc_name)
        
        auc_metrics[0] = idx_highest
        auc_metrics[1] = idx_lowest
        
        if args.track_CE:
            ce_plt_name = auc_name + "_CE"
            idx_highest_CE, idx_lowest_CE = graph_hist(highest_auc_ce[-1],noisy_indexes,noisy_labels,ce_plt_name,measure_info=["CE"])
            auc_metrics[2] = idx_highest_CE
            auc_metrics[3] = idx_lowest_CE
            
        auc_metrics_arr = np.asarray(auc_metrics)
        np.save("relabel_compare/" + auc_name + '.npy',auc_metrics_arr)    
        
def graph_hist(measure,noisy_indexes,noisy_labels,plt_name,ratio=0.2,measure_info=["Loss"]):
    
    n = round(len(measure)*ratio)
    
    sorted_indxs_measure = np.argsort(measure) # measure is sorted in tracking
    
    idx_highest = sorted_indxs_measure[-n:] 
    idx_lowest = sorted_indxs_measure[:n] 
    
    # calculate % of noisy samples in high loss/CE
    noisy_in_high_loss = np.isin(idx_highest,noisy_indexes)
    val, amount_noisy_in_HiLoss = np.unique(noisy_in_high_loss,return_counts=True)
    percent_noisy_in_HiLoss = round(amount_noisy_in_HiLoss[1]/np.sum(amount_noisy_in_HiLoss),2)
    # calculate % of clean samples in low loss/CE
    clean_in_low_loss = np.isin(idx_lowest,noisy_indexes)
    val, amount_clean_in_LoLoss = np.unique(clean_in_low_loss,return_counts=True)
    percent_clean_in_LoLoss = round(amount_clean_in_LoLoss[0]/np.sum(amount_clean_in_LoLoss),2)
    
    # plot histogram of loss
    bins = np.linspace(0, np.max(measure)-(0.3*np.median(measure)), 30)
    plt.hist(measure[noisy_labels == 0], bins, alpha=0.5, label='Clean')
    plt.hist(measure[noisy_labels == 1], bins, alpha=0.5, label='Noisy')
    plt.legend(loc='upper right')
    plt.xlabel(measure_info[0])
    plt.ylabel("Number of samples")
    plt.title(measure_info[0] + "histogram - highest AUC")
    plt.figtext(.01,0.97,'% noisy samples in 20% highest {}: {}'.format(measure_info[0],percent_noisy_in_HiLoss), fontsize=8, ha='left')
    plt.figtext(.01,0.93,'% clean samples in 20% lowest {}: {}'.format(measure_info[0],percent_clean_in_LoLoss), fontsize=8, ha='left')
    #plt.show()
    plt.savefig("relabel_compare/" + plt_name + '.png', dpi = 150)
    plt.close()
    
    return idx_highest, idx_lowest
        
if __name__ == "__main__":
    
    args = parse_args()
    main(args)  