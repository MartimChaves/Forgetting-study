import faiss
import os

# os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")

import numpy as np
import argparse
import random
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from datasets.cifar10.cifar10_dataset import get_dataset as get_cifar10_dataset
from datasets.svhn.svhn import get_dataset as get_svhn_dataset
from datasets.cifar100.cifar100_dataset import get_dataset as get_cifar100_dataset
from torchvision import datasets as torch_datasets

from utils.utils import CE_loss, track_training_loss_plus 

from utils.graphs import *
import utils.models as mod

import time
def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate 1st stage')
    parser.add_argument('--lr_2nd', type=float, default=0.1, help='learning rate 2nd stage')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--gamma_scheduler', default=0.1, type=float, help='Value to decay the learning rate')
    parser.add_argument('--M_2nd', action='append', type=int, default=[], help="Milestones for the LR sheduler for the second stage of training")
    
    parser.add_argument('--batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    
    parser.add_argument('--epoch_1st', type=int, default=200, help='training epoches for the 1st stage')
    parser.add_argument('--epoch', type=int, default=1, help='training epoches for the 2nd stage')
    
    parser.add_argument('--first_stage_num_classes', type=int, default=10, help='number of classes for the first stage of training')
    parser.add_argument('--first_stage_noise_ration', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--first_stage_noise_type', default='real_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--first_stage_data_name', type=str, default='cifar10', help='Dataset to use in the first stage of model training')
    parser.add_argument('--first_stage_subset', nargs='+', type=int, default=[], help='Classes of dataset to use as subset')
    
    parser.add_argument('--second_stage_num_classes', type=int, default=10, help='number of classes for the first stage of training')
    parser.add_argument('--second_stage_noise_ration', type=float, default=0.0, help='noise ratio for the first stage of training')
    parser.add_argument('--second_stage_noise_type', default='random_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--second_stage_data_name', type=str, default='cifar100', help='Dataset to use in the first stage of model training')
    parser.add_argument('--second_stage_subset', nargs='+', type=int, default=[2,14,23,35,48,51,69,74,87,90], help='Classes of dataset to use as subset')
    
    parser.add_argument('--unfreeze_secondStage', type=int, default=10, help='Step/epoch at which models inner layers are set to not frozen')
    parser.add_argument('--freeze_epochWise', dest='freeze_epochWise', default=False, action='store_true', help='if true, inner layers are frozen for the duration of epochs')
    parser.add_argument('--freeze_earlySecondStage', dest='freeze_earlySecondStage', default=False, action='store_true', help='if true, for the first steps in second stage, inner layers of model are frozen')
    parser.add_argument('--freeze_layers', nargs='+', type=str, default=["layer3","layer4","conv1","bn1"], help='Layers to freeze')
    
    parser.add_argument('--save_best_AUC_model', dest='save_best_AUC_model', default=True, action='store_true', help='if true, measure AUC after tracking and save model for best AUC')
    parser.add_argument('--track_CE', dest='track_CE', default=True, action='store_true', help='if true, track CE')
    parser.add_argument('--save_BMM_probs', dest='save_BMM_probs', default=True, action='store_true', help='if true, save bmm probs')
    
    parser.add_argument('--second_stg_max_median_loss', type=int, default=1500, help='First stage data loss when retraining maximum median loss - after that point, training is alted')
    parser.add_argument('--step_number', type=int, default=1500, help='Max number of steps')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for replicability (default: 42)')
    
    parser.add_argument('--train_root', default='./data', help='root for train data')
    
    parser.add_argument('--experiment_name', type=str, default = 'freeze',help='name of the experiment (for the output files)')
    
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
        
        try:
            os.chdir("../..")
        except:
            # If relative path is not working use absulte path
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
        
        try:
            os.chdir("../..")
        except:
            # If relative path is not working use absulte path
            os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
    
    elif data_name == 'cifar100':
        print("Loading cifar100 data.")
        
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        
        transform_train, transform_test = data_augmentation_transforms(mean,std)
        
        os.chdir("./datasets/cifar100")
        trainset, testset, clean_labels, noisy_indexes = get_cifar100_dataset(args,args.train_root,noise_type,subset,noise_ratio,transform_train, transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        try:
            os.chdir("../..")
        except:
            # If relative path is not working use absulte path
            os.chdir("/home/martim/Documents/work_insight/study_forgetting_v2")
    
    else:
        print("Dataset not recognized. Please chose between cifar10 and svhn datasets")
        return -1
    
    return train_loader, test_loader, noisy_indexes, clean_labels

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def freeze_model_layers(args,model):
    
    model_dic = {"conv1":model.conv1,
                 "bn1":model.bn1,
                 "layer1":model.layer1,
                 "layer2":model.layer2,
                 "layer3":model.layer3,
                 "layer4":model.layer4}
    
    for layer_name in args.freeze_layers:
        layer = model_dic[layer_name]
        if "layer" in layer_name:
            for k in range(2):
                layer[k].bn1.weight.requires_grad = False
                layer[k].bn1.bias.requires_grad = False
                layer[k].bn2.weight.requires_grad = False
                layer[k].bn2.bias.requires_grad = False
                
                layer[k].conv1.weight.requires_grad = False
                #layer[k].conv1.bias.requires_grad = False
                layer[k].conv2.weight.requires_grad = False
                #layer[k].conv2.bias.requires_grad = False
        else:
            if "bn" in  layer_name:
                layer.bias.requires_grad = False
            layer.weight.requires_grad = False

def main(args):
    
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
    
    
    # load model
    if args.first_stage_data_name == 'cifar10' or args.first_stage_data_name == 'cifar100': #second_stage_num_classes
        model = mod.PreActResNet18(num_classes_first_stage=args.first_stage_num_classes,num_classes_second_stage=args.second_stage_num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(model_1st_path))
        #model.eval()
        print("Loaded model from path: ", model_1st_path)
    except:
        print("Model not found.")
        return
    
    # load data - 2nd stage
    second_stage_train_loader, second_stage_test_loader, _, _ = data_config(args.second_stage_data_name, first_stage=False)
    
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr_2nd, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M_2nd, gamma=args.gamma_scheduler)
    
    second_stage_CE_track = []
    loss_per_epoch_train = []
    
    for epoch in range(1, args.epoch + 1):
        
        print("Training!")
        
        counter = 1
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
        for batch_idx, (images, labels, soft_labels, index, _, rot_img, rot_labels) in enumerate(second_stage_train_loader):
            model.train()
            model.apply(set_bn_eval)
        
            images, labels, soft_labels, index = images.to(device), labels.to(device), soft_labels.to(device), index.to(device)
            
            if epoch==1 and counter == 1: 
                if args.freeze_layers:
                    freeze_model_layers(args,model)
            
            out = model.forward_features(images) 
            outputs = model.forward_supervised_2(out)
            
            # compute loss
            prob, loss, _ = CE_loss(outputs, labels, device, args, criterion)
            
            # compute gradient and do SGD step       
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if counter % 1 == 0:
                num_samples =  len(second_stage_train_loader.sampler)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                    epoch, counter * len(images), num_samples, 100. * counter / len(second_stage_train_loader), loss.item(),
                    optimizer.param_groups[0]['lr']))
            
            epoch_losses_train  = track_training_loss_plus(args, model, device, first_stage_train_loader, epoch,fixed_last_layer=True,second_stage_CE_track=second_stage_CE_track)
            
            loss_per_epoch_train.append(epoch_losses_train)
            
            counter += 1
                       
        scheduler.step()
    
    # plot and save graphs
    loss_tr = np.asarray(loss_per_epoch_train)
    loss_tr_t = np.transpose(loss_tr)
    
    noisy_labels = np.zeros(shape = loss_tr_t.shape[0])
    noisy_labels[noisy_indexes] = 1
    
    if args.track_CE:
        ce_tr = np.asarray(second_stage_CE_track)
        ce_tr_t = np.transpose(ce_tr)
    
    
    experiment_info = [args.first_stage_data_name,args.first_stage_noise_type,"_",args.second_stage_data_name,"_","_"] 
    save_file_name = '_'.join(experiment_info)
    
    for layer in args.freeze_layers:
        args.experiment_name += "_" + layer
    
    # Loss, CE and H
    measure_info = {
            'Loss':{
                'measure_arr':loss_tr_t,##
                'title': experiment_info,
                'xlabel':'Step','ylabel':'Loss',
                'plot_name': args.experiment_name + '_loss_cifar' # without .png
                }
            }
    
    if args.track_CE:
        measure_info['CE'] = {
            'measure_arr':ce_tr_t,##
            'title': experiment_info,
            'xlabel':'Step','ylabel':'CE',
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
    
    # save model (for feature visualization)
    freeze_model_path = model_1st_path[:-4]
    
    for layer in args.freeze_layers:
        freeze_model_path += "_" + layer
        
    freeze_model_path += '.pth'
    torch.save(model.state_dict(), freeze_model_path)
    
    return


if __name__ == "__main__":
    
    args = parse_args()
    main(args) 

