from __future__ import print_function

import faiss

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
#from .utils import accuracy_v2
from utils.AverageMeter import AverageMeter #pylint: disable=no-name-in-module,import-error
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing

import sys
from tqdm import tqdm

from math import pi
from math import cos

import sklearn.decomposition as sk
from sklearn.metrics import roc_curve, auc

def track_wrt_original(args,model,train_loader,device):
    
    model.eval()
    
    with torch.no_grad():
        all_losses_t = torch.Tensor().to(device)
        all_index = torch.LongTensor().to(device)
        
        for imgs, _, labels, soft_labels, index in train_loader:
            
            imgs, labels, index = imgs.to(device), labels.to(device), index.to(device)
            
            # clean_targets = torch.from_numpy(train_loader.dataset.clean_labels)[index]
            # target = clean_targets.to(device)
            
            prediction_preSoft = model(imgs)
            
            prediction = F.log_softmax(prediction_preSoft, dim=1)

            # Losses
            idx_loss = F.nll_loss(prediction, labels, reduction = 'none')
            idx_loss.detach_()

            all_index = torch.cat((all_index, index))
            all_losses_t = torch.cat((all_losses_t, idx_loss))
    
        all_losses = torch.zeros(all_losses_t.size())
        all_losses[all_index.cpu()] = all_losses_t.data.cpu()
    
    return all_losses.data.numpy()
        

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def loss_mixup_reg_ep(preds, labels, targets_a, targets_b, device, lam, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    mixup_loss_a = -torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b

    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = mixup_loss + args.alpha * L_p + args.beta * L_e
    return prob, loss

def train_CrossEntropy_partialRelab(args, model, device, train_loader, optimizer, epoch, train_noisy_indexes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    w = torch.Tensor([0.0])

    top1_origLab = AverageMeter()

    # switch to train mode
    model.train()
    loss_per_batch = []
    acc_train_per_batch = []

    alpha_hist = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)

    alpha = args.Mixup_Alpha  # alpha for mixup

    target_original = torch.from_numpy(train_loader.dataset.original_labels) 

    counter = 1

    for imgs, img_pslab, labels, soft_labels, index in train_loader:
        
        if len(imgs) != 100:
            print("Length of images batch different than 100.")
        
        images = imgs.to(device)
        labels = labels.to(device)
        soft_labels = soft_labels.to(device)
        
        tempdrop = model.drop.p # Using CNN
        model.drop.p = 0.0

        optimizer.zero_grad()
        output_x1 = model(images)
        output_x1.detach_()
        optimizer.zero_grad()

        model.drop.p = tempdrop # extra forward prop
        
        images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        outputs = model(images)

        prob = F.softmax(output_x1, dim=1)
        prob_mixup, loss = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args)
        outputs = output_x1

        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 1])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        top1_origLab_avg = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                       prec1, optimizer.param_groups[0]['lr']))

        counter = counter + 1

    # update soft labels
    train_loader.dataset.update_labels_randRelab(results, train_noisy_indexes, args.label_noise)

    return train_loss.avg, top5.avg, top1_origLab_avg, top1.avg, batch_time.sum
    # Add update_labels_randRelab to dataset files (in this case just cifar10)

def testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)

def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result

def graph_accuracy(args,acc_train,acc_val):
    epochs = range(len(acc_train))
    
    fig3 = plt.figure(3) 
    ax3 = fig3.add_subplot(str(1)+str(1)+str(1))
    ax3.plot(epochs, acc_val, label = 'Max val acc: ' + str(round(np.max(acc_val),3)))
    ax3.plot(epochs, acc_train, label = 'Max train acc: ' + str(round(np.max(acc_train),3)))
    ax3.set_ylabel('Acc')
    ax3.set_xlabel('Epoch')
    ax3.legend(loc='lower right', prop={'size': 10})
    ax3.grid(True)
    fig3.savefig(args.experiment_name + '_accuracy.png', dpi = 150)
    plt.close(fig3)
    
    return 

# Graph utils
def process_measures(measure_arr,noisy_labels,get_auc=False):
        ################ Process Loss ...
        avg_clean = measure_arr[noisy_labels == 0].mean(axis=0)
        std_clean = measure_arr[noisy_labels == 0].std(axis=0)
        if sum(noisy_labels)>0:
            std_noisy = measure_arr[noisy_labels == 1].std(axis=0)
            avg_noisy = measure_arr[noisy_labels == 1].mean(axis=0)

        quart25_clean = np.quantile(measure_arr[noisy_labels == 0], 0.25, axis=0)
        quart75_clean = np.quantile(measure_arr[noisy_labels == 0], 0.75, axis=0)
        median_clean = np.quantile(measure_arr[noisy_labels == 0], 0.5, axis=0)

        if sum(noisy_labels)>0:
            quart25_noisy = np.quantile(measure_arr[noisy_labels == 1], 0.25, axis=0)
            quart75_noisy = np.quantile(measure_arr[noisy_labels == 1], 0.75, axis=0)
            median_noisy = np.quantile(measure_arr[noisy_labels == 1], 0.5, axis=0)
            
        clean_measures = {'avg':avg_clean,'std':std_clean,'quart25':quart25_clean,'quart75':quart75_clean,'median':median_clean}
        
        if sum(noisy_labels)>0:
            noisy_measures = {'avg':avg_noisy,'std':std_noisy,'quart25':quart25_noisy,'quart75':quart75_noisy,'median':median_noisy}
        else:
            noisy_measures = {'avg':'','std':'','quart25':'','quart75':'','meadian':''}
        
        if get_auc:
            auc_values = []
            for i in range(measure_arr.shape[1]):
                fpr, tpr, _ = roc_curve(noisy_labels, measure_arr[::,i])
                roc_auc = auc(fpr, tpr)
                auc_values.append(roc_auc)
        else:
            auc_values = 0
        
        
        return clean_measures, noisy_measures, auc_values
    
def graph_measures(title,xlabel,ylabel,clean_measures,noisy_measures,noisy_labels,plot_name,auc=''):
    x = np.linspace(0, len(clean_measures['avg']),len(clean_measures['avg']))
    nRows = 1
    nCols = 1
    
    fig = plt.figure(1)
    ax = fig.add_subplot(str(nRows)+str(nCols)+str(1))
    
    #ax.set_title(title, y=1.0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(x, clean_measures['median'], 'b-', label='Clean samples')
    ax.fill_between(x, clean_measures['quart25'], clean_measures['quart75'], alpha=0.2, color='b')

    if sum(noisy_labels)>0:
        ax.plot(x, noisy_measures['median'], 'r-', label='Noisy samples')
        ax.fill_between(x, noisy_measures['quart25'], noisy_measures['quart75'], alpha=0.2, color='r')
    
    if auc:
        ax2 = ax.twinx()
        ax2.plot(x, auc, 'k-', label='Auc (max = %0.2f)' % np.max(np.array(auc)))
        ax2.set(ylim=(0, 1))
        ax2.set_ylabel('AUC')
    
    #ax.legend(loc='upper left', prop={'size': 10})
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
    ax.grid(True)
    
    plt.figtext(.01,0.97,'1st stage dataset: {}'.format(title[0]), fontsize=8, ha='left')
    plt.figtext(.5,0.97,'2nd stage dataset: {}'.format(title[3]), fontsize=8, ha='center')
    plt.figtext(.99,.97,'Noise type: {} (0.4)'.format(title[1]), fontsize=8, ha='right')
    plt.figtext(.01,.935,'1st stage subset: {}'.format(title[2]), fontsize=8, ha='left')
    plt.figtext(.99,.935,'2nd stage subset: {}'.format(title[4]), fontsize=8, ha='right')
    plt.figtext(.01,.90,'Freeze method: {}'.format(title[5]), fontsize=8, ha='left')
    
    fig.savefig(plot_name + '.png', dpi = 150)
    plt.close(fig)
    
    return 