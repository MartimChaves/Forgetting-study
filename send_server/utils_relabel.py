import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from utils.AverageMeter import * #pylint: disable=import-error,no-name-in-module
from utils.bmm_model import * #pylint: disable=import-error,no-name-in-module
import matplotlib.pyplot as plt

import time
from IPython import embed

import sklearn.decomposition as sk
from sklearn.metrics import roc_curve, auc

def print_model_gradState(model):
    for params in model.parameters():
        print(params.requires_grad)

def track_training_loss_plus(args, model, device, train_loader, epoch, fixed_last_layer=False):
    model.eval()
    
    with torch.no_grad():
        all_losses_t = torch.Tensor().to(device)

        all_pred = torch.Tensor().to(device)
        all_pred_L4 = torch.Tensor().to(device)

        all_index = torch.LongTensor().to(device)
        counter = 0

        ###########################################################################################
        # Prepare model to study layer 4 features
        # activation = {}
        # def get_activation(name):
        #     def hook(model, input, output):
        #         activation[name] = output.detach()
        #     return hook

        # model.layer4.register_forward_hook(get_activation("layer4"))
        
        for batch_idx, (data, _, soft_labels, index, target, _ ,_) in enumerate(train_loader):

            data, target, soft_labels, index = data.to(device), target.to(device), soft_labels.to(device), index.to(device)
            
            if fixed_last_layer:
                data_pre_lastLayer = model.forward_features(data)
                prediction_preSoft = model.forward_supervised(data_pre_lastLayer)
            else:
                prediction_preSoft = model(data)
            
            prediction = F.log_softmax(prediction_preSoft, dim=1)

            # Losses
            idx_loss = F.nll_loss(prediction, target, reduction = 'none')
            idx_loss.detach_()

            all_pred = torch.cat((all_pred, prediction_preSoft))    
            all_index = torch.cat((all_index, index))
            all_losses_t = torch.cat((all_losses_t, idx_loss))
            
            # features from Layer 4
            # prediction_L4 = activation['layer4']
            # prediction_L4 = F.adaptive_avg_pool2d(prediction_L4, (1, 1))
            
            # all_pred_L4 = torch.cat((all_pred_L4, prediction_L4))
            

            if counter % 15 == 0:
                print('Tracking iteration [{}/{} ({:.0f}%)]'.format(counter * len(data), len(train_loader.dataset),
                               100. * counter / len(train_loader)))
                print("Tracking step Loss:" + str(idx_loss.mean(axis=0)))
                
            counter += 1

        all_losses = torch.zeros(all_losses_t.size())
        all_losses[all_index.cpu()] = all_losses_t.data.cpu()
        
        if fixed_last_layer and args.track_CE: # 
            #CE
            all_pred_reordered = torch.zeros(all_pred.size())
            all_pred_reordered[all_index.cpu()] = all_pred.data.cpu()
            
            all_pred_L4_flat = all_pred_L4.view(all_pred_L4.shape[0],-1)
            all_pred_L4_reordered = torch.zeros((all_pred_L4_flat.shape[0], all_pred_L4_flat.size()[1]))
            all_pred_L4_reordered[all_index.cpu()] = all_pred_L4_flat.data.cpu()

            del(all_pred_L4_flat)
            torch.cuda.empty_cache()
            
            faiss.normalize_L2(all_pred_reordered.numpy())
            pca = sk.PCA(n_components=all_pred_reordered.shape[1], whiten=True)
            all_pred_reordered = pca.fit_transform(all_pred_reordered)
            all_pred_reordered = torch.Tensor(all_pred_reordered)
            faiss.normalize_L2(all_pred_reordered.numpy())        
            
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(all_pred_reordered.shape[1])
            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(all_pred_reordered.numpy())
            
            k = args.NN_k # k = 100
            rd, rd_indexes = index.search(all_pred_reordered.numpy(), k+1)
            
            all_labels_soft = train_loader.dataset.soft_labels.copy()
            all_labels = train_loader.dataset.labels.copy() #labelsNoisyOriginal.copy()
            
            for idx, i in enumerate(rd_indexes):
                a, b = np.unique(all_labels[i], return_counts=True)
                # in "a" we have the labels and in "b" the counts
                train_loader.dataset.neighbour_labels[idx][a] += b
            
            faiss.normalize_L2(all_pred_L4_reordered.numpy())
            pca = sk.PCA(n_components=all_pred_L4_reordered.shape[1], whiten=True)
            all_pred_L4_reordered = pca.fit_transform(all_pred_L4_reordered)
            all_pred_L4_reordered = torch.Tensor(all_pred_L4_reordered)
            faiss.normalize_L2(all_pred_L4_reordered.numpy()) 
            
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(all_pred_L4_reordered.shape[1])
            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(all_pred_L4_reordered.numpy())           
            
            rd, rd_indexes = index.search(all_pred_L4_reordered.numpy(), k)
            
            for idx, i in enumerate(rd_indexes):
                a, b = np.unique(all_labels[i], return_counts=True)
                # in "a" we have the labels and in "b" the counts
                train_loader.dataset.neighbour_labels[idx][a] += b
            
            train_loader.dataset.neighbour_labels = F.softmax(torch.from_numpy(train_loader.dataset.neighbour_labels/10), dim = 1).numpy()
            
            neighbour_CE = all_labels_soft*np.log(train_loader.dataset.neighbour_labels + 1e-6)
            neighbour_CE = -np.mean(neighbour_CE, axis = 1)
        else:
            torch.cuda.empty_cache()

    return all_losses.data.numpy()

def test_cleaning(args, model, device, test_loader):
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
    #acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, num_classes=10):
    
    batch_time = AverageMeter() #pylint: disable=undefined-variable
    train_loss = AverageMeter() #pylint: disable=undefined-variable
    top1 = AverageMeter() #pylint: disable=undefined-variable
    top5 = AverageMeter() #pylint: disable=undefined-variable
    
    # switch to train mode
    model.train()

    end = time.time()

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    counter = 1
       
    def freeze_model(model):
        #model.eval()
        for params in model.parameters():
            params.requires_grad = False
        
    def unfreeze_model(model):
        #model.train()
        for params in model.parameters():
            params.requires_grad = True
            
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            
    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    
    for batch_idx, (images, labels, soft_labels, index, _, rot_img, rot_labels) in enumerate(train_loader):
        model.train()
            
        #embed()
        
        # prepare variables
        numb_labels = len(np.unique(labels))
        images, labels, soft_labels, index = images.to(device), labels.to(device), soft_labels.to(device), index.to(device)

        # compute output
        outputs = model(images)
        
        # compute loss
        if epoch < args.warmup_e and args.ce_loss_inWarmup:
            prob, loss, _ = CE_loss(outputs, labels, device, args, criterion)
        else:
            prob, loss = joint_opt_loss(outputs, soft_labels, device, args, num_classes, epoch)

        # compute gradient and do SGD step       
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if numb_labels>=5:
            max_size_acc = 5
        else:
            max_size_acc = 4

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, max_size_acc])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
            
        num_samples =  len(train_loader.sampler)
        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), num_samples, 100. * counter / len(train_loader), loss.item(),
                prec1, optimizer.param_groups[0]['lr']))
        
        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()
                
        counter = counter + 1
    
    if epoch >= args.warmup_e:
        train_loader.dataset.update_labels(results)
    
    return train_loss.avg, top5.avg, top1.avg, batch_time.sum 

# Metrics utils
def joint_opt_loss(preds, soft_labels, device, args, num_classes, epoch):
    # introduce prior prob distribution p

    p = torch.ones(num_classes).to(device) / num_classes

    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))


    loss = L_c + args.reg_term1 * L_p + args.reg_term2 * L_e

    return prob, loss

def CE_loss(preds, labels, device, args, criterion):
    # introduce prior prob distribution p
    prob = F.softmax(preds, dim=1)

    # ignore constant
    loss_all = criterion(preds, labels)
    loss = torch.mean(loss_all)
    return prob, loss, loss_all

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

# Graphs
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

