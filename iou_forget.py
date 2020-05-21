import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')

    parser.add_argument('--noise_ratio', type=float, default=0.4, help='noise ratio for the first stage of training')
    parser.add_argument('--noise_type', default='random_in_noise', help='noise type of the dataset for the first stage of training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use in the first stage of model training')

    args = parser.parse_args()
    return args

def main(args):

    array_name = "forget_" + str(args.noise_ratio) + "_" + str(args.noise_type) + "_" + str(args.dataset)
    loss_ce_cifar100 = np.load("accuracy_measures/" + array_name + ".npy")

    loss_cifar100 = loss_ce_cifar100[1]
    ce_cifar100 = loss_ce_cifar100[0]
    
    loss_cifar100_si = np.argsort(loss_cifar100)
    ce_cifar100_si = np.argsort(ce_cifar100)
    
    array_name += "_svhn"
    loss_ce_svhn = np.load("accuracy_measures/" + array_name + ".npy")
    
    loss_svhn = loss_ce_svhn[1]
    ce_svhn = loss_ce_svhn[0]

    loss_svhn_si = np.argsort(loss_svhn)
    ce_svhn_si = np.argsort(ce_svhn)
    
    ths = [100, 500, 1000, 2000, 5000, 10000, 12000, 15000, 17000, 20000, 25000, 30000]
    
    x = np.arange(0,50000,1)
    
    iou_loss = []
    iou_ce = []
    for th in range(1,50000):
        indxs_loss_iou = iou(loss_cifar100_si,loss_svhn_si,th)
        indxs_ce_iou = iou(ce_cifar100_si,ce_svhn_si,th)
        
        iou_loss.append(indxs_loss_iou)
        iou_ce.append(indxs_ce_iou)
    
    # save table
    # print("Thresholds:",ths)
    # print("Loss iou:",iou_loss)
    # print("CE iou:",iou_ce)
    
    plt.plot(x,iou_loss,'g',x,iou_ce,'b')
    plt.show()

def iou(arr_1,arr_2,th):
    
    arr_1_indxs = arr_1[-th::] 
    arr_2_indxs = arr_2[-th::]
    
    overlap_indxs = len(np.where(np.isin(arr_1_indxs,arr_2_indxs)==True)[0])
    
    iou = overlap_indxs / (2*len(arr_1_indxs)-overlap_indxs)
    
    return round(iou,5)


if __name__ == "__main__":
    
    args = parse_args()
    main(args)  