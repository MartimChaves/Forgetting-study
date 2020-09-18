import numpy as np
from utils.bmm_model import * 
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

def plt_hist(data,data2,min,max,bin_no,xlabel="Probability",ylabel="Number of samples",title="BMM histogram - Relabel data",log=True):
    # Loss histogram with info
    bins = np.linspace(min, max, bin_no)
    plt.hist(data, bins, alpha=0.5, label='Unlabeled')
    plt.hist(data[data2<0.05], bins,alpha=0.5, label='Labeled')
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log:
        plt.yscale('log')

    plt.show()
    # plt.savefig('a.png', dpi = 150)
    # plt.close()
def main_hist():
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    relabel_arr_name = "relabel_" + str(0.4) + "_" + "real_in_noise" + "_" + "cifar10"
    relabel_arr = np.load("accuracy_measures/" + relabel_arr_name + ".npy")
    
    forget_arr_name = "forget_0.4_real_in_noise_cifar10"
    forget_arr = np.load("accuracy_measures/" + forget_arr_name + ".npy")
    f_arr_l = forget_arr[1]
    forget_arr = forget_arr[0]
    
    noisy_labels_name = "cifar10_real_in_noise_complete_cifar100_2_14_23_35_48_51_69_74_87_90__noFreeze_lr2nd_0.001_cutoff_val1500_noisy_indx"
    noisy_labels = np.load(noisy_labels_name + ".npy")
    noisy_indxs = np.where(noisy_labels==1)[0]
    

    # plt_hist(relabel_arr,0.1,19.5,100,xlabel="Loss",ylabel="Number of samples",title="Loss values histogram - Relabel data",log=False) #0.1 19.5
    # plt_hist(relabel_arr,0,19.5,100,xlabel="Loss",ylabel="Log(Number of samples)",title="Loss values histogram - Relabel data",log=True)
    # change metrics
    # fit bmm
    all_index = np.array(range(len(relabel_arr)))
    B_sorted = bmm_probs(relabel_arr,all_index,device,indx_np=True)
    B_sorted_f = bmm_probs(forget_arr,all_index,device,indx_np=True)
    B_f_l = bmm_probs(f_arr_l,all_index,device,indx_np=True)
    
    clean_relabel_indxs = np.where(B_sorted<0.05)[0]

    false_neg = noisy_indxs[np.where(np.isin(noisy_indxs,clean_relabel_indxs)==True)[0]]
    
    plt_hist(B_f_l,B_f_l,0,1,100,ylabel="Log(Number of samples)")
    # plt_hist(B_sorted_f[false_neg],B_sorted_f[false_neg],0,1,100,ylabel="Log(Number of samples)")
    # plt_hist(B_sorted,0,1,100,ylabel="Log(Number of samples)")
    # plt_hist(B_sorted,0,1,100,log=False)
    
    p_r = []
    t_dFN = []
    p_r_lce = []
    t_dFN_lce = []
    x = []
    
    for i in range(1,40):
        th = 0.025*i
        detected_false_neg = np.where(B_sorted_f>th)[0]
        
        dFN_loss = np.where(B_f_l>th)[0]
        
        true_det_FN = detected_false_neg[np.where(np.isin(detected_false_neg,false_neg)==True)[0]]
        possible_removed = len(np.where(np.isin(detected_false_neg,clean_relabel_indxs)==True)[0])
        # pass #fazer corte de prob 0.05 ver que samples ficam

        u_l_ceFN = dFN_loss[np.where(np.isin(dFN_loss,detected_false_neg)==True)[0]]
        true_det_lCE_FN = u_l_ceFN[np.where(np.isin(u_l_ceFN,false_neg)==True)[0]]
        possible_removed_lCE = len(np.where(np.isin(u_l_ceFN,clean_relabel_indxs)==True)[0])
        
        p_r.append(possible_removed)
        t_dFN.append(len(true_det_FN))
        
        p_r_lce.append(possible_removed_lCE)
        t_dFN_lce.append(len(true_det_lCE_FN))
        
        x.append(th)
    
    p_r = np.asarray(p_r)
    t_dFN = np.asarray(t_dFN)
    p_r_lce = np.asarray(p_r_lce)
    t_dFN_lce = np.asarray(t_dFN_lce)
    x = np.asarray(x)
    
    plt.plot(x,t_dFN,'r',label="False negatives (CE - Forgetting)")
    plt.plot(x,p_r,'c',label="Total samples removed (CE - Forgetting)")
    plt.plot(x,t_dFN_lce,'g',label="False negatives (Loss - Forgetting)")
    plt.plot(x,p_r_lce,'k',label="Total samples removed (Loss - Forgetting)")
    plt.xlabel('Probability Threshold')
    plt.ylabel('Number of samples removed')
    plt.title('Relation between number of samples removed and probability (from bmm) threshold')
    plt.legend(loc="upper right")
    plt.show()
    
    #plt.plot(x,t_dFN/1138,'r',x,(p_r-t_dFN)/38701,'c',x,t_dFN_lce/1138,'g',x,(p_r_lce-t_dFN_lce)/38701,'k')
    
    plt.plot(x,t_dFN/1138,'r',label="Percentage of False negatives (CE - Forgetting)")
    plt.plot(x,(p_r-t_dFN)/38701,'c',label="Percentage of True negatives removed (CE - Forgetting)")
    plt.plot(x,t_dFN_lce/1138,'g',label="Percentage of False negatives (Loss - Forgetting)")
    plt.plot(x,(p_r_lce-t_dFN_lce)/38701,'k',label="Percentage of True negatives removed (Loss - Forgetting)")
    plt.xlabel('Probability Threshold')
    plt.ylabel('Number of samples removed')
    plt.title('Relation between number of samples removed and probability (from bmm) threshold')
    plt.legend(loc="upper right")
    plt.show()
    
    
    debugPlt_hist = False
    
    if debugPlt_hist:
        plt_hist(relabel_arr,B_sorted,0.1,19.5,100,xlabel="Loss",ylabel="Number of samples",title="Loss values histogram - Relabel data",log=False) #0.1 19.5
        plt_hist(relabel_arr,B_sorted,0,19.5,100,xlabel="Loss",ylabel="Log(Number of samples)",title="Loss values histogram - Relabel data",log=True)

def main_roc():
    forget_arr_name = "relabel_0.4_real_in_noise_cifar10"
    forget_arr = np.load("accuracy_measures/" + forget_arr_name + ".npy")
    arr = forget_arr#[0]
    
    noisy_labels_name = "cifar10_real_in_noise_complete_cifar100_2_14_23_35_48_51_69_74_87_90__noFreeze_lr2nd_0.001_cutoff_val1500_noisy_indx"
    noisy_labels = np.load(noisy_labels_name + ".npy")
    
    fpr, tpr, _ = roc_curve(noisy_labels, arr)
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
    plt.savefig('relabelRealCifar0.4_roc_curve' + '.png', dpi = 150)
    plt.close()
    
if __name__ == "__main__":
    #main_roc()
    main_hist()
