import numpy as np
from utils.bmm_model import * 
from matplotlib import pyplot as plt
import torch

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

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

relabel_arr_name = "relabel_" + str(0.4) + "_" + "random_in_noise" + "_" + "cifar10"
relabel_arr = np.load("accuracy_measures/" + relabel_arr_name + ".npy")

# plt_hist(relabel_arr,0.1,19.5,100,xlabel="Loss",ylabel="Number of samples",title="Loss values histogram - Relabel data",log=False) #0.1 19.5
# plt_hist(relabel_arr,0,19.5,100,xlabel="Loss",ylabel="Log(Number of samples)",title="Loss values histogram - Relabel data",log=True)
# change metrics
# fit bmm
all_index = np.array(range(len(relabel_arr)))
B_sorted = bmm_probs(relabel_arr,all_index,device,indx_np=True)

# plt_hist(B_sorted,0,1,100,ylabel="Log(Number of samples)")
# plt_hist(B_sorted,0,1,100,log=False)

# pass #fazer corte de prob 0.05 ver que samples ficam

plt_hist(relabel_arr,B_sorted,0.1,19.5,100,xlabel="Loss",ylabel="Number of samples",title="Loss values histogram - Relabel data",log=False) #0.1 19.5
plt_hist(relabel_arr,B_sorted,0,19.5,100,xlabel="Loss",ylabel="Log(Number of samples)",title="Loss values histogram - Relabel data",log=True)

