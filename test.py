# import numpy as np
# T = [4,940,444,10934,42395]
# for indx,test in enumerate(T):
#     number = test
#     number_arr = np.array([int(char) for char in str(number)])
#     order = number_arr.shape[0] #pylint:disable=unsubscriptable-object
#     position_4 = np.where(number_arr == 4)[0] 
#     subtractor = 0
#     for pos in position_4:
#         subtractor += 10**(order-(1+pos))
        
#     print("case #"+str(indx)+": ",number-subtractor," + ",subtractor)

import numpy as np
import matplotlib.pyplot as plt
import math

# x = np.linspace(0, 4*math.pi, 1000)
# y = np.sin(x)
# nRows = 1
# nCols = 1

# fig = plt.figure(1)
# #plt.axes([.1,.1,.8,.7])
# #ax = plt.axes()
# ax = fig.add_subplot(str(nRows)+str(nCols)+str(1))


# plt.figtext(.01,0.97,'1st stage dataset: Cifar10', fontsize=8, ha='left')
# plt.figtext(.5,0.97,'2nd stage dataset: Svhn', fontsize=8, ha='center')
# plt.figtext(.99,.97,'Noise type: Real_in (0.4)', fontsize=8, ha='right')
# plt.figtext(.01,.935,'1st stage subset: 00 11 22 33 44 55 66 77 88 99', fontsize=8, ha='left')
# plt.figtext(.99,.935,'2nd stage subset: 00 11 22 33 44 55 66 77 88 99', fontsize=8, ha='right')
# plt.figtext(.01,.90,'Freeze method: 15 total Epochs; M: 5 10 15; lr2nd: 0.1; Cutoff value: 1500; Unfreeze @ Epoch: 2;', fontsize=8, ha='left')
# #"freeze_" + str(args.epoch_2nd) + "totalE" + milestones + "lr2nd_" + str(args.lr_2nd) + "_cutoff_val" + str(args.second_stg_max_median_loss) + "_unfreezeE_" + str(args.unfreeze_secondStage)

# ax.plot(x, y, 'b-', label='Sin(x)')
# #fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
# ax.grid(True)

# ax.set_xlabel("Step")
# ax.set_ylabel("Loss")

# ax2 = ax.twinx()
# auc = [2,3,4,5,6,7,8,9]
# ax2.plot(x, np.cos(x), 'k-', label='Auc (max = %0.2f)' % np.max(np.array(auc)))
# ax2.set(ylim=(0, 1))
# ax2.set_ylabel('AUC')

# fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

# fig.savefig("title_test" + '.png', dpi = 190)
# plt.close(fig)

lst = np.array([3,4,6,12,131,15,8,9])

vals = np.array([0.91,0.45,0.62,1.2,0.31,0.115,0.08,0.9])

sorting = np.argsort(vals)

print(lst[sorting])

