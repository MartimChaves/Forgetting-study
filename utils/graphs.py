import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def graph_accuracy(args,acc_train,acc_val):
    
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
    plt.close(fig3)
    
    return 