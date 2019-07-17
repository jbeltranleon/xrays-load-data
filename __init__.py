import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def show(multi_disease_model, test_X, test_Y, history, all_labels):
    #### Pred one batch ####
    pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = False)
    plt.figure(figsize=[20,6])
    grid = plt.GridSpec(2, 4, wspace=0.2, hspace=0.4)
    ######### Plots of losses
    ### 1 ###
    plt.subplot(grid[0, 0])
    # batch_losses line
    plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)
    # Axis labels
    plt.xlabel('# of batches trained')
    plt.ylabel('Training loss')
    # Title
    plt.title('1) Training loss vs batches trained')
    #Small square with convension on the rigth side
    plt.legend()
    plt.ylim(0,1)
    plt.grid(True)
    ### 2 ###
    plt.subplot(grid[0, 1])
    # epochs_losses line
    plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)
    # epochs_val_losses line
    plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)
    # Axis labels
    plt.xlabel('# of epochs trained')
    plt.ylabel('Training loss')
    # Title
    plt.title('2) Training loss vs epochs trained')
    #Small square with convension on the rigth side
    plt.legend()
    plt.ylim(0,0.5)
    plt.grid(True)
    ### 3 ###
    ###### Plots of acc ######
    plt.subplot(grid[1, 0])
    # batch_acc line
    plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)
    # Axis labels
    plt.xlabel('# of batches trained')
    plt.ylabel('Training accuracy')
    # Title
    plt.title('3) Training accuracy vs batches trained')
    #Small square with convension on the left side
    plt.legend(loc=3)
    plt.ylim(0,1.1)
    plt.grid(True)
    ### 4 ###
    plt.subplot(grid[1, 1])
    # epochs_acc line
    plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)
    # epochs_val_acc line
    plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)
    # Axis labels
    plt.xlabel('# of epochs trained')
    plt.ylabel('Training accuracy')
    # Sub comment
    plt.text(22,0.3, "(a) my label", size=12, ha="center")
    # Title
    plt.title('4) Training accuracy vs epochs trained')
    #Small square with convension on the left side
    plt.legend(loc=3)
    plt.ylim(0.5,1)
    plt.grid(True)
    #### AUCRoc Curve
    plt.subplot(grid[0:,2:])
    for (idx, c_label) in enumerate(all_labels):
        #Points to graph
        fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
        plt.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    #convention
    plt.legend()
    #Labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate');