import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error,r2_score
from Training_pkg.Statistic_Func import regress2, linear_regression
from Training_pkg.utils import *
from Evaluation_pkg.utils import *
import pickle
import pandas as pd

def plot_valid_training_loss_accuracy_with_epoch_together(training_loss_recording:np.array, training_accuracy_recording:np.array,valid_loss_recording:np.array,
                                                        valid_accuracy_recording:np.array, outfile:str):
    train_loss = np.mean(training_loss_recording,axis=(0,1))
    train_loss = train_loss[np.where(train_loss>0)]
    valid_loss = np.mean(valid_loss_recording,axis=(0,1))
    valid_loss = valid_loss[np.where(valid_loss>0)]
    train_accuracy = np.mean(training_accuracy_recording,axis=(0,1))
    train_accuracy = train_accuracy[np.where(train_accuracy>0)]
    valid_accuracy = np.mean(valid_accuracy_recording,axis=(0,1))
    valid_accuracy = valid_accuracy[np.where(valid_accuracy>0)]

    Train_COLOR_ACCURACY = "#69b3a2"
    Train_COLOR_LOSS = "#3399e6"
    Valid_COLOR_ACCURACY = "#F3b3a2"
    Valid_COLOR_LOSS = "#CBC244"
    train_epoch_x = np.array(range(len(train_accuracy)))
    train_batchsize = np.around(len(train_loss)/len(train_accuracy))
    valid_epoch_x = np.array(range(len(valid_accuracy)))
    valid_batchsize = np.around(len(valid_loss)/len(valid_accuracy))

    train_accuracy_x = train_epoch_x * train_batchsize
    train_loss_x = np.array(range(len(train_loss))) 
    valid_accuracy_x = valid_epoch_x * train_batchsize
    valid_loss_x = np.array(range(len(valid_loss)))*np.round(len(train_loss)/len(valid_loss))

    fig = plt.figure(figsize=(24, 8))
    
    ax1 = fig.add_axes([0.1, 0.2, 0.9, 0.9])
    #fig, ax1 = plt.subplots(figsize=(24, 8))
    ax2 = ax1.twinx()

    ax1.plot(train_loss_x, train_loss, color=Train_COLOR_LOSS, lw=1, label='Train_Loss')
    ax1.set_yscale('log')
    ax2.plot(train_accuracy_x, train_accuracy, color=Train_COLOR_ACCURACY, lw=3, label = 'Train_Accu')

    ax1.plot(valid_loss_x, valid_loss, color=Valid_COLOR_LOSS, lw=1, label='Valid_Loss')
    ax1.set_yscale('log')
    ax2.plot(valid_accuracy_x, valid_accuracy, color=Valid_COLOR_ACCURACY, lw=3, label = 'Valid_Accu')

    x_labels = [str(i) for i in train_epoch_x]

    ax1.set_xlabel("Epoch",fontsize=24)
    ax1.set_ylabel("Loss", fontsize=24)
    ax2.set_ylabel("R2", fontsize=24)
    
    ax1.set_xticks(train_accuracy_x, x_labels, fontsize=20)
    
    ax1.legend(loc='best', bbox_to_anchor=(1.2, 0.5),fontsize=25, frameon=False)
    ax2.legend(loc='best', bbox_to_anchor=(1.2, 0.35),fontsize=25, frameon=False)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()
    return             

def plot_loss_accuracy_with_epoch(loss_recording, accuracy_recording, outfile):

    loss = np.mean(loss_recording,axis = (0,1))
    loss = loss[np.where(loss>0.0)]
    accuracy = np.mean(accuracy_recording, axis=(0,1))
    COLOR_ACCURACY = "#69b3a2"
    COLOR_LOSS = "#3399e6"
    epoch_x = np.array(range(len(accuracy)))
    batchsize = np.around(len(loss)/len(accuracy))

    accuracy_x = epoch_x * batchsize
    loss_x = np.array(range(len(loss))) 
    
    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_axes([0.1, 0.2, 0.9, 0.9])
    #fig, ax1 = plt.subplots(figsize=(24, 8))
    ax2 = ax1.twinx()

    ax1.plot(loss_x, loss, color=COLOR_LOSS, lw=1)
    ax1.set_yscale('log')
    ax2.plot(accuracy_x, accuracy, color=COLOR_ACCURACY, lw=3)

    x_labels = [str(i) for i in epoch_x]
    ax1.set_xlabel("Epoch",fontsize=24)
    ax1.set_xticks(accuracy_x, x_labels, fontsize=20)
    ax1.set_ylabel("Loss", color=COLOR_LOSS, fontsize=24)
    ax1.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax1.tick_params(axis='y',labelsize=20)

    ax2.set_ylabel("R2", color=COLOR_ACCURACY, fontsize=24)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACCURACY)
    ax2.tick_params(axis='y',labelsize=20)


    fig.suptitle("Loss and R2 vs Epoch", fontsize=32)

    fig.savefig(outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()
    return                                                                                                                                                                                                                                                                                                                

