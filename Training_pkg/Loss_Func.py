import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Training_pkg.utils import *

class SelfDesigned_LossFunction(nn.Module):
    def __init__(self,losstype,size_average=None,reduce=None,reduction:str='mean')->None:
        super(SelfDesigned_LossFunction,self).__init__()
        self.Loss_Type = losstype
        self.reduction = reduction
        self.alpha = Adj_MSE_alpha
        self.beta  = Adj_MSE_beta
        self.GeoMSE_Lamba1_Penalty1 = GeoMSE_Lamba1_Penalty1
        self.GeoMSE_Lamba1_Penalty2 = GeoMSE_Lamba1_Penalty2
        self.GeoMSE_Gamma  = GeoMSE_Gamma
    def forward(self,model_output,target,geophysical_species,geophysical_mean,geophysical_std, LO_mean, LO_std):
        if self.Loss_Type == 'MSE':
            loss = F.mse_loss(model_output, target,reduction=self.reduction)
            print('MSE Loss: {}'.format(loss))
            return loss       
        elif self.Loss_Type == 'Adj_MSE':
            sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
            loss = F.mse_loss(sigmoid_coefficient*model_output,sigmoid_coefficient*target)
            print('Adj MSE loss: {}'.format(loss))
            return loss
        elif self.Loss_Type == 'GeoMSE':
            if normalize_bias or normalize_species:
                if LO_mean is None or LO_std is None:
                    raise ValueError("LO_mean and LO_std are required when normalization is enabled")
                geophysical_species = (geophysical_species * geophysical_std + geophysical_mean - LO_mean) / LO_std
            else:
                geophysical_species = geophysical_species * geophysical_std + geophysical_mean
            
            MSE_loss = F.mse_loss(model_output, target)
            Penalty1 = self.GeoMSE_Lamba1_Penalty1 * torch.mean(torch.relu(-model_output - geophysical_species)) # To force the model output larger than -geophysical_species
            Penalty2 = self.GeoMSE_Lamba1_Penalty2 * torch.mean(torch.relu(model_output - self.GeoMSE_Gamma * geophysical_species)) # To force the model output larger than -geophysical_species
            loss = MSE_loss + Penalty1 + Penalty2
            print('Total loss: {}, GeoMSE Loss: {}, Penalty 1: {}, Penalty 2: {}'.format(loss, MSE_loss, Penalty1, Penalty2))
            return loss
        elif self.Loss_Type == 'Adj_GeoMSE':
            sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target)))+1)
            if normalize_bias or normalize_species:
                if LO_mean is None or LO_std is None:
                    raise ValueError("LO_mean and LO_std are required when normalization is enabled")
                geophysical_species = (geophysical_species * geophysical_std + geophysical_mean - LO_mean) / LO_std
            else:
                geophysical_species = geophysical_species * geophysical_std + geophysical_mean
            #sigmoid_coefficient = torch.sqrt(self.beta * 1/(1+torch.exp(self.alpha*torch.square(target+geophysical)))+1)
            MSE_loss = F.mse_loss(sigmoid_coefficient*model_output,sigmoid_coefficient*target)
            Penalty1 = self.GeoMSE_Lamba1_Penalty1 * torch.mean(torch.relu(-model_output - geophysical_species)) # To force the model output larger than -geophysical_species
            Penalty2 = self.GeoMSE_Lamba1_Penalty2 * torch.mean(torch.relu(model_output - self.GeoMSE_Gamma * geophysical_species)) # To force the model output larger than -geophysical_species
            loss = MSE_loss + Penalty2 +  Penalty1
            print('Total loss: {}, Adj GeoMSE Loss: {}, Penalty 1: {}, Penalty 2: {}'.format(loss, MSE_loss, Penalty1, Penalty2))
            return loss
        
        elif self.Loss_Type == 'CrossEntropyLoss':
            loss = F.cross_entropy(model_output, target)
            print('CrossEntropyLoss: {}'.format(loss))
            return loss