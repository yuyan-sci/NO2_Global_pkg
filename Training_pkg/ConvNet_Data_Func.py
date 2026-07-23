import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Training_pkg.utils import Data_Augmentation_Switch

if Data_Augmentation_Switch:
# For training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

# For validation - keep exactly what you have
val_transform = transforms.Compose([
    transforms.ToTensor(), 
])

class Dataset(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for training datasets. It is used for the global datasets, which is continuous data.
    '''
    def __init__(self, traindata, truedata):  # 'Initialization' Data Loading
        '''

        :param traindata:
            Training data.
        :param truedata:
            Ture data to learn.
        :param beginyear:
            The begin year.
        :param endyear:
            The end year.
        :param nsite:
            The number of sites. For example, for overall observation it is 10870.
        '''
        super(Dataset, self).__init__()
        self.traindatasets = torch.squeeze(torch.Tensor(traindata))  # Read training data from npy file
        self.truedatasets = torch.squeeze(torch.Tensor(truedata))

        print(self.truedatasets.shape)
        print(self.traindatasets.shape)
        self.transforms = train_transform  # 转为tensor形式
        self.shape = self.traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
        # Select sample
        traindata = self.traindatasets[index, :, :]
        truedata = self.truedatasets[index]
        return traindata, truedata
        # Load data and get label
    def __len__(self):  # 'Denotes the total number of samples'
        return self.traindatasets.shape[0]  # Return the total number of dataset

class Dataset_Val(torch.utils.data.Dataset):  # 'Characterizes a dataset for PyTorch'
    '''
    This class is for validation datasets/ estimation datasets
    '''
    def __init__(self, traindata):  # 'Initialization' Data Loading
            super(Dataset_Val, self).__init__()
            self.traindatasets = torch.Tensor(traindata) #torch.squeeze(torch.Tensor(traindata))
            print(self.traindatasets.shape)
            self.transforms = val_transform  # 转为tensor形式
            self.shape = self.traindatasets.shape
    def __getitem__(self, index):  # 'Generates one sample of data'
            # Select sample
            traindata = self.traindatasets[index, :, :]
            return traindata
            # Load data 
    def __len__(self):  # 'Denotes the total number of samples'
            return self.traindatasets.shape[0]  # Return the total number of datasets