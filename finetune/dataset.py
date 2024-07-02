import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import wfdb
from scipy.io import loadmat

'''
In this code:
PTB-XL has four subset: superclass, subclass, form, rhythm
ICBEB is CPSC2018 dataset mentioned in the original paper
Chapman is the CSN dataset from the original paper
'''

class ECGDataset(Dataset):
    def __init__(self, data_path, csv_file, mode='train', dataset_name='ptbxl', backbone='resnet18'):
        """
        Args:
            data_path (string): Path to store raw data.
            csv_file (string): Path to the .csv file with labels and data path.
            mode (string): ptbxl/icbeb/chapman.
        """
        self.dataset_name = dataset_name

        if self.dataset_name == 'ptbxl':
            self.labels_name = list(csv_file.columns[6:])
            self.num_classes = len(self.labels_name)

            self.data_path = data_path
            self.ecg_path = csv_file['filename_hr']
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 6:].values
            
        elif self.dataset_name == 'icbeb':
            self.labels_name = list(csv_file.columns[7:])
            self.num_classes = len(self.labels_name)

            self.data_path = data_path
            self.ecg_path = csv_file['ecg_id'].astype(str)
            # in icbeb, the column 0-6 is other meta data, the column 7-end is the label
            self.labels = csv_file.iloc[:, 7:].values

        elif self.dataset_name == 'chapman': 
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)

            self.data_path = data_path
            self.ecg_path = csv_file['ecg_path'].astype(str)
            # in icbeb, the column 0-6 is other meta data, the column 7-end is the label
            self.labels = csv_file.iloc[:, 3:].values

        else:
            raise ValueError("dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if self.dataset_name == 'ptbxl':
            ecg_path = os.path.join(self.data_path, self.ecg_path[idx])
            # the wfdb format file include ecg and other meta data
            # the first element is the ecg data
            ecg = wfdb.rdsamp(ecg_path)[0]
            # the raw ecg shape is (5000, 12)
            # transform to (12, 5000)
            ecg = ecg.T

            ecg = ecg[:, :5000]
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            
        elif self.dataset_name == 'icbeb':
            ecg_path = os.path.join(self.data_path, self.ecg_path[idx])
            # icbeb has dat file, which is the raw ecg data
            ecg = wfdb.rdsamp(ecg_path)
            # the raw ecg shape is (n, 12), n is different for each sample
            # transform to (12, n)
            ecg = ecg[0].T
            # icbeb has different length of ecg, so we need to preprocess it to the same length
            # we only keep the first 2500 points as METS did
            ecg = ecg[:, :2500]
            
            # padding to 5000 to match the pre-trained data length
            ecg = np.pad(ecg, ((0, 0), (0, 2500)), 'constant', constant_values=0)
            ecg = ecg[:, :5000]

            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            
        elif self.dataset_name == 'chapman':
            # chapman ecg_path has / at the start, so we need to remove it
            ecg_path = os.path.join(self.data_path, self.ecg_path[idx][1:])
            # raw data is (12, 5000), do not need to transform
            ecg = loadmat(ecg_path)['val']
            ecg = ecg.astype(np.float32)

            ecg = ecg[:, :5000]
            
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()

        # switch AVL and AVF
        # In MIMIC-ECG, the lead order is I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
        # In downstream datasets, the lead order is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        ecg[[4, 5]] = ecg[[5, 4]]  

        

        return ecg, target

def getdataset(data_path, csv_path, mode='train', dataset_name='ptbxl', ratio=100, backbone='resnet18'):
    ratio = int(ratio)

    if dataset_name == 'ptbxl':
        csv = pd.read_csv(csv_path)
        if mode == 'train' and ratio != 100:
            csv, _ = train_test_split(csv, train_size=(ratio/100), random_state=42)
    elif dataset_name == 'icbeb':
        csv = pd.read_csv(csv_path)
        if mode == 'train' and ratio != 100:
            csv, _ = train_test_split(csv, train_size=(ratio/100), random_state=42)
    elif dataset_name == 'chapman':
        csv = pd.read_csv(csv_path)
        if mode == 'train' and ratio != 100:
            csv, _ = train_test_split(csv, train_size=(ratio/100), random_state=42)
    else:
        raise ValueError("dataset_name should be either 'ptbxl' or 'icbeb' or 'chapman!")
    
    csv.reset_index(drop=True, inplace=True)

    dataset = ECGDataset(data_path, csv, mode=mode, dataset_name=dataset_name,backbone=backbone)

    return dataset