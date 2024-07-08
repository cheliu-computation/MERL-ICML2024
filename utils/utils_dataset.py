import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from PIL import Image
import wfdb
from tqdm import tqdm
import os

# these two datasets will read the raw ecg

class Ori_MIMIC_E_T_Dataset(Dataset):
    def __init__(self, ecg_meta_path, transform=None, **args):
        self.ecg_meta_path = ecg_meta_path
        self.mode = args['train_test']
        self.text_csv = args['text_csv']
        self.record_csv = args['record_csv']
        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get ecg
        study_id = self.text_csv['study_id'].iloc[idx]
        if study_id == self.record_csv['study_id'].iloc[idx]:
            path = self.record_csv['path'].iloc[idx]
        else:
            print('Error: study_id not match!')
        path = os.path.join(self.ecg_meta_path, path)
        ecg = wfdb.rdsamp(path)[0]
        ecg = ecg.T

        # check nan and inf
        if np.isinf(ecg).sum() == 0:
            for i in range(ecg.shape[0]):
                nan_idx = np.where(np.isnan(ecg[:, i]))[0]
                for idx in nan_idx:
                    ecg[idx, i] = np.mean(ecg[max(0, idx-6):min(idx+6, ecg.shape[0]), i])
        if np.isnan(ecg).sum() == 0:
            for i in range(ecg.shape[0]):
                inf_idx = np.where(np.isinf(ecg[:, i]))[0]
                for idx in inf_idx:
                    ecg[idx, i] = np.mean(ecg[max(0, idx-6):min(idx+6, ecg.shape[0]), i])

        # noramlize
        ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

        # get raw text
        report = self.text_csv.iloc[idx][['report_0', 'report_1',
       'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7',
       'report_8', 'report_9', 'report_10', 'report_11', 'report_12',
       'report_13', 'report_14', 'report_15', 'report_16', 'report_17']]
        # only keep not NaN
        report = report[~report.isna()]
        # concat the report
        report = '. '.join(report)
        # preprocessing on raw text
        report = report.replace('EKG', 'ECG')
        report = report.replace('ekg', 'ecg')
        report = report.strip('*** ')
        report = report.strip(' ***')
        report = report.strip('***')
        report = report.strip('=-')
        report = report.strip('=')
        # convert to all lower case
        report = report.lower()

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample


class Ori_ECG_TEXT_Dsataset:

    def __init__(self, ecg_path, csv_path, dataset_name='mimic'):
        # if you use this dataset, please replace ecg_path from config.yaml to the 'your path/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
        self.ecg_path = ecg_path
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.csv = pd.read_csv(self.csv_path, low_memory=False)
        self.record_csv = pd.read_csv(os.path.join(self.ecg_path, 'record_list.csv'), low_memory=False)
        
        # sort and reset index by study_id
        self.csv = self.csv.sort_values(by=['study_id'])
        self.csv.reset_index(inplace=True, drop=True)
        self.record_csv = self.record_csv.sort_values(by=['study_id'])
        self.record_csv.reset_index(inplace=True, drop=True)

        # split train and val
        self.train_csv, self.val_csv, self.train_record_csv, self.val_record_csv = \
            train_test_split(self.csv, self.record_csv, test_size=0.02, random_state=42)
        # sort and reset index by study_id
        self.train_csv = self.train_csv.sort_values(by=['study_id'])
        self.val_csv = self.val_csv.sort_values(by=['study_id'])
        self.train_csv.reset_index(inplace=True, drop=True)
        self.val_csv.reset_index(inplace=True, drop=True)

        self.train_record_csv = self.train_record_csv.sort_values(by=['study_id'])
        self.val_record_csv = self.val_record_csv.sort_values(by=['study_id'])
        self.train_record_csv.reset_index(inplace=True, drop=True)
        self.val_record_csv.reset_index(inplace=True, drop=True)
        
        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')

    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        
        if self.dataset_name == 'mimic':
            
            if train_test == 'train':
                misc_args = {'train_test': train_test,
                   'text_csv': self.train_csv,
                   'record_csv': self.train_record_csv}
            else:
                misc_args = {'train_test': train_test,
                   'text_csv': self.val_csv,
                   'record_csv': self.val_record_csv}
            
        
            dataset = Ori_MIMIC_E_T_Dataset(ecg_data=self.ecg_path,
                                       transform=Transforms,
                                       **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset


# these two datasets will read the ecg from preprocessed npy file
# we suggest to use these two datasets for accelerating the IO speed


class MIMIC_E_T_Dataset(Dataset):
    def __init__(self, ecg_meta_path, transform=None, **args):
        self.ecg_meta_path = ecg_meta_path
        self.mode = args['train_test']
        if self.mode == 'train':
            self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_train.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')
            
        else:
            self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_val.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')


        self.text_csv = args['text_csv']

        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we have to divide 1000 to get the real value
        ecg = self.ecg_data[idx]/1000
        # ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
        

        # get raw text
        report = self.text_csv.iloc[idx]['total_report']

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample


class ECG_TEXT_Dsataset:

    def __init__(self, data_path, dataset_name='mimic'):
        self.data_path = data_path
        self.dataset_name = dataset_name

        print(f'Load {dataset_name} dataset!')
        self.train_csv = pd.read_csv(os.path.join(self.data_path, 'train.csv'), low_memory=False)
        self.val_csv = pd.read_csv(os.path.join(self.data_path, 'val.csv'), low_memory=False)

        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')
        print(f'total size: {self.train_csv.shape[0] + self.val_csv.shape[0]}')
        
    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        
            
        if self.dataset_name == 'mimic':
            
            if train_test == 'train':
                misc_args = {'train_test': train_test,
                   'text_csv': self.train_csv,
                   }
            else:
                misc_args = {'train_test': train_test,
                   'text_csv': self.val_csv,
                   }
            
        
            dataset = MIMIC_E_T_Dataset(ecg_meta_path=self.data_path,
                                       transform=Transforms,
                                       **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset
