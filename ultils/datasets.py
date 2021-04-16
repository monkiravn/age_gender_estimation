import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class ImdbDataset(Dataset):
    def __init__(self,dataframe_path, data_root_path, transform=None):
        self.data_path = dataframe_path
        self.data_root = data_root_path

        # Loading dataframe
        df = pd.read_csv(self.data_path)

        # Setting labels
        file_path = df['filename']
        label_age = df['age']
        label_gender = df['gender']

        self.x = file_path
        self.age_y = label_age
        self.gender_y = label_gender

            # Applying transformation
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
          idx = idx.tolist()
        image_path = os.path.join(self.data_root, self.x.iloc[idx])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype('float') / 255.0
        label1 = np.array([self.age_y.iloc[idx]]).astype('float') / 100.0
        label2 = np.array([self.gender_y.iloc[idx]]).astype('float')

        sample = {'image': np.uint8(image), 'label_age': label1,
                  'label_gender': label2, }

        # Applying transformation
        if self.transform:
            sample = self.transform(sample)

        return sample