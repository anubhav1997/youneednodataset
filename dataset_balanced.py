import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import tensorflow as tf
import glob
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from simswap import simswap_init, simswap
from sberswap import sberswap_init, sberswap
from balanced_set import get_labels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import pandas as pd


class DatasetBalanced(Dataset):
    def __init__(self):
        ffhq_path = '/scratch/aj3281/ffhq-dataset/train/'
        self.path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))
        self.labels = []
        self.label_to_index = {'black': 0, 'indian': 1, 'white': 2, 'latino hispanic': 3, 'asian': 4,
                               'middle eastern': 5}
        for i in range(path_list):
            I = cv2.imread(self.path_list[i])
            race, _, _ = get_labels(I)
            self.labels.append(label_to_index[race])

        dict = {'Filepath': self.path_list, 'Label': self.labels}
        df = pd.DataFrame(dict)
        df.to_csv('FFHQ_ethnicity_classifications.csv')

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        I = cv2.imread(self.path_list[idx])
        label = self.labels[idx]
        # label =
        return I, label
