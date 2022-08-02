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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class DatasetFinetune(Dataset):
    def __init__(self, dataset_name, swap_model):
        self.swap_model = swap_model
        # self.path_list = glob.glob(os.path.join())
        # finetune_dataset = args.finetune_dataset  # 'celeba-hq'
        #
        # if self.swap_model == 'sberswap':
        #     model_sberswap, handler, netArc, G_sberswap, app_sberswap = sberswap_init()
        # if self.swap_model == 'simswap':
        #     spNorm, model_simswap, app, net = simswap_init()
        #     self.spNorm = spNorm
        #     self.model_simswap = model_simswap
        #     self.app = app
        #     self.net = net
        #     # self.det_model = det_model

        if dataset_name == 'ffhq':
            ffhq_path = '/scratch/aj3281/ffhq-dataset/train/'

            self.path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))

        if dataset_name == 'celeba-hq':
            path = '/scratch/aj3281/celebA-HQ-dataset-download/data1024x1024/train/'

            self.path_list = glob.glob(os.path.join(path, "*.jpg"))

    def __len__(self):
        return len(self.path_list) * 2 - 1

    def __getitem__(self, idx):
        # print(idx)
        if idx % 2 == 0:
            plt.close()
            plt.imshow(cv2.imread(self.path_list[idx // 2]))
            plt.show()
            return torch.Tensor(cv2.imread(self.path_list[idx // 2])).permute(2, 0, 1), 1 #torch.tensor(1).long()
        else:
            # print("trying swapping")
            swapped = None

            # if self.swap_model == 'sberswap':
            #     model_sberswap, handler, netArc, G_sberswap, app_sberswap = sberswap_init()
            # if self.swap_model == 'simswap':
            #     spNorm, model_simswap, app, net = simswap_init()
            count = 0
            # print("First step towards swapping done")
            while swapped is None:
                # print("First try at swapping")
                img1 = cv2.imread(self.path_list[np.random.randint(len(self.path_list))])
                img2 = cv2.imread(self.path_list[np.random.randint(len(self.path_list))])
                # print("imgs read")
                if self.swap_model == 'sberswap':
                    swapped = sberswap(img1, img2, model_sberswap, handler, netArc,
                                       G_sberswap, app_sberswap, mode) #[..., ::-1]
                elif self.swap_model == 'simswap':
                    swapped = simswap(img1, img2, spNorm, model_simswap, app, net, mode) #, self.det_model)
                # print('count', count)
                count+=1

            # print("swapped shape", swapped.shape)
            # swapped = swapped[..., ::-1]
            # print("afetr op shape", swapped.shape)
            print("done with swapping")
            plt.close()
            plt.imshow(swapped)
            plt.show()
            return torch.Tensor(swapped).permute(2,0,1), 0
            # return torch.from_numpy(swapped.copy()).permute(2, 0, 1), 0 #torch.tensor(0).long()
        return -1, -1
