import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import tensorflow as tf
import PIL
import glob
import argparse
import os
from sklearn.svm import SVC
# from sklearn.mixture import GMM
from deepface import DeepFace
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torch.utils.data import TensorDataset, DataLoader
from stylegan3 import generate_images, generate_images_batch, parse_range, make_transform, parse_vec2, get_G
# from dataset_balanced import DatasetBalanced
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os.path
# import os
import random
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import re
from typing import List, Optional, Tuple, Union
# import click
import dnnlib
import legacy
import torch.nn
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
# import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable
# %matplotlib inline
# import argparse
from stylegan3_fun.projection import run_projection
from npy_append_array import NpyAppendArray
import time
from deepface.extendedmodels import Race
from deepface import DeepFace


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

network = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024' \
          '.pkl'
race_model = Race.loadModel()

class DatasetBalanced(Dataset):
    def __init__(self, finetune=False, finetune_count=12000):
        ffhq_path = '/scratch/aj3281/ffhq-dataset/train/'
        self.path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))
        self.labels = []
        self.label_to_index = {'black': 0, 'indian': 1, 'white': 2, 'latino hispanic': 3, 'asian': 4,
                               'middle eastern': 5}
        # if finetune:
        #     self.path_list = self.path_list[:finetune_count]

        if os.path.exists('FFHQ_ethnicity_classifications.csv'):
            df = pd.read_csv('FFHQ_ethnicity_classifications.csv')
            self.path_list = df['Filepath'].to_numpy()
            self.labels = df['Label'].to_numpy()
        else:
            for i in range(len(self.path_list)):
                I = cv2.imread(self.path_list[i])
                race, _, _ = get_labels(I)
                self.labels.append(self.label_to_index[race])

            dict = {'Filepath': self.path_list, 'Label': self.labels}
            df = pd.DataFrame(dict)
            df.to_csv('FFHQ_ethnicity_classifications.csv')

        if finetune:
            # random.shuffle(self.path_list)
            temp = list(zip(self.path_list, self.labels))
            random.shuffle(temp)
            self.path_list, self.labels = zip(*temp)

            self.path_list = self.path_list[:finetune_count]
            self.labels = self.labels[:finetune_count]
            print(self.path_list, self.labels)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):

        I = cv2.imread(self.path_list[idx])
        label = self.labels[idx]
        # label =
        return I, label


class DatasetLatentBalanced(Dataset):
    def __init__(self, n_samples=10000):
        self.G = get_G()
        self.G.eval()

        self.label = torch.zeros([1, self.G.c_dim], device=device)
        self.class_idx = None
        self.len = n_samples
        if self.G.c_dim != 0:
            if self.class_idx is None:
                raise click.ClickException('Must specify class label with --class when using a conditional network')
            self.label[:, class_idx] = 1
        else:
            if self.class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seed = random.randint(1, 9999)

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=0.7)

        img = G.synthesis(w, noise_mode='const')

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        img = img[0].detach().cpu().numpy()

        race, gender, age = get_labels(img)

        img = torch.flatten(w[0]).detach().cpu().numpy()
        return img, race, gender, age


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(ClassificationModel, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x = self.flatten(x)
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        # print(logits.shape)
        return nn.Softmax(dim=1)(logits)


def get_img_adversarial_optimization(label, model, epsilon=0.01, num_steps=20, alpha=0.025):
    G = get_G()
    model.eval()
    G.eval()

    label = Variable(torch.LongTensor([label]), requires_grad=False)
    seed = random.randint(1, 9999)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    z_variable = Variable(z, requires_grad=True)

    label_g = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(z_variable, label_g, truncation_psi=0.7)
    w_variable = Variable(w, requires_grad=True)

    for i in range(num_steps):
        # z_grads = []
        if w_variable.grad is not None:
            w_variable.grad.zero_()
        # z_variable.zero_grad()
        # zero_gradients(z_variable)   #flush gradients
        w_variable.retain_grad()
        # output1 = G.synthesis(w_variable, noise_mode='const')
        # w_variable.retain_grad()
        # output1 = (output1 * 127.5 + 128).clamp(0, 255)  # .to(torch.uint8)

        # w_variable.retain_grad()

        output = model.forward(torch.unsqueeze(torch.flatten(w_variable), 0).float())  # perform forward pass

        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, label.to(device))

        print("Optimization: Epoch: {} Loss: {}".format(i, loss_cal.item()))

        loss_cal.backward()
        w_grad = alpha * torch.sign(w_variable.grad.data)  # as per the formula
        adv_temp = w_variable.data + w_grad  # add perturbation to img_variable which also contains perturbation from previous iterations
        total_grad = adv_temp - z  # total perturbation
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)
        # print(torch.max(w), torch.min(w))
        w_adv = w + total_grad  # add total perturbation to the original image

        w_variable.data = w_adv

    img = G.synthesis(w_variable, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # print("printing img", img)
    # print("printing img shape", img.shape)

    return img[0].detach().cpu().numpy()


def compareList(l1, l2):
    return [i == j for i, j in zip(l1, l2)]


def train_latent_classifier(attribute='race', epochs=20, n_samples=100000):
    # model_temp = ClassificationModel(100, 6)
    # print("here")

    X, y_race, y_gender, y_age = get_labelled_data(n_samples)

    label_encoder = LabelEncoder()

    if attribute == 'race':
        y = label_encoder.fit_transform(y_race)
    elif attribute == 'gender':
        y = label_encoder.fit_transform(y_gender)
    elif attribute == 'age':
        y = label_encoder.fit_transform(y_age)
    else:
        print("Attribute is not supported.")
        exit(0)

    input_dims = len(X[0])
    n_classes = len(label_encoder.classes_)

    print(n_classes)
    print(input_dims)
    model = ClassificationModel(input_dims, n_classes)

    tensor_x = torch.tensor(X)
    tensor_y = torch.tensor(y)
    print("Tensor X shape", tensor_x.shape)
    print("Tensor Y shape", tensor_y.shape)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = model.to(device)

    for epoch in range(epochs):
        count = 0
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # print(X.shape, y.shape)

            model.zero_grad()
            pred = model(X)
            loss = criterion(pred, y.long())

            count += sum(compareList(torch.argmax(pred, 1).cpu().numpy(), y))

            loss.backward()
            optimizer.step()
            print(epoch, i, loss.item())

        print("Epoch {}, Accuracy : {}".format(epoch, count / float(len(X))))

        torch.save(model, "latent_space_classification_" + str(epoch) + ".pth")

    return model


def balanced_adv(model, batch_size=12, identities=['black', 'indian', 'white', 'latino hispanic', 'asian', 'middle '
                                                                                                           'eastern']):
    X = []
    n_samples = batch_size // len(identities)
    if batch_size % len(identities) != 0:
        n_samples += 1

    for i in range(len(identities)):
        for j in range(n_samples):
            seed = i * j
            img = get_img_adversarial_optimization(i, model)
            X.append(img)
    return X


def get_labels(img):
    obj = DeepFace.analyze(img, actions=['age', 'gender', 'race'], enforce_detection=False)[0]

    race = obj['dominant_race']
    gender = obj['dominant_gender']
    age = obj['age']

    return race, gender, age


def get_race(img):
    labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]
    preds = race_model(tf.expand_dims(tf.image.resize(tf.convert_to_tensor(img), (224, 224)), axis=0)).numpy()
    preds = np.argmax(preds)
    return labels[preds]

    # obj = DeepFace.analyze(img, actions=['age', 'gender', 'race'], enforce_detection=False)[0]
    #
    # race = obj['dominant_race']
    # gender = obj['dominant_gender']
    # age = obj['age']

    # return race, gender, age


def get_labelled_data(network_pkl=None, n_samples=100000):
    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    # G = get_G()
    # model.eval()
    G.eval()
    batch_size = 8
    X = []
    y_race = []
    y_gender = []
    y_age = []
    label = torch.zeros([1, G.c_dim], device=device)
    class_idx = None

    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    filename = 'w_stylegan2.npy'
    filename2 = 'y_stylegan2.npy'

    with NpyAppendArray(filename) as npaa, NpyAppendArray(filename2) as npaa2:

        for i in range(n_samples):
            print("INDEX: ", i)

            z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
            w = G.mapping(z, label, truncation_psi=0.7)
            # print("w shape", w.shape)
            img = G.synthesis(w, noise_mode='const')
            # print("img shape", img.shape)
            # img = G(z, label, truncation_psi=0.7, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # print(img.shape)
            img = img[0].detach().cpu().numpy()

            # images = generate_images_batch(network_pkl=network, seed=i, truncation_psi=0.7,
            #                                noise_mode='const',
            #                                outdir='./',
            #                                translate=parse_vec2('0,0'), rotate=0, BATCH_SIZE=batch_size,
            #                                class_idx=None)

            # for img in images:
            race, gender, age = get_labels(img)
            y_race.append(race)
            y_gender.append(gender)
            y_age.append(age)

            X.append(torch.flatten(w[0]).detach().cpu().numpy())
            npaa.append(torch.flatten(w[0]).detach().cpu().numpy())

            y = np.array([race, gender, age])
            # print(y)
            npaa2.append(y)

    with open('labelled_data_w_stylegan2.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y_race)
        np.save(f, y_gender)
        np.save(f, y_age)

    print("saved data")
    # print("exiting")
    # exit(0)
    # for i in range(np.unique(y_race)):
    #     print(y_race.count(i))

    labels = set(y_race)
    for label in labels:
        print(label, y_race.count(label))

    return X, y_race, y_gender, y_age


def rejection_sampling(network_pkl, target_race, n_samples=1000):
    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    # G = get_G()
    # model.eval()
    G.eval()
    batch_size = 8
    X = []
    y_race = []
    y_gender = []
    y_age = []
    label = torch.zeros([1, G.c_dim], device=device)
    class_idx = None

    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    filename = 'w_stylegan2_temp.npy'
    filename2 = 'y_stylegan2_temp.npy'
    start_time = time.time()
    curr_count = 0
    with NpyAppendArray(filename) as npaa, NpyAppendArray(filename2) as npaa2:

        # for i in range(n_samples):
        i =0
        while curr_count < n_samples:
            i+=1

            print("INDEX: ", i, curr_count)

            z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
            w = G.mapping(z, label, truncation_psi=0.7)
            # print("w shape", w.shape)
            img = G.synthesis(w, noise_mode='const')
            # print("img shape", img.shape)
            # img = G(z, label, truncation_psi=0.7, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # print(img.shape)
            img = img[0].detach().cpu().numpy()

            # images = generate_images_batch(network_pkl=network, seed=i, truncation_psi=0.7,
            #                                noise_mode='const',
            #                                outdir='./',
            #                                translate=parse_vec2('0,0'), rotate=0, BATCH_SIZE=batch_size,
            #                                class_idx=None)

            # for img in images:
            race = get_race(img)
            if race == target_race:
                curr_count +=1

            #
            # race, gender, age = get_labels(img)
            # y_race.append(race)
            # y_gender.append(gender)
            # y_age.append(age)

            # X.append(torch.flatten(w[0]).detach().cpu().numpy())
            # npaa.append(torch.flatten(w[0]).detach().cpu().numpy())
            #
            # y = np.array([race, gender, age])
            # # print(y)
            # npaa2.append(y)


    print("--- %s seconds ---" % (time.time() - start_time))

    # with open('labelled_data_w_stylegan2.npy', 'wb') as f:
    #     np.save(f, X)
    #     np.save(f, y_race)
    #     np.save(f, y_gender)
    #     np.save(f, y_age)
    #
    # print("saved data")
    # # print("exiting")
    # # exit(0)
    # # for i in range(np.unique(y_race)):
    # #     print(y_race.count(i))
    #
    # labels = set(y_race)
    # for label in labels:
    #     print(label, y_race.count(label))
    #
    #
    # return X, y_race, y_gender, y_age


def projection(network_pkl=None, dataset='ffhq'):
    if network_pkl is None:
        network_pkl = network

    w_vecs = []
    images = []
    labels = []
    path_list = []
    subset = 'Caucasian'
    if dataset == 'ffhq':

        ffhq_path = '/scratch/aj3281/ffhq-dataset/train/'
        path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))
    elif dataset == 'BUPT_transfer':
        path = '/scratch/aj3281/train/data/'

        if subset == 'Caucasian':

            folder_list = glob.glob(os.path.join(path + subset + '/*'))
            path_list = []
            for folder in folder_list:

                print(folder)
                # print(len(glob.glob(folder + '/*.jpg')))
                # path_list.append(glob.glob(folder + '/*.jpg')[0])
                for filename in glob.glob(folder + '/*.jpg'):
                    path_list.append(filename)
                    break

        else:

            path_list = glob.glob(os.path.join(
                path + subset + "/*.jpg"))  # glob.glob(os.path.join(path + "Asian/*.jpg")) + glob.glob(os.path.join(path + "African/*.jpg")) #

    print(path_list)
    print(len(path_list))

    path_list = path_list[118:]
    count = 0

    filename1 = 'projection_w_' + dataset + subset + '_stylegan2_256.npy'
    filename2 = 'projection_y_' + dataset + subset + '_stylegan2_256.npy'
    # path_list = path_
    with NpyAppendArray(filename1) as npaa, NpyAppendArray(filename2) as npaa2:

        for filename in path_list:
            print("Number of images done:", count)
            count += 1

            w, image = run_projection(network_pkl, filename)
            img = cv2.imread(filename)

            race, gender, age = get_labels(img)

            if dataset == 'BUPT_transfer':

                if subset == 'Caucasian':
                    race = filename.split('/')[-3].lower()
                else:
                    race = filename.split('/')[-2].lower()
                if race == 'african':
                    race = 'black'
                if race == 'caucasian':
                    race = 'white'

            y = np.array([race, gender, age])
            w = w.detach().cpu().numpy()
            labels.append(y)
            w_vecs.append(w)
            images.append(image)

            npaa.append(w)
            npaa2.append(y)

    with open('projection_' + dataset + subset + '_stylegan2_256.npy', 'wb') as f:
        np.save(f, w_vecs)
        np.save(f, labels)
        np.save(f, images)


# def train_classification_model(X, y, epochs, pretrained_path=None):
#
#     if pretrained is not None:
#         return torch.load(pretrained_path)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
#     model = ClassificationModel(X.shape[-1], np.unique(y)).to(device)
#     X = torch.Tensor(X)
#     y = torch.Tensor(y)
#     dataset = torch.utils.data.TensorDataset(X, y)
#     dataloader = torch.DataLoader(dataset, batch_size = 32)
#
#     for epoch in range(epochs):
#         for i, (data, labels) in enumerate(dataloader):
#             data, labels = data.to(device), labels.to(device)
#             model.zero_grad()
#             y_pred = model(data)
#             loss = criterion(y_pred, labels.long())
#             loss.backward()
#             optimizer.step()
#
#     return model


def train_SVC(X, y):
    clf = SVC(gamma='auto')
    clf = clf.fit(X, y)
    return clf


def train_GMM(X, y, covar_type='full'):
    n_classes = np.unique(y)

    clf = GMM(n_components=n_classes,
              covariance_type=covar_type, init_params='wc', n_iter=20)

    return clf


def balanced_random(seed, batch_size=12, category='race',
                    identities=['black', 'indian', 'white', 'latino hispanic', 'asian', 'middle eastern']):
    X = []
    n_samples = batch_size // len(identities)
    if batch_size % len(identities) != 0:
        n_samples += 1

    dict_counts = {}
    for i in identities:
        dict_counts[i] = 0

    i = seed
    while len(X) < batch_size:
        i += 1
        images = generate_images_batch(network_pkl=network, seed=i, truncation_psi=0.7,
                                       noise_mode='const',
                                       outdir='./',
                                       translate=parse_vec2('0,0'), rotate=0, BATCH_SIZE=batch_size,
                                       class_idx=None)

        for img in images:
            race, gender, age = get_labels(img)
            if dict_counts[race] > n_samples:
                continue
            else:
                dict_counts[race] += 1
                X.append(img)

    seed = i + 1
    return X, seed


def get_balanced_batch_latent_opt(latent_classifier, n_labels, batch_size=6):
    X = []
    n_samples = batch_size // n_labels
    if batch_size % n_labels != 0:
        n_samples += 1

    for i in range(n_labels):
        for j in range(n_samples):
            # seed = random.randint(1, 999)
            image = get_img_adversarial_optimization(i, latent_classifier, epsilon=0.1, num_steps=20, alpha=0.5)
            X.append(image)

    return X


def get_dataloader_oversampling_balanced(batch_size=12, finetune=False, finetune_count=12000):
    train_dataset = DatasetBalanced(finetune, finetune_count)
    labels = train_dataset.labels
    classes = np.unique(labels)
    class_sample_count = [0 for i in range(len(classes))]

    for l in labels:
        class_sample_count[l] += 1

    class_sample_count = np.array(class_sample_count)

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])  # added later - files with new are trained with this line
    samples_weight = np.array(samples_weight)

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                             len(samples_weight))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1,
                                               sampler=sampler)

    return train_loader


def get_starting_w(G, target_race, starting_seed=100):
    label = torch.zeros([1, G.c_dim], device=device)
    class_idx = None

    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    i = starting_seed
    race = ''

    #     while len(X) < batch_size:
    while race != target_race:
        i += 1

        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=0.7)

        img = G.synthesis(w, noise_mode='const')

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        img = img[0].detach().cpu().numpy()
        # print(img.shape, type(img))

        # _, _, _ = get_labels(cv2.imread('0_1_1_real1.png'))
        race, gender, age = get_labels(np.array(img))
        print(race)

    return w, i


def get_starting_z(G, target_race, starting_seed=100):
    label = torch.zeros([1, G.c_dim], device=device)
    class_idx = None

    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    i = starting_seed
    race = ''

    #     while len(X) < batch_size:
    while race != target_race:


        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=0.7)

        img = G.synthesis(w, noise_mode='const')

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        img = img[0].detach().cpu().numpy()
        # print(img.shape, type(img))

        # _, _, _ = get_labels(cv2.imread('0_1_1_real1.png'))
        race = get_race(np.array(img))

        # race, gender, age = get_labels(np.array(img))
        print(race, "finding seed")
        i += 1

    return z, i-1


def extract_data(data, flattened=False, model='stylegan3'):
    dims = 1
    if flattened:
        dims = 512

    len_ = 16 * dims
    if model == 'stylegan2':
        len_ = 18 * dims
    i = 0
    out = []
    while i < len(data):
        w = data[i:i + len_]
        i = i + len_

        out.append(w)

    return out


def get_mean_std(model_name="stylegan2"):
    len_ = 16
    if model_name == 'stylegan2':
        len_ = 18
        filename = "w_stylegan2.npy"
    else:
        filename = "w.npy"

    if os.path.exists(filename):

        data_temp = np.load(filename)
        print(data_temp.shape)
        data_temp = extract_data(data_temp, flattened=True, model=model_name)
        data_temp = np.array(data_temp)
        print(filename, data_temp.shape)
        data_temp = data_temp.reshape(-1, len_, 512)
    else:
        print("file not found: Running get_labelled_data to generate at least 1000 random samples. ")
        data_temp = get_labelled_data()

    mean = np.mean(data_temp, axis=0)
    std = np.std(data_temp, axis=0)
    mean = mean.reshape(len_, 512)
    std = std.reshape(len_, 512)
    return mean, std


def get_label_from_vec(w):
    img = G.synthesis(w, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].detach().cpu().numpy()

    race, gender, age = get_labels(img)

    return race, gender, age


def mutate(target_race, network_pkl, outdir='/vast/aj3281/generated_images_new2/', n_mutations=10,
           model_name='stylegan2', mul_factor_std=3, n_iterations=500, starting_seed=101):
    filename = outdir + '/w_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'
    filename2 = outdir + '/y_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'

    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    len_ = 16
    if model_name == 'stylegan2':
        len_ = 18
    # all_ = []
    # all_age = []
    # all_gender = []
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # with  as face_detection:
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9)

    if target_race == 'white':
        n_iterations = 100
    elif target_race == 'black' or target_race == 'middle eastern' or target_race == 'indian' or target_race == 'latino hispanic' or target_race == 'asian':
        n_iterations = 500
    _, std = get_mean_std(model_name)

    save_dir = f'{outdir}/{target_race}/'

    # path_list = path_
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # index = len(os.listdir(save_dir)) + 1

    with NpyAppendArray(filename) as npaa, NpyAppendArray(filename2) as npaa2:

        seed = starting_seed

        if os.path.exists(save_dir):
            print("getting old seed value start point")
            files = glob.glob(save_dir + '/*.png')
            seeds = [int(filename.split('/')[-1].split('_')[1]) for filename in files]
            seed = max(seeds) + 1
            print("Starting seed: ", seed)

        while True:
            # for seed in range(101, 1000):

            queue = []
            index = 0

            queue_write_fname = f'{outdir}/{target_race}_{seed}_queue.pt'

            w, seed = get_starting_w(G, target_race, starting_seed=seed)
            w = w.detach().cpu().numpy()
            seed += 1
            if os.path.isfile(queue_write_fname):
                queue = np.load(queue_write_fname)
                # w = np.load(filename)[:len_]
                # print(w.shape)

            else:
                for _ in range(n_mutations):
                    rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))
                    temp = w + rand_vec * std  # add some way of mutating the current vec
                    queue.append(temp)

            count_iter = 0
            count_no_face = 0
            while len(queue) != 0:

                if count_no_face > 20:
                    queue = []
                    if os.path.isfile(queue_write_fname):
                        os.remove(queue_write_fname)
                        # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                        with open(queue_write_fname, 'wb') as f:
                            np.save(f, np.array(queue))
                    break

                count_iter += 1
                vec = queue.pop(0)
                vec = torch.Tensor(vec).to(device)

                img = G.synthesis(vec, noise_mode='const')
                # print("img shape", img.shape)
                # img = G(z, label, truncation_psi=0.7, noise_mode='const')
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # print(img.shape)
                img = img[0].detach().cpu().numpy()

                results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.detections:
                    count_no_face += 1
                    print("No face detected")
                    continue

                try:
                    # print(img.shape, type(img))
                    race_out, gender, age = get_labels(img)
                    print(race_out)
                except:
                    count_no_face += 1
                    print("no face detected")
                    continue

                if race_out == target_race:
                    # print("here")
                    index += 1

                    PIL.Image.fromarray(img, 'RGB').save(
                        f'{outdir}/{target_race}/{str(index)}_{seed}_{age}_{gender}.png')
                    vec = vec.detach().cpu().numpy()

                    # all_.append(vec)
                    # all_age.append(age)
                    # all_gender.append(gender)

                    npaa.append(vec)
                    npaa2.append(np.array([race_out, gender, age]))

                    #             out = mutuate(vec)

                    if count_iter < n_iterations:

                        for _ in range(n_mutations):
                            rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))

                            temp = vec + rand_vec * std  # add some way of mutating the current vec
                            if np.linalg.norm(temp - w) > np.linalg.norm(vec - w):
                                queue.append(temp)

                # if index % 1000 == 0:

                if os.path.isfile(queue_write_fname):
                    os.remove(queue_write_fname)
                    # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                    with open(queue_write_fname, 'wb') as f:
                        np.save(f, np.array(queue))

    # return all_, all_gender, all_age
import sys



# improves upon old method to allow multiple experiments to run simultaneously using different seed brackets


def mutate_new(target_race, network_pkl, outdir='/vast/aj3281/generated_images_new_w/', n_mutations=3,
             model_name='stylegan2', mul_factor_std=3, n_iterations=1000000000, starting_seed=0,
             ending_seed=sys.maxsize):

    filename = outdir + '/w_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_' + str(starting_seed) + '_' + str(ending_seed) + '.npy'
    filename2 = outdir + '/w_y_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_' + str(starting_seed) + '_' + str(ending_seed) + '.npy'
    filename3 = outdir + '/w_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_' + str(starting_seed) + '_' + str(ending_seed) + '_all_w.npy'


    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    label = torch.zeros([1, G.c_dim], device=device)
    data_dict = get_augmentation_dict()
    len_ = 16
    if model_name == 'stylegan2':
        len_ = 18
    # all_ = []
    # all_age = []
    # all_gender = []
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # with  as face_detection:
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)


    if target_race == 'white':
        n_iterations = 1000
    elif target_race == 'black' or target_race == 'middle eastern' or target_race == 'indian' or target_race == 'latino hispanic' or target_race == 'asian':
        n_iterations = 5000
    # _, std = get_mean_std(model_name)

    save_dir = f'{outdir}/{target_race}/'

    # path_list = path_
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_dir = outdir + '/dataset_images/'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)


    # index = len(os.listdir(save_dir)) + 1

    with NpyAppendArray(filename) as npaa, NpyAppendArray(filename2) as npaa2, NpyAppendArray(filename3) as npaa3:

        seed = starting_seed

        while seed < ending_seed:
            # for seed in range(101, 1000):
            print("inside seed loop")
            queue = []
            index = 0

            queue_write_fname = f'{outdir}/{target_race}_{seed}_queue.pt'

            w, seed = get_starting_w(G, target_race, starting_seed=seed)
            w = w.detach().cpu().numpy()

            seed += 1
            print(queue_write_fname)

            queue.append(w)

            for _ in range(n_mutations):
                # rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))
                temp = w + np.random.uniform(low=-1*mul_factor_std/10, high=mul_factor_std/10, size=(len_, 512))  # rand_vec * std  # add some way of mutating the current vec
                queue.append(temp)

            count_iter = 0
            count_no_face = 0
            while len(queue) != 0:
                print("length of queue", len(queue))
                if count_no_face > 20:
                    queue = []
                    if os.path.isfile(queue_write_fname):
                        os.remove(queue_write_fname)
                        # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                        with open(queue_write_fname, 'wb') as f:
                            np.save(f, np.array(queue))
                    break

                count_iter += 1
                vec = queue.pop(0)
                vec = torch.Tensor(vec).to(device)

                # vec2 = G.mapping(vec, label, truncation_psi=0.7)
                img = G.synthesis(vec, noise_mode='const')
                # print("img shape", img.shape)
                # img = G(z, label, truncation_psi=0.7, noise_mode='const')
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # print(img.shape)
                img = img[0].detach().cpu().numpy()

                results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.detections:
                    count_no_face+=1
                    print("No face detected")
                    continue

                try:
                    # print(img.shape, type(img))
                    race_out, gender, age = get_labels(img)
                    print(race_out, "inside seed loop")
                except:
                    count_no_face += 1
                    print("no face detected")
                    continue

                if race_out == target_race:
                    print("hereeeeee")
                    index += 1

                    n_files = glob.glob(outdir + '/dataset_images/' + target_race + '*')
                    print("current n files", len(n_files))
                    if len(n_files) >= 50000:
                        print("Done with 50k files")
                        exit(0)

                    PIL.Image.fromarray(img, 'RGB').save(
                        f'{outdir}/{target_race}/{str(index)}_{seed}_{age}_{gender}.png')
                    vec = vec.detach().cpu().numpy()
                    generate_w_forone(w=vec, G=G, save_dir=outdir + '/dataset_images/', target_race=target_race,
                                      i=str(seed) + '_' + str(index), data=data_dict, npaa=npaa3)
                    print("generate mutations done")
                    # all_.append(vec)
                    # all_age.append(age)
                    # all_gender.append(gender)

                    npaa.append(vec)
                    npaa2.append(np.array([race_out, gender, age]))

                    #             out = mutuate(vec)

                    if count_iter < n_iterations:

                        for _ in range(n_mutations):
                            # rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))
                            temp = vec + np.random.uniform(low=-1*mul_factor_std/10, high=mul_factor_std/10,size=(len_, 512))
                            # temp = vec + np.random.normal(0, 0.5, size=(1, 512))  # + rand_vec * std # add some way of mutating the current vec
                            if np.linalg.norm(temp - w) > np.linalg.norm(vec - w):
                                queue.append(temp)

                # if index % 1000 == 0:

                if os.path.isfile(queue_write_fname):
                    os.remove(queue_write_fname)
                    # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                    with open(queue_write_fname, 'wb') as f:
                        np.save(f, np.array(queue))

    # return all_, all_gender, all_age


def mutate_z(target_race, network_pkl, outdir='/vast/aj3281/generated_images_new_z/', n_mutations=3,
             model_name='stylegan2', mul_factor_std=3, n_iterations=1000000000, starting_seed=0, ending_seed=sys.maxsize):


    filename = outdir + '/z_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations)  + '_' + str(starting_seed) + '_' + str(ending_seed) + '.npy'
    filename2 = outdir + '/z_y_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_' + str(starting_seed) + '_' + str(ending_seed) + '.npy'
    filename3 = outdir + '/z_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_' + str(starting_seed) + '_' + str(ending_seed) + '_all_w.npy'

    print("inside correct function")
    print(filename3)
    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    label = torch.zeros([1, G.c_dim], device=device)
    data_dict = get_augmentation_dict()

    save_dir = f'{outdir}/{target_race}/'

    # path_list = path_
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # index = len(os.listdir(save_dir)) + 1

    start_time = time.time()
    curr_count = 0

    with NpyAppendArray(filename) as npaa, NpyAppendArray(filename2) as npaa2, NpyAppendArray(filename3) as npaa3:
        # while curr_count < 1000:

        seed = starting_seed

        # if os.path.exists(save_dir):
        #     # print("getting old seed value start point")
        #     files = glob.glob(save_dir + '/*.png')
        #     seeds = [int(filename.split('/')[-1].split('_')[1]) for filename in files]
        #     if len(seeds) > 0:
        #         seed = max(seeds)
        # else:
        #     # if not os.path.exists(outdir):
        #     os.makedirs(save_dir)
        # seed = 0
        # if starting_seed > seed:
        #     seed = starting_seed
        # print("Starting seed: ", seed)

        # seed = starting_seed

        while seed < ending_seed and curr_count < 1000:
            # for seed in range(101, 1000):
            # print("inside seed loop")
            queue = []
            index = 0

            queue_write_fname = f'{outdir}/{target_race}_{seed}_queue.pt'

            w, seed = get_starting_z(G, target_race, starting_seed=seed)
            w = w.detach().cpu().numpy()

            seed += 1
            # print(queue_write_fname)

            # if os.path.isfile(queue_write_fname):
            #     queue = np.load(queue_write_fname)
                # w = np.load(filename)[:len_]
                # print(w.shape)

            # else:

            queue.append(w)

            for _ in range(n_mutations):
                # rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))
                temp = w + np.random.normal(0, 0.5, size=(1, 512))  # rand_vec * std  # add some way of mutating the current vec
                queue.append(temp)

            count_iter = 0
            count_no_face = 0
            while len(queue) != 0 and curr_count < 1000:
                # print("length of queue", len(queue))
                if count_no_face > 20:
                    queue = []
                    if os.path.isfile(queue_write_fname):
                        os.remove(queue_write_fname)
                        # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                        with open(queue_write_fname, 'wb') as f:
                            np.save(f, np.array(queue))
                    break

                count_iter += 1
                vec = queue.pop(0)
                vec = torch.Tensor(vec).to(device)

                vec2 = G.mapping(vec, label, truncation_psi=0.7)
                img = G.synthesis(vec2, noise_mode='const')
                # print("img shape", img.shape)
                # img = G(z, label, truncation_psi=0.7, noise_mode='const')
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # print(img.shape)
                img = img[0].detach().cpu().numpy()

                # results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # if not results.detections:
                #     count_no_face+=1
                #     print("No face detected")
                #     continue

                try:
                    # print(img.shape, type(img))
                    race_out = get_race(img)
                    # race_out, gender, age = get_labels(img)
                    # print(race_out, "inside seed loop")
                except:
                    count_no_face += 1
                    print("no face detected")
                    continue

                if race_out == target_race:
                    # print("hereeeeee")
                    index += 1
                    curr_count +=1
                    if curr_count >= 1000:
                        break

                    # n_files = glob.glob(outdir + '/dataset_images/' + target_race + '*')
                    # print("current n files", len(n_files))
                    # if len(n_files) >= 50000:
                    #     print("Done with 50k files")
                    #     exit(0)

                    # PIL.Image.fromarray(img, 'RGB').save(
                    #     f'{outdir}/{target_race}/{str(index)}_{seed}_{age}_{gender}.png')
                    vec = vec.detach().cpu().numpy()
                    # generate_z_forone(z=vec, G=G, save_dir=outdir + '/dataset_images/', target_race=target_race, i= str(seed) + '_' + str(index), data=data_dict, npaa=npaa3)
                    # print("generate mutations done")

                    # npaa.append(vec)
                    # npaa2.append(np.array([race_out, gender, age]))

                    #             out = mutuate(vec)

                    if count_iter < n_iterations:

                        for _ in range(n_mutations):
                            # rand_vec = np.random.uniform(low=-1 * mul_factor_std, high=mul_factor_std, size=(len_, 512))

                            temp = vec + np.random.normal(0, 0.5, size=(1, 512))  # + rand_vec * std # add some way of mutating the current vec
                            if np.linalg.norm(temp - w) > np.linalg.norm(vec - w):
                                queue.append(temp)

                # if index % 1000 == 0:

                # if os.path.isfile(queue_write_fname):
                #     os.remove(queue_write_fname)
                #     # torch.save(torch.Tensor(queue), f'{outdir}/{target_race}_queue.pt')
                #     with open(queue_write_fname, 'wb') as f:
                #         np.save(f, np.array(queue))
    print("--- %s seconds ---" % (time.time() - start_time))

    # return all_, all_gender, all_age


import numpy as np

LATENT_DIM = 512


def expression_augmentation(w, expression_analysis, scaling=3 / 4):
    new_covariates = []
    new_latents = []
    for direction, analysis in expression_analysis.items():
        #         print(direction, analysis)
        new_covariates.append(direction[1])
        normal = analysis["normal"]
        #         normal = torch.Tensor(analysis["normal"])
        # 1. Cancel neutral component
        # print(w.shape, normal.shape)
        x = w.dot(normal.T) * normal
        w_augmented = w - x
        # 2. Move towards expression using the mean distance computed on train set to ensure realistic outcome
        w_augmented += scaling * analysis["pos_stats"]["mean"] * normal
        new_latents.append(w_augmented)

    return np.concatenate(new_latents), np.stack(new_covariates)


def binary_augmentation(w, analysis, num_latents, scaling):
    neg_mean = analysis["neg_stats"]["mean"]
    pos_mean = analysis["pos_stats"]["mean"]

    extremum = max(-neg_mean, pos_mean)

    normal = analysis["normal"]
    #     print()
    x = np.linspace(-extremum, extremum, num_latents)

    scales = scaling * np.linspace(-extremum, extremum, num_latents)[:, None]
    scales = np.array(scales)
    w = np.array(w)
    # print(scales.shape, normal.shape, w.shape)  # , (w + scales * normal).shape)
    return [w + scales[i] * normal for i in range(num_latents)], scales


def pose_augmentation(w, pose_analysis, num_latents, scaling=3 / 4):
    return binary_augmentation(w, pose_analysis, num_latents, scaling)


def illumination_augmentation(w, illumination_analysis, num_latents, scaling=5 / 4):
    return binary_augmentation(w, illumination_analysis, num_latents, scaling)


def get_augmentation_dict(
        path='/scratch/aj3281/bob.paper.ijcb2021_synthetic_dataset/precomputed/latent_directions.pkl'):
    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)

    normal = data['expression'][('neutral', 'smile')]['normal']
    return data


def save_img_from_w(w, G, path):
    img = G.synthesis(w, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].detach().cpu().numpy()
    PIL.Image.fromarray(img, 'RGB').save(path)


def generate(target_race, network_pkl, outdir='/vast/aj3281/generated_images_new2/', num_latents=10,
             n_mutations=10, model_name='stylegan2', mul_factor_std=3, start_iter = 0, end_iter = 11000):
    filename = outdir + '/w_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'
    filename2 = outdir + '/y_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'

    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    len_ = 16
    if model_name == 'stylegan2':
        len_ = 18

    all_w_filename = outdir + '/w_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_all_w.npy'

    save_dir = f'{outdir}/dataset_images/{target_race}/'

    data = get_augmentation_dict()

    ws = np.load(filename)

    with NpyAppendArray(all_w_filename) as npaa:

        for i in range(start_iter, end_iter):

            if not os.path.exists(save_dir + '/' + str(i) + '/'):
                os.makedirs(save_dir + '/' + str(i) + '/')
            else:
                continue

            count = 0

            w = np.expand_dims(ws[i], 0)
            # print(w.shape)
            npaa.append(w.astype('float64'))
            count += 1
            w_expressions, new_convariates = expression_augmentation(w, data['expression'])
            # w_expressions_tensor = torch.Tensor(w_expressions)

            for j in range(len(w_expressions)):
                npaa.append(np.expand_dims(w_expressions[j], 0).astype('float64'))
                save_img_from_w(torch.Tensor(w_expressions[j]).unsqueeze(0), G, save_dir + f'/{i}/{count}.png')
                count += 1

            w_pose, new_convariates = pose_augmentation(w, data['pose'], num_latents)
            # w_pose_tensor = torch.Tensor(w_pose)

            for j in range(len(w_pose)):
                npaa.append(w_pose[j].astype('float64'))
                save_img_from_w(torch.Tensor(w_pose[j]), G, save_dir + f'/{i}/{count}.png')
                count += 1

            w_illumination, new_convariates = illumination_augmentation(w, data['illumination'], num_latents)

            # w_illumination_tensor = torch.Tensor(w_illumination)

            for j in range(len(w_illumination)):
                npaa.append(w_illumination[j].astype('float64'))
                save_img_from_w(torch.Tensor(w_illumination[j]), G, save_dir + f'/{i}/{count}.png')
                count += 1


def generate_z_forone(z, G, save_dir, target_race, i, data, npaa,  num_latents=20):
    label = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(torch.Tensor(z), label, truncation_psi=0.7)
    w = w.detach().cpu().numpy()

    generate_w_forone(w, G, save_dir, target_race, i, data, npaa,  num_latents=20)




def generate_w_forone(w, G, save_dir, target_race, i, data, npaa,  num_latents=20):

    if not os.path.exists(save_dir + '/' + target_race + '_' + str(i) + '/'):
        os.makedirs(save_dir + '/' + target_race + '_' + str(i) + '/')
    else:
        return


    count = 0
    w_expressions, new_convariates = expression_augmentation(w, data['expression'])
    # w_expressions_tensor = torch.Tensor(w_expressions)

    for j in range(len(w_expressions)):
        npaa.append(np.expand_dims(w_expressions[j], 0).astype('float64'))
        save_img_from_w(torch.Tensor(w_expressions[j]).unsqueeze(0), G, save_dir + f'/{target_race}_{i}/{count}.png')
        count += 1

    w_pose, new_convariates = pose_augmentation(w, data['pose'], num_latents)
    # w_pose_tensor = torch.Tensor(w_pose)

    for j in range(len(w_pose)):
        npaa.append(w_pose[j].astype('float64'))
        save_img_from_w(torch.Tensor(w_pose[j]), G, save_dir + f'/{target_race}_{i}/{count}.png')
        count += 1

    w_illumination, new_convariates = illumination_augmentation(w, data['illumination'], num_latents)

    # w_illumination_tensor = torch.Tensor(w_illumination)

    for j in range(len(w_illumination)):
        npaa.append(w_illumination[j].astype('float64'))
        save_img_from_w(torch.Tensor(w_illumination[j]), G, save_dir + f'/{target_race}_{i}/{count}.png')
        count += 1


def generate_z(target_race, network_pkl, outdir='/vast/aj3281/generated_images_new_z/', num_latents=20, n_identities=50000,
             n_mutations=3,
             model_name='stylegan2', mul_factor_std=3, start_iter = 0, end_iter = 11000):
    filename = outdir + '/z_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'
    filename2 = outdir + '/y_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '.npy'



    if network_pkl is None:
        network_pkl = network

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    label = torch.zeros([1, G.c_dim], device=device)

    len_ = 1

    all_w_filename = outdir + '/z_' + target_race + "_" + model_name + '_' + str(mul_factor_std) + '_' + str(
        n_mutations) + '_all_w.npy'

    save_dir = f'{outdir}/dataset_images/'

    data = get_augmentation_dict()

    zs = np.load(filename)

    with NpyAppendArray(all_w_filename) as npaa:

        for i in range(start_iter, end_iter):
            print("in the loop for ", i)

            if not os.path.exists(save_dir + '/' + target_race + '_' + str(i) + '/'):
                os.makedirs(save_dir + '/' + target_race + '_' + str(i) + '/')
            else:
                continue

            print("generating for: ", i)
            count = 0
            # print(zs[i].shape)
            z = np.expand_dims(zs[i], 0)
            # print(z.shape)
            w = G.mapping(torch.Tensor(z), label, truncation_psi=0.7)
            # print(w.shape)
            w = w.detach().cpu().numpy()

            # npaa.append(w.astype('float64'))
            count += 1
            w_expressions, new_convariates = expression_augmentation(w, data['expression'])
            # w_expressions_tensor = torch.Tensor(w_expressions)

            for j in range(len(w_expressions)):
                npaa.append(np.expand_dims(w_expressions[j], 0).astype('float64'))
                save_img_from_w(torch.Tensor(w_expressions[j]).unsqueeze(0), G, save_dir + f'/{target_race}_{i}/{count}.png')
                count += 1

            w_pose, new_convariates = pose_augmentation(w, data['pose'], num_latents)
            # w_pose_tensor = torch.Tensor(w_pose)

            for j in range(len(w_pose)):
                npaa.append(w_pose[j].astype('float64'))
                save_img_from_w(torch.Tensor(w_pose[j]), G, save_dir + f'/{target_race}_{i}/{count}.png')
                count += 1

            w_illumination, new_convariates = illumination_augmentation(w, data['illumination'], num_latents)

            # w_illumination_tensor = torch.Tensor(w_illumination)

            for j in range(len(w_illumination)):
                npaa.append(w_illumination[j].astype('float64'))
                save_img_from_w(torch.Tensor(w_illumination[j]), G, save_dir + f'/{target_race}_{i}/{count}.png')
                count += 1
