import numpy as np

# %matplotlib inline

import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# from torchvision import models

import argparse
import os
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib
from simswap import simswap_init, simswap
from sberswap import sberswap_init, sberswap
import tensorflow as tf

mode = 'train'
# from stylegan3 import generate_images, generate_images_batch, parse_range, make_transform, parse_vec2
matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument('--detector_model_path', default='/scratch/aj3281/youneednodataset/models_simswap/26.pth', type=str)
# parser.add_argument('--label', default=0, type=int)
parser.add_argument('--data_dir', default='/scratch/aj3281/ffhq-dataset/test/65000/', type=str)
parser.add_argument('--file_format', default='png', type=str)
parser.add_argument('--swap_model', default=None, type=str)
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--dataset_name', default='', type=str)

args = parser.parse_args()

tf.config.experimental.set_visible_devices([], 'GPU')

def attribute_image_features(algorithm, input, **kwargs):
    detector_model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=label,
                                              **kwargs
                                              )
    return tensor_attributions


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# net = torch.load('youneednodataset/models/old_models/train_real_simswap_ffhq_8_full.pth', map_location=device).to(device)
# net2 = torch.load('youneednodataset/models_simswap/26.pth', map_location=device).to(device)



# label = args.label


def get_grads(detector_model, input, label):
    saliency = Saliency(detector_model)
    grads = saliency.attribute(input, target=label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return grads


def get_ig(detector_model, input, label):
    ig = IntegratedGradients(detector_model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))
    return attr_ig


def get_ig_nt(detector_model, input, label):
    ig = IntegratedGradients(detector_model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                          nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return attr_ig_nt


def get_dl(detector_model, input, label):
    dl = DeepLift(detector_model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return attr_dl


def get_occ(detector_model, input, pred_label_idx):
    occlusion = Occlusion(detector_model)
    attributions_occ = occlusion.attribute(input,
                                           strides=(3, 8, 8),
                                           target=pred_label_idx,
                                           sliding_window_shapes=(3, 15, 15),
                                           baselines=0)
    return attributions_occ


def get_pred_label(detector_model, input):
    # print(args.detector_model_path)
    # net = net.to(device)
    detector_model = detector_model.to(device)
    output = detector_model(input.to(device))
    # output = F.softmax(output, dim=1)
    # print(output[0].shape)
    # print(type(output))
    prediction_score, pred_label_idx = torch.topk(output, 1)

    print('Predicted:', pred_label_idx[0], '(', prediction_score.squeeze().item(), ')')
    return pred_label_idx[0]


path = 'captum_' + str(args.swap_model) + '_' + args.data_dir.split('/')[3] + '_' + args.detector_model_path.split('/')[-1].split('.')[0] + '/'

if not os.path.exists(path):
    os.makedirs(path)

path_s = path + 'Saliency/'
if not os.path.exists(path_s):
    os.makedirs(path_s)
path_s_0 = path_s + '/0/'
if not os.path.exists(path_s_0):
    os.makedirs(path_s_0)

path_s_1 = path_s + '/1/'
if not os.path.exists(path_s_1):
    os.makedirs(path_s_1)

path_o = path + 'Occlusion'

if not os.path.exists(path_o):
    os.makedirs(path_o)
path_o_0 = path_o + '/0/'
if not os.path.exists(path_o_0):
    os.makedirs(path_o_0)

path_o_1 = path_o + '/1/'
if not os.path.exists(path_o_1):
    os.makedirs(path_o_1)



# if args.swap_model is None:
label = int(args.swap_model is None)

if args.swap_model == 'sberswap':
    # label = 1
    model_sberswap, handler, netArc, G_sberswap, app_sberswap = sberswap_init(mode)
if args.swap_model == 'simswap':
    # label = 1
    spNorm, model_simswap, app, net = simswap_init(mode)

if args.dataset_name == 'ffhq':
    path_dataset = './ffhq-dataset/test/'
    # path = '../ffhq_temp/'
    path_list = glob.glob(os.path.join(path_dataset + "*/*.png"))

if args.dataset_name == 'celeba-hq':
    path_dataset = './celebA-HQ-dataset-download/data1024x1024/test/'

    path_list = glob.glob(os.path.join(path_dataset, "*.jpg"))

for i in range(args.n_samples):

    if args.swap_model is not None:

        # face = None
        #
        # while face is None:

        # img1 = cv2.imread(path_list[np.random.randint(len(path_list))])
        # img2 = cv2.imread(path_list[np.random.randint(len(path_list))])
        images = []
        images.append(cv2.imread(path_list[i]))
        images.append(cv2.imread(path_list[i+1]))
        # print(img1.shape)
        # print(img2.shape)
        # print(type(img1), type(img2))
        # plt.imshow(img1)
        # plt.show()
        if args.swap_model == 'sberswap':
            face = sberswap(images[0], images[1], model_sberswap, handler, netArc,
                            G_sberswap, app_sberswap, mode)  # [..., ::-1]
        elif args.swap_model == 'simswap':
            face = simswap(images[0], images[1], spNorm, model_simswap, app, net, mode)  # , self.det_model)
        # plt.imshow(face)
        # plt.show()

        print(face)
        if face is None:
            continue
    else:
        face = cv2.imread(path_list[i])

    detector_model = torch.load(args.detector_model_path, map_location=device).to(device)

    input = torch.FloatTensor(face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    print(input.shape)
    pred_label_idx = get_pred_label(detector_model, input)
    input.requires_grad = True

    from captum.attr import IntegratedGradients
    from captum.attr import Saliency
    from captum.attr import DeepLift
    from captum.attr import NoiseTunnel
    from captum.attr import visualization as viz
    from captum.attr import Occlusion
    grads = get_grads(detector_model, input, label)
    fig_s = viz.visualize_image_attr(grads, face, method="blended_heat_map", sign="absolute_value",
                                     show_colorbar=True, title="Overlayed Gradient Magnitudes")

    # fig = plt.figure()
    fig_s[0].canvas.draw()
    o = np.array(fig_s[0].canvas.buffer_rgba())
    o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
    path_s = path + '/Saliency/'
    path_save = path_s + "/" + str(label) + "/" + path_list[i].split('/')[-1].split('.')[0] + '_' + str(
        pred_label_idx.detach().cpu().numpy()[0]) + '.png'
    print(path_save)

    cv2.imwrite(path_save, o)
    # cv2.

    attributions_occ = get_occ(detector_model, input, pred_label_idx)
    fig_o = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        #                                       np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
        face,
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
        )
    # fig = plt.figure()
    fig_o[0].canvas.draw()
    o = np.array(fig_o[0].canvas.buffer_rgba())
    o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
    path_o = path + '/Occlusion'
    path_save = path_o + "/" + str(label) + "/" + path_list[i].split('/')[-1].split('.')[0] + '_' + str(
        pred_label_idx.detach().cpu().numpy()[0]) + '.png'
    print(path_save)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(o, aspect='auto')
    fig.savefig(path_save)

    # plt.imsave()
    # cv2.imwrite(path_save, o)

# original_image = plt.imread(pic_pth6) #np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
#
# _ = viz.visualize_image_attr(None, original_image,
#                       method="original_image", title="Original Image")
#
# print(net(input))
# print(net2(input))
#
# _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
#                           show_colorbar=True, title="Overlayed Integrated Gradients")
#
# _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value",
#                              outlier_perc=10, show_colorbar=True,
#                              title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
#
# _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True,
#                           title="Overlayed DeepLift")
#
# _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
#                                       ["original_image", "heat_map"],
#                                       ["all", "positive"],
#                                       show_colorbar=True,
#                                       outlier_perc=2,
#                                      )
