import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import numpy as np
import torch
import legacy
import os
import random
#import required libs
import torch
import torch.nn
# from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable
# %matplotlib inline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--swap_model', default='simswap', type=str)
parser.add_argument('--epsilon', default=0.8, type=float)
parser.add_argument('--alpha', default=0.025, type=float)
parser.add_argument('--num_steps', default=10, type=int)
parser.add_argument('--seed', default=1, type=int)



# steps = 2000
args = parser.parse_args()

epsilon = args.epsilon
num_steps = args.num_steps
alpha = args.alpha

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq' \
              '-1024x1024.pkl'

network = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024' \
          '.pkl '

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)



def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m



def get_img(
        network_pkl: str,
        seeds: List[int],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        translate: Tuple[float, float],
        rotate: float,
        class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    # print('Loading networks from "%s"...' % network_pkl)
    #
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')
    arr = []
    # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    z = torch.from_numpy(np.random.RandomState(seeds[0]).randn(1, G.z_dim)).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB') #.save(f'{outdir}/seed{seed:04d}.png')
    # arr.append(img[0].cpu().numpy())
    return img, z

def get_img_from_z(z, truncation_psi, noise_mode, class_idx ):

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img



def get_img_batch(
        network_pkl: str,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        translate: Tuple[float, float],
        rotate: float,
        BATCH_SIZE: int,
        class_idx: Optional[int]
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    # print('Loading networks from "%s"...' % network_pkl)

    # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')
    # arr = []
    # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    z = torch.from_numpy(np.random.RandomState(seed).randn(BATCH_SIZE//2, G.z_dim), requires_grad=True).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    # plt.imshow(img[0].cpu().numpy())
    # plt.show()
    img = img.cpu().numpy()

    # arr.append(img[0].cpu().numpy())
    return img, z

# def attack_step(detector_model, generator_model, img, z):

swap_model ='simswap'
if swap_model == 'simswap':
    epoch = 26
    filename = 'models_' + swap_model + '/' + str(epoch) + '.pth'
else:
    epoch = 48
    filename = 'models_' + swap_model + '/' + str(epoch) + '.pth'


model = torch.load(filename, map_location=device).to(device)
model.eval()
G.eval()

# seeds = parse_range('1')
# print(seeds)

seeds = [args.seed]



y_true = Variable(torch.LongTensor([1]), requires_grad=False)   #tiger cat

#above three are hyperparameters



# label = torch.zeros([1, G.c_dim], device=device)
class_idx = None

label = torch.zeros([1, G.c_dim], device=device)
if G.c_dim != 0:
    if class_idx is None:
        raise click.ClickException('Must specify class label with --class when using a conditional network')
    label[:, class_idx] = 1
else:
    if class_idx is not None:
        print('warn: --class=lbl ignored when running on an unconditional network')



z = torch.from_numpy(np.random.RandomState(seeds[0]).randn(1, G.z_dim)).to(device)
z_variable = Variable(z, requires_grad=True)

w = G.mapping(z_variable, label, truncation_psi=random.uniform(0, 1))
w_variable = Variable(w, requires_grad=True)

with torch.no_grad():

    # img = get_img_from_z(z, truncation_psi=random.uniform(0, 1), noise_mode='const', class_idx=None)
    # w = G.mapping(z_variable, label, truncation_psi=0.7)
    img = G.synthesis(w, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    plt.imshow(img.squeeze(0).cpu())
    plt.savefig("input_adv.png")
    plt.show()
    output_adv = model.forward(img.permute(0, 3, 1, 2).float())
    print("Before optimizatiom", output_adv)



for i in range(num_steps):
    # z_grads = []
    if w_variable.grad is not None:
        w_variable.grad.zero_()
    # z_variable.zero_grad()
    # zero_gradients(z_variable)   #flush gradients
    w_variable.retain_grad()
    output1 = G.synthesis(w_variable, noise_mode='const')
    w_variable.retain_grad()
    output1 = (output1 * 127.5 + 128).clamp(0, 255) #.to(torch.uint8)

    # output1 = get_img_from_z(z_variable, truncation_psi=random.uniform(0, 1), noise_mode='const', class_idx=None)
    w_variable.retain_grad()
    # output1 = Variable(output1, requires_grad=True)
    #output1 = output1.permute(0, 3, 1, 2).float()
    output = model.forward(output1)  #perform forward pass
    # output = inceptionv3.forward(z_variable)
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, y_true.to(device))
    # w_variable.retain_grad()
    # w_variable.register_hook(lambda d: z_grads.append(d))
    loss_cal.backward()
    w_grad = alpha * torch.sign(w_variable.grad.data)   # as per the formula
    adv_temp = w_variable.data + w_grad                 #add perturbation to img_variable which also contains perturbation from previous iterations
    total_grad = adv_temp - z                  #total perturbation
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    print(torch.max(w), torch.min(w))
    w_adv = w + total_grad                      #add total perturbation to the original image

    w_variable.data = w_adv


img = G.synthesis(w_variable, noise_mode='const')
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# img = get_img_from_z(z_variable, truncation_psi=random.uniform(0, 1), noise_mode='const', class_idx=None)
output_adv = model.forward(img.permute(0, 3, 1, 2).float()) #
plt.imshow(img.squeeze(0).cpu())
plt.savefig("output_adv.png")
plt.show()

print("After optimization", output_adv)

#final adversarial example can be accessed at- img_variable.data
#
# output_adv = model.forward(z_variable)
# x_adv_pred = torch.max(output_adv.data, 1)[1][0]  #classify adversarial example
# output_adv_probs = F.softmax(output_adv, dim=1)
# x_adv_pred_prob =  round((torch.max(output_adv_probs.data, 1)[0][0]) * 100,4)
# visualize(image_tensor, img_variable.data, total_grad, epsilon, x_pred,x_adv_pred, x_pred_prob,  x_adv_pred_prob)  #class and prob of original ex will remain same