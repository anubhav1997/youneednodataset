import cv2
import numpy as np
import os
import glob
import argparse

from deepface import DeepFace
from balanced_set import balanced_random, get_dataloader_oversampling_balanced, get_balanced_batch_latent_opt, train_latent_classifier, mutate, generate, mutate_z, generate_z, mutate_new, rejection_sampling
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--target_race', default='black', type=str)
parser.add_argument('--model_name', default='stylegan2', type=str)
parser.add_argument('--starting_seed', default=101, type=int)
parser.add_argument('--outdir', default='/vast/aj3281/generated_images_new_w/', type=str)
parser.add_argument('--mode', default= 'explore', type = str)
parser.add_argument('--start_iter', default=0, type = int)
parser.add_argument('--end_iter', default= 11000, type = int)
parser.add_argument('--ending_seed', default=sys.maxsize, type = int)

args = parser.parse_args()

network_pkl = None
model_name = args.model_name
if model_name == 'stylegan2' or model_name == 'stylegan2_ffhq':
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
elif model_name == 'stylegan2_celeba':
    network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl"
elif model_name == 'stylegan3':
    network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024' \
              '.pkl'

identities=['black', 'indian', 'white', 'latino hispanic', 'asian', 'middle eastern']


target_race = args.target_race

if args.target_race == 'latino':
    target_race = 'latino hispanic'
if args.target_race == 'middle_eastern':
    target_race = 'middle eastern'


if args.mode == 'explore':
    if network_pkl is not None:
        mutate_new(target_race, network_pkl, outdir=args.outdir, starting_seed=args.starting_seed, ending_seed = args.ending_seed)
    else:
        print("This GAN model has not been incorporated yet. ")

elif args.mode == 'explore_z':
    if network_pkl is not None:
        print("here")
        mutate_z(target_race, network_pkl, starting_seed=args.starting_seed, ending_seed = args.ending_seed)
    else:
        print("This GAN model has not been incorporated yet. ")

elif args.mode == 'generate':
    if network_pkl is not None:
        generate(target_race, network_pkl, outdir=args.outdir, start_iter = args.start_iter, end_iter = args.end_iter)
    else:
        print("This GAN model has not been incorporated yet. ")

elif args.mode == 'generate_z':
    if network_pkl is not None:
        generate_z(target_race, network_pkl, start_iter = args.start_iter, end_iter = args.end_iter)
    else:
        print("This GAN model has not been incorporated yet. ")

elif args.mode == 'rejection_sampling':
    if network_pkl is not None:
        rejection_sampling(network_pkl, target_race, n_samples=1000)
    else:
        print("This GAN model has not been incorporated yet. ")
