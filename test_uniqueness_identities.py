from deepface import DeepFace
import torch
import numpy as np

import argparse
import os
import glob

import sys
from npy_append_array import NpyAppendArray

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--start_iter', default=0, type=int)

args = parser.parse_args()

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

start_iter = args.start_iter
end_iter = start_iter + 10000
all_filenames = glob.glob(os.path.join(args.data_dir, '*/0.png'))
if args.start_iter + 10000 > len(all_filenames):
    end_iter = len(all_filenames)

scores = []

outfilename = 'uniqueness_scores' + str(start_iter) + '_' + str(end_iter) + '_sface_w_old.npy'

outfilename2 = 'uniqueness_filenames_sface_w_old.npy'

with NpyAppendArray(outfilename) as npaa:  # , NpyAppendArray(outfilename2) as npaa2:

    for i in range(start_iter, end_iter):
        filename = all_filenames[i]
        # print(filename)

        for j in range(i + 1, len(all_filenames)):

            test_filename = all_filenames[j]

            try:
                if test_filename is not filename:

                    # face verification
                    obj = DeepFace.verify(img1_path=filename,
                                          img2_path=test_filename,
                                          model_name="SFace"
                                          )
                    dist = obj['distance']
                    if i == 1 and j == 1:
                        print(obj)
                    # print(dist, test_filename, filename)

                    npaa.append(np.array([dist]))
                    # npaa2.append(np.array([filename, test_filename]))
            except:
                continue
