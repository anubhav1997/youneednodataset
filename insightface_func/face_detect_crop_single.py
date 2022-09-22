from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align
from insightface.app import FaceAnalysis
from mtcnn import MTCNN

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.det_model = MTCNN()
        self.mode = 'None'


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.mode = mode


    def get(self, img, crop_size, max_num=0):

        result = self.det_model.detect_faces(img)
        confs = [r['confidence'] for r in result]
        if len(result) == 0:
            return None, None

        best_index = np.argmax(confs)
        bbox = result[best_index]['box']
        kps_list = list(result[best_index]['keypoints'].values())
        kps = []
        for k in kps_list:

            kps.append(list(k))

        kps = np.array(kps)

        M, _ = face_align.estimate_norm(kps, crop_size, mode = self.mode)
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        return [align_img], [M]
