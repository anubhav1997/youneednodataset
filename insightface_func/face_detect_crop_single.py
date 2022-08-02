'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:46:04
Description: 
'''
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
        # self.models = {}
        # print("inside face detect crop init")
        # root = os.path.expanduser(root)
        # onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        # onnx_files = sorted(onnx_files)
        # for onnx_file in onnx_files:
        #     if onnx_file.find('_selfgen_')>0:
        #         #print('ignore:', onnx_file)
        #         continue
        #     model = model_zoo.get_model(onnx_file)
        #     if model.taskname not in self.models:
        #         print('find model:', onnx_file, model.taskname)
        #         self.models[model.taskname] = model
        #     else:
        #         print('duplicated model task type, ignore:', onnx_file, model.taskname)
        #         del model
        # assert 'detection' in self.models
        # self.det_model = self.models['detection']
        self.det_model = MTCNN()
        # print('MTCNN initialized')
        self.mode = 'None'
        # self.det_model = FaceAnalysis(name='antelopev2') #allowed_modules=['detection']
        # self.models['detection'] = self.det_model

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        # self.det_thresh = det_thresh
        self.mode = mode
        # assert det_size is not None
        # print('set det-size:', det_size)
        # self.det_size = det_size
        # for taskname, model in self.models.items():
        #     if taskname=='detection':
        #         model.prepare(ctx_id, input_size=det_size)
        #     else:
        #         model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        # print(img)
        #
        # bboxes, kpss = self.det_model.detect(img,
        #                                      threshold=self.det_thresh,
        #                                      max_num=max_num,
        #                                      metric='default')

        # print("Inside insightface", img.shape)

        result = self.det_model.detect_faces(img)
        confs = [r['confidence'] for r in result]
        if len(result) == 0:
            return None, None

        best_index = np.argmax(confs)
        bbox = result[best_index]['box']
        kps_list = list(result[best_index]['keypoints'].values())
        kps = []
        for k in kps_list:
            # kps = np.append(kps, k, axis=1)
            kps.append(list(k))
        # kps = kps[1:]
        kps = np.array(kps)
        #
        # print('bboxes', bboxes)
        # print('kpss', kpss)
        # if bboxes.shape[0] == 0:
        #     # return None
        #     bboxes, kpss = self.det_model.detect(img,
        #                                          threshold=0.10,#self.det_thresh/4.0,
        #                                          max_num=max_num,
        #                                          metric='default')
        #     print("bboxes shape 2", bboxes.shape)
        #
        #     if bboxes.shape[0] == 0:
        #         return None

        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        # for i in range(bboxes.shape[0]):
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        # det_score = bboxes[..., 4]
        #
        # # select the face with the hightest detection score
        # best_index = np.argmax(det_score)
        # print("best_index", best_index)
        # # print()
        # kps = None
        # if kpss is not None:
        #     kps = kpss[best_index]

        M, _ = face_align.estimate_norm(kps, crop_size, mode = self.mode)
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        # print(align_img, M)
        return [align_img], [M]
