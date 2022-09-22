import base64
from io import BytesIO
from typing import Callable, List

import numpy as np
import torch
import cv2
from .masks import face_mask_static 
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    """
    # from mtcnn import MTCNN
    # det_model = MTCNN()
    image, _ = app.get(image_full, crop_size)#, det_model)
    return image


def normalize_and_torch(image: np.ndarray, mode='') -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    if mode == 'train_real':

        image = torch.tensor(image.copy(), dtype=torch.float32)#.to(device)#.cuda()
    else:
        image = torch.tensor(image.copy(), dtype=torch.float32).to(device)  # .cuda()

    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray, mode='') -> torch.tensor:
    """
    Normalize batch images and transform to torch
    """

    if mode == 'train_real':

        batch_frames = torch.tensor(frames.copy(), dtype=torch.float32)#.to(device)  # .cuda()
    else:
        # print("hereeeeee")
        batch_frames = torch.tensor(frames.copy(), dtype=torch.float32).to(device)  # .cuda()

    # batch_frames = torch.from_numpy(frames.copy()).to(device)#.cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.
    
    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def get_final_image(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    handler) -> None:
    """
    Create final video from frames
    """
    final = full_frame.copy()
    params = [None for i in range(len(final_frames))]
    
    for i in range(len(final_frames)):
        frame = cv2.resize(final_frames[i][0], (224, 224))
        
        landmarks = handler.get_without_detection_without_transform(frame)     
        landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i][0])

        mask, _ = face_mask_static(crop_frames[i][0], landmarks, landmarks_tgt, params[i])
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t*swap_t + (1-mask_t)*final
    final = np.array(final, dtype='uint8')
    return final


def show_images(images: List[np.ndarray], 
                titles=None, 
                figsize=(20, 5), 
                fontsize=15):
    if titles:
        assert len(titles) == len(images), "Amount of images should be the same as the amount of titles"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image[:, :, ::-1])
        if titles:
            ax.set_title(titles[idx], fontsize=fontsize)
        ax.axis("off")