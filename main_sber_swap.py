
import cv2
import torch
import time
import os

from utils.inference.image_processing import crop_face, get_final_image, show_images
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_single import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

# main model for generation
G = AEI_Net(backbone='unet', num_blocks=2, c_id=512)
G.eval()
G.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))
G = G.to(device)#.cuda()
if torch.cuda.is_available():
    G = G.half()

# arcface model to get face embedding
netArc = iresnet100(fp16=False)
netArc.load_state_dict(torch.load('arcface_model/backbone.pth', map_location=device))
netArc=netArc.to(device)#.cuda()
netArc.eval()

# model to get face landmarks
handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

# model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
use_sr = True
if use_sr:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.backends.cudnn.benchmark = True
    opt = TestOptions()
    #opt.which_epoch ='10_7'
    model = Pix2PixModel(opt)
    model.netG.train()


target_type = 'image'


source_path = 'examples/images/elon_musk.jpg'
target_path = 'examples/images/beckham.jpg'
# path_to_video = 'examples/videos/nggyup.mp4'

source_full = cv2.imread(source_path)
# OUT_VIDEO_NAME = "examples/results/result.mp4"
crop_size = 224 # don't change this


# check, if we can detect face on the source image

try:
    source = crop_face(source_full, app, crop_size)[0]
    source = [source[:, :, ::-1]]
    print("Everything is ok!")
except TypeError:
    print("Bad source images")

# read video
# if target_type == 'image':
target_full = cv2.imread(target_path)
full_frames = [target_full]

target = get_target(full_frames, app, crop_size)


batch_size = 40

START_TIME = time.time()

final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                                                   source,
                                                                                   target,
                                                                                   netArc,
                                                                                   G,
                                                                                   app,
                                                                                   set_target = False,
                                                                                   crop_size=crop_size,
                                                                                   BS=batch_size)

if use_sr:
    final_frames_list = face_enhancement(final_frames_list, model)

# if target_type == 'video':
#     get_final_video(final_frames_list,
#                     crop_frames_list,
#                     full_frames,
#                     tfm_array_list,
#                     OUT_VIDEO_NAME,
#                     fps,
#                     handler)
#
#     add_audio_from_another_video(path_to_video, OUT_VIDEO_NAME, "audio")
#
#     print(f'Full pipeline took {time.time() - START_TIME}')
#     print(f"Video saved with path {OUT_VIDEO_NAME}")
# else:
result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
cv2.imwrite('examples/results/result.png', result)

#@markdown #**Visualize Image to Image swap**

import matplotlib.pyplot as plt

show_images([source[0][:, :, ::-1], target_full, result], ['Source Image', 'Target Image', 'Swapped Image'], figsize=(20, 15))
plt.show()
