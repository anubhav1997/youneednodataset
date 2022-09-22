import cv2
from PIL import Image
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import torch
import torch.nn.functional as F
from insightface_func.face_detect_crop_single import Face_detect_crop
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def simswap_init(mode):
    # print("Inside init func")
    # transformer_Arcface = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    opt = TestOptions().parse()
    # print("parsed test options")

    # opt.output_path = './output/'
    opt.temp_path = './tmp'
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    opt.use_mask = True  ## new feature up-to-date
    opt.no_simswaplogo = True

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    if mode != 'train_real':
        model = model.to(device)

    # print("got model")

    spNorm = SpecificNorm(mode=mode)

    # print("sp norm done")
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    n_classes = 19
    # print("here")
    net = BiSeNet(n_classes=n_classes)  # .to(device)
    if mode != 'train_real':
        net = net.to(device)
    # model = nn.DataParallel(model)
    # net = nn.DataParallel(net)
    # from mtcnn import MTCNN
    # det_model = MTCNN()
    return spNorm, model, app, net  # , det_model


def simswap(img_a_whole, img_b_whole, spNorm, model, app, net, mode):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    crop_size = 224
    # app.prepare(ctx_id= 0, det_thresh=det_threshold, det_size=(640,640))#)(1024, 1024)
    use_mask = True
    no_simswaplogo = True
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    output_path = './output/'
    # print("inside simswap")

    with torch.no_grad():
        # print("here 1 ")
        # from mtcnn import MTCNN
        # det_model = MTCNN()

        # print("Inside Simswap:", img_a_whole.shape)
        # try:

        img_a_align_crop, _ = app.get(img_a_whole, crop_size)#, det_model)
        # except:
        #     return None

        # print("here 2 ")
        if img_a_align_crop is None:
            return None
        # print("got crop aligned")
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))

        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        if mode != 'train_real':
            img_id = img_id.to(device)

        # create latent id
        img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
        latend_id = model.netArc(img_id_downsample)

        latend_id = F.normalize(latend_id, p=2, dim=1)

        if mode == 'train_real':
            latend_id = latend_id.cpu()

        ############## Forward Pass ######################

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)#, det_model)
        if (img_b_align_crop_list is None):
            return None
        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:
            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...]
            if mode != 'train_real':
                b_align_crop_tenor = b_align_crop_tenor.to(device)
            else:
                b_align_crop_tenor = b_align_crop_tenor.cpu()
            swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if use_mask:
            if mode != 'train_real':
                net.to(device)

            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')

            if mode != 'train_real':

                net.load_state_dict(torch.load(save_pth, map_location=device))
            else:
                net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
            net.eval()
        else:
            net = None

        final_image = reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole,
                                         logoclass, mode, \
                                         os.path.join(output_path, 'result_whole_swapsingle.jpg'), no_simswaplogo,
                                         pasring_model=net, use_mask=use_mask, norm=spNorm)

    return final_image
