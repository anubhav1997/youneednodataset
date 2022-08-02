'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-23 17:08:08
Description: 
'''
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='models/BEST_checkpoint.tar', help="run ONNX model via TRT")
        # self.parser.add_argument("--pic_a_path", type=str, default='./crop_224/gdg.jpg', help="Person who provides identity information")
        # self.parser.add_argument("--pic_b_path", type=str, default='./crop_224/zrf.jpg', help="Person who provides information other than their identity")
        # self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
        self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
        # self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
        self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
        # self.parser.add_argument("--output_path", type=str, default='./output/', help="results path")
        self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
        self.parser.add_argument('--no_simswaplogo', action='store_true', help='Remove the watermark')
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        # self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        
        self.isTrain = False

        self.parser.add_argument('--batch_size', default=6, type=int)
        self.parser.add_argument('--crop_size', default=224, type=int, help="Don't change this")
        # self.parser.add_argument('--swap_model', default='simswap', type=str)
        self.parser.add_argument('--mode', default='test', type=str)
        self.parser.add_argument('--test_model_path', default='test', type=str)
        self.parser.add_argument('--test_dataset', default='celeba-hq', type=str)
        self.parser.add_argument('--finetune_dataset', default='celeba-hq', type=str)
        self.parser.add_argument('--test_swap_model', default='sberswap', type=str)
        self.parser.add_argument('--full_test_model_path', default=None, type=str)
        self.parser.add_argument('--num_workers', default=5, type=int)
        self.parser.add_argument('--save_model_suffix', default="", type=str)
        self.parser.add_argument('--n_steps', default=99999 // 12, type=int)
        self.parser.add_argument('--checkpoint_path', default=None, type=str)
        self.parser.add_argument('--detector_model_path', default='/scratch/aj3281/youneednodataset/models_simswap/26.pth', type=str)
        # parser.add_argument('--label', default=0, type=int)
        self.parser.add_argument('--data_dir', default='/scratch/aj3281/ffhq-dataset/test/65000/', type=str)
        self.parser.add_argument('--file_format', default='png', type=str)
        self.parser.add_argument('--swap_model', default=None, type=str)
        self.parser.add_argument('--n_samples', default=10, type=int)
        self.parser.add_argument('--dataset_name', default='', type=str)

