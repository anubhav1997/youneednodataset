import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import tensorflow as tf
import glob
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from simswap import simswap_init, simswap
from sberswap import sberswap_init, sberswap
from stylegan3 import generate_images, generate_images_batch, parse_range, make_transform, parse_vec2
# !git clone https://github.com/neuralchen/SimSwap
# !cd SimSwap && git pull


# !pip install insightface==0.2.1 onnxruntime moviepy
# !pip install googledrivedownloader
# !pip install imageio==2.4.1


# !wget --no-check-certificate "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" -O antelope.zip
# !unzip ./antelope.zip -d ./insightface_func/models/

# !wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
# !wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
# !unzip ./checkpoints.zip  -d ./checkpoints
# !wget -P ./parsing_model/checkpoint https://github.com/neuralchen/SimSwap/releases/download/1.0/79999_iter.pth


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=12, type=int)
parser.add_argument('--crop_size', default=224, type=int, help="Don't change this")
parser.add_argument('--swap_model', default='simswap', type=str, required=True)
parser.add_argument('--mode', default='test', type=str, required=True)
# parser.add_argument('--test_model_path', default='', type=str)
parser.add_argument('--test_dataset', default='celeba-hq', type=str, help='[celeba-hq, ffhq, adfes, IJCB2017, pasc]')
parser.add_argument('--finetune_dataset', '--train_real_dataset', default='ffhq', type=str)
parser.add_argument('--test_swap_model', default='sberswap', type=str)
parser.add_argument('--full_test_model_path', default=None, type=str)
parser.add_argument('--num_workers', default=5, type=int)
parser.add_argument('--save_model_suffix', default="", type=str)
parser.add_argument('--n_steps', default=None, type=int) #2000
parser.add_argument('--checkpoint_path', default=None, type=str)


# steps = 2000
args = parser.parse_args()

BATCH_SIZE = args.batch_size  # 6
if args.n_steps is None:
    args.n_steps = 99999//BATCH_SIZE
det_threshold = 0.1
mode = args.mode  # 'test'

swap_model = args.swap_model  # 'sberswap'
test_swap_model = args.test_swap_model  # 'sberswap'
index = 101
num_classes = 2
image_size = args.crop_size
finetune_dataset = args.finetune_dataset
tf.config.experimental.set_visible_devices([], 'GPU')
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
network = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024' \
          '.pkl '

from deepface import DeepFace

races_count = {}
races_p = {}
races_n = {}
races_TP = {}
races_TN = {}
races_FN = {}
races_FP = {}


age_count = {}
age_p = {}
age_n = {}
age_TP = {}
age_TN = {}
age_FP = {}
age_FN = {}


gender_count = {}
gender_p = {}
gender_n = {}
gender_TP = {}
gender_TN = {}
gender_FP = {}
gender_FN = {}

def analyze_biases(images, y_pred, y):
    # index = 0
    # for img in images:
    for i in range(len(images)):
        # pred = preds[index]
        # index+=1
        # try:

        img = images[i]
        # print(img.shape)
        obj = DeepFace.analyze(img, actions=['age', 'gender', 'race'], enforce_detection=False, prog_bar = False)

        if obj['dominant_race'] in races_count.keys():
            races_count[obj['dominant_race']] += 1
        else:
            races_count[obj['dominant_race']] = 1
            races_TN[obj['dominant_race']] = 0
            races_FN[obj['dominant_race']] = 0
            races_FP[obj['dominant_race']] = 0
            races_TP[obj['dominant_race']] = 0
            races_n[obj['dominant_race']] = 0
            races_p[obj['dominant_race']] = 0




        if obj['gender'] in gender_count.keys():
            gender_count[obj['gender']] += 1
            # gender_TP[obj['gender']] += int(pred.numpy())
        else:
            gender_count[obj['gender']] = 1
            gender_TN[obj['gender']] = 0
            gender_FN[obj['gender']] = 0
            gender_FP[obj['gender']] = 0
            gender_TP[obj['gender']] = 0
            gender_n[obj['gender']] = 0
            gender_p[obj['gender']] = 0
            # gender_TP[obj['gender']] = int(pred.numpy())

        if obj['age'] <10:
            age = '<10'
        elif 10 <= obj['age'] < 20:
            age = '10-20'
        elif 20 <= obj['age'] < 30:
            age = '20-30'
        elif 30 <= obj['age'] < 40:
            age = '30-40'
        elif 40 <= obj['age'] < 50:
            age = '40-50'
        elif 50 <= obj['age'] < 60:
            age = '50-60'
        elif 60 <= obj['age'] < 70:
            age = '60-70'
        elif 70 <= obj['age'] < 80:
            age = '70-80'
        elif 80 <= obj['age'] < 90:
            age = '80-90'
        elif obj['age'] >= 90:
            age = '>90'

        if age in age_count.keys():
            age_count[age] += 1
            # age_TP[age] += int(pred.numpy())
        else:
            age_count[age] = 1
            age_TN[age] = 0
            age_FN[age] = 0
            age_FP[age] = 0
            age_TP[age] = 0
            age_n[age] = 0
            age_p[age] = 0
            # age_TP[age] = int(pred.numpy())

        if y[i] == 1:
            races_p[obj['dominant_race']] += 1
            age_p[age] += 1
            gender_p[obj['gender']] += 1

            if y_pred[i] == 1:
                races_TP[obj['dominant_race']] += 1
                age_TP[age] += 1
                gender_TP[obj['gender']] += 1
            elif y_pred[i] == 0:
                races_FN[obj['dominant_race']] += 1
                age_FN[age] += 1
                gender_FN[obj['gender']] += 1
        elif y[i] == 0:
            races_n[obj['dominant_race']] += 1
            age_n[age] += 1
            gender_n[obj['gender']] += 1

            if y_pred[i] == 0:
                races_TN[obj['dominant_race']] += 1
                age_TN[age] += 1
                gender_TN[obj['gender']] += 1
            elif y_pred[i] == 1:
                races_FP[obj['dominant_race']] += 1
                age_FP[age] += 1
                gender_FP[obj['gender']] += 1

                # return obj['dominant_race'], obj['dominant_age'], obj['dominant_gender']
        # except:
        #     print("No face detected by deepface")


def test2(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = []

        scores = []
        loss = 0
        count = 0
        for i in range(len(X)):
            y_pred_temp = model(X[i].unsqueeze(0).to(device))
            # print(y_pred_temp.shape)
            # print(y[i].shape)
            loss_temp = criterion(y_pred_temp, y[i].unsqueeze(0).long().to(device)).item()
            loss += loss_temp
            y_pred.append(torch.argmax(y_pred_temp, 1).cpu().numpy()[0])
            count += int(torch.argmax(y_pred_temp, 1) == y[i])
            scores.append(y_pred_temp.cpu().numpy())

        # acc = torch.sum(torch.argmax(y_pred,1)==y)/float(len(y))
        acc = count / float(len(y))
        loss_cal = loss / float(len(y))

    scores = np.array(scores)
    # print(scores.shape)
    scores = np.squeeze(scores, axis=1)
    # print(y_pred)

    return acc, loss_cal, len(y), scores, count, loss


def compareList(l1, l2):
    return [i==j for i, j in zip(l1, l2)]


def test_batch(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = []

        scores = []
        loss = 0
        count = 0
        # for i in range(len(X)):
        y_pred_temp = model(X.to(device))
        # print(y_pred_temp.shape)
        # print(y[i].shape)
        loss_temp = criterion(y_pred_temp, y.long().to(device)).item()
        loss += loss_temp
        y_pred.append(y_pred_temp)
        # print(torch.argmax(y_pred_temp, 1).cpu().numpy())
        # print(y)
        # print(sum(compareList(torch.argmax(y_pred_temp, 1).cpu().numpy(), y)))
        # z = input()

        count = sum(compareList(torch.argmax(y_pred_temp, 1).cpu().numpy(), y))
        analyze_biases(X.detach().cpu().permute(0, 2, 3, 1).numpy(), torch.argmax(y_pred_temp, 1).cpu().numpy(), y.cpu().numpy())
        scores = y_pred_temp.cpu().numpy()

        # acc = torch.sum(torch.argmax(y_pred,1)==y)/float(len(y))
        acc = count / float(len(y))
        loss_cal = loss / float(len(y))

    scores = np.array(scores)
    # print(scores.shape)
    # scores = np.squeeze(scores, axis=1)
    # print()


    return acc, loss_cal, len(y), scores, count, loss

def test(path_list, model):
    i = 0
    label_real = 1
    label_swap = 0
    acc_total = 0
    loss_total = 0
    count_total =0
    scores_total = []
    y_test_total = []
    while i < len(path_list):
        images = []
        swapped_images = []

        for j in range(i, min(i+BATCH_SIZE, len(path_list))):

            # y = []
            image1 = cv2.imread(path_list[j])
            image1 = cv2.resize(image1, (1024, 1024))
            # image2 = cv2.imread(path_list[j + 1])
            images.append(image1)
            # images.append(image2)

        for j in range(len(images)-1):
            if swap_model == 'sberswap':
                swapped = sberswap(images[j], images[j+1], model_sberswap, handler, netArc, G_sberswap, app_sberswap, mode)
            elif swap_model == 'simswap':
                swapped = simswap(images[j], images[j+1], spNorm, model_simswap, app, net, mode)

            if swapped is not None:
                # count += 1
                swapped_images.append(swapped)

        swapped_images = np.array(swapped_images)
        images = np.array(images)
        # print("Images", images.shape)
        # print("Swapped", swapped_images.shape)
        X_test = np.append(swapped_images, images, axis=0)

        y_test = np.append(np.zeros(len(swapped_images)), np.ones(len(images)))

        X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        y_test = torch.Tensor(y_test)

        _, _, count, scores, acc, loss = test_batch(model, X_test, y_test)
        acc_total += acc
        count_total += count
        loss_total += loss

        scores_total = np.append(scores_total, scores[:,1])
        y_test_total = np.append(y_test_total, y_test)

        i += BATCH_SIZE


    print("Total Count")
    print(races_count)
    print(gender_count)
    print(age_count)



    print("True Positives")
    print(races_TP)
    print(gender_TP)
    print(age_TP)

    print("True Negatives")
    print(races_TN)
    print(gender_TN)
    print(age_TN)

    print("False Positives")
    print(races_FP)
    print(gender_FP)
    print(age_FP)


    print("False Negatives")
    print(races_FN)
    print(gender_FN)
    print(age_FN)


    print("Count Positives")
    print(races_p)
    print(gender_p)
    print(age_p)

    print("Count Negatives")
    print(races_n)
    print(gender_n)
    print(age_n)


    for key in races_count.keys():
        races_TP[key] = (races_TP[key] + races_TN[key])/float(races_count[key])

    print(races_TP)

    for key in gender_count.keys():
        gender_TP[key] = (gender_TP[key]+ gender_TN[key])/ float(gender_count[key])

    print(gender_TP)

    for key in age_count.keys():
        age_TP[key] = (age_TP[key]+age_TN[key])/ float(age_count[key])

    print(age_TP)

    print("Test Accuracy: %.4f, Test Error: %.6f" % (acc_total/float(count_total), loss/float(count_total)))
    np.savetxt('scores_' + str(test_dataset) + '_' + swap_model + '_' + test_swap_model + '_' +
               args.full_test_model_path.split('/')[-1].split('.')[0] + '.csv',
               np.concatenate((np.expand_dims(y_test_total, 1), np.expand_dims(scores_total, 1)), axis=1), delimiter=',')


def get_XceptionNet(output_classes, input_img_size=224):
    # model = timm.create_model('xception', pretrained=True, num_classes=output_classes)

    import torch.nn as nn
    from cnn_finetune import make_model

    def make_classifier(in_features, num_classes):
        return nn.Sequential(nn.Linear(in_features, num_classes), nn.Sigmoid())

    disc_3 = make_model('xception', num_classes=output_classes, pretrained=True,
                        input_size=(input_img_size, input_img_size))  # ,
    # classifier_factory=make_classifier)
    return disc_3
    # return model


epoch = 1

if args.checkpoint_path is not None:
    model = torch.load(args.checkpoint_path, map_location=device).to(device)
    try:
        epoch = int(args.checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0])+1
    except:
        epoch = int(args.checkpoint_path.split('/')[-1].split('_')[0])+1

else:
    model = get_XceptionNet(num_classes, image_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if swap_model == 'sberswap' or test_swap_model == 'sberswap':
    model_sberswap, handler, netArc, G_sberswap, app_sberswap = sberswap_init(mode)
if swap_model == 'simswap' or test_swap_model == 'simswap':
    spNorm, model_simswap, app, net = simswap_init(mode)


if mode == 'bias_eval':

    for i in range(70000):
        print("here 1")
        images = generate_images_batch(network_pkl=network, seed=i, truncation_psi=random.uniform(0, 1),
                                       noise_mode='const',
                                       outdir='./',
                                       translate=parse_vec2('0,0'), rotate=0, BATCH_SIZE=2,
                                       class_idx=None)  # pylint: disable=no-value-for-parameter
        print("here 2")
        images = images[0]

        cv2.imwrite( 'bias_generated_imgs/' + str(i) + '.png', images)

        obj = DeepFace.analyze(images, actions=['age', 'gender', 'race'], enforce_detection=False, prog_bar=False)

        if obj['dominant_race'] in races_count.keys():
            races_count[obj['dominant_race']] += 1
        else:
            races_count[obj['dominant_race']] = 1

        if obj['gender'] in gender_count.keys():
            gender_count[obj['gender']] += 1
        else:
            gender_count[obj['gender']] = 1


        if obj['age'] < 10:
            age = '<10'
        elif 10 <= obj['age'] < 20:
            age = '10-20'
        elif 20 <= obj['age'] < 30:
            age = '20-30'
        elif 30 <= obj['age'] < 40:
            age = '30-40'
        elif 40 <= obj['age'] < 50:
            age = '40-50'
        elif 50 <= obj['age'] < 60:
            age = '50-60'
        elif 60 <= obj['age'] < 70:
            age = '60-70'
        elif 70 <= obj['age'] < 80:
            age = '70-80'
        elif 80 <= obj['age'] < 90:
            age = '80-90'
        elif obj['age'] >= 90:
            age = '>90'

        if age in age_count.keys():
            age_count[age] += 1
            # age_TP[age] += int(pred.numpy())
        else:
            age_count[age] = 1

    print(age_count)
    print(gender_count)
    print(races_count)




if mode == 'train' or mode == 'train_test' or mode == 'train_real_gpu':

    train_error = 1000.0
    train_acc_count = 0
    train_loss_count = 0

    count = 0
    batches = 1
    c = 0
    if mode == 'train_real_gpu':
        if finetune_dataset == 'ffhq':
            ffhq_path = '/scratch/aj3281/ffhq-dataset/train/'

            path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))

        if finetune_dataset == 'celeba-hq':
            path = '/scratch/aj3281/celebA-HQ-dataset-download/data1024x1024/train/'

            path_list = glob.glob(os.path.join(path, "*.jpg"))

        # import random
        # random.shuffle(path_list)

    while (train_error > 0.00001 and index < float('inf')) or epoch >= 500:

        if mode == 'train_real_gpu':

            images = []
            for j in range(batches*BATCH_SIZE//2, min((batches+1)*BATCH_SIZE//2, len(path_list))):
                images.append(cv2.imread(path_list[j]))
        else:
            # seeds = parse_range(str(index) + '-' + str(index + BATCH_SIZE-1)) #
            index = index + 1  # + BATCH_SIZE

            images = generate_images_batch(network_pkl=network, seed=index, truncation_psi=random.uniform(0, 1), noise_mode='const',
                                           outdir='./',
                                           translate=parse_vec2('0,0'), rotate=0, BATCH_SIZE=BATCH_SIZE,
                                           class_idx=None) # pylint: disable=no-value-for-parameter


            # c += 1
            # cv2.imwrite(str(batches) + '_' + str(epoch) + '_' + str(c) + '_real.png', images[0])

            # images = images[..., ::-1]


        swapped_images = []
        i = 0
        temp = 0
        while i < len(images) - 1:

            if swap_model == 'sberswap':
                swapped = sberswap(images[i], images[i + 1], model_sberswap, handler, netArc,
                                   G_sberswap, app_sberswap, mode)
            elif swap_model == 'simswap':
                # images = images[..., ::-1]
                swapped = simswap(images[i], images[i + 1], spNorm, model_simswap, app, net, mode)

            if swapped is not None:
                # swapped = swapped[..., ::-1]
                swapped_images.append(swapped)
                # plt.close()
                # plt.imshow(swapped)
                # plt.show()
                c += 1
                # print("got swapped")


                #
                # cv2.imwrite(str(batches) + '_' + str(epoch) + '_' + str(c) + '_' + swap_model + '_swapped.png', swapped)
                # cv2.imwrite(str(batches) + '_' + str(epoch) + '_' + str(c) + '_' + swap_model + '_real1.png', images[i])
                # cv2.imwrite(str(batches) + '_' + str(epoch) + '_' + str(c) + '_' + swap_model + '_real2.png', images[i+1])

                # plt.imshow('abc', swapped)
                # plt.show()
            else:
                temp+=1
                # print("No face was detected in the image thus skipping over this image. ")

            i += 1
        swapped_images = np.array(swapped_images)
        print("No of faces not detected: ", temp)

        if len(swapped_images) == 0:
            print("NO FACES AT ALL WERE DETECTED")
            print(len(images))
            print(epoch, batches)
            continue
        # print("hereeee")
        # exit(0)
        #
        # x = input()

        images = np.array(images)


        X = np.append(swapped_images, images, axis=0)
        y = np.append(np.zeros(len(swapped_images)), np.ones(len(images)))

        temp = list(zip(X, y))
        random.shuffle(temp)
        X, y = zip(*temp)
        # train_acc, train_error = train(model, X, y)
        X = torch.Tensor(np.array(X)).to(device).permute(0, 3, 1, 2)
        y = torch.Tensor(y).to(device)
        # print(epoch, batches, X.shape)

        model.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y.long())
        loss.backward()
        optimizer.step()
        # train_loss = loss.item()
        train_acc_temp, train_error_temp, count_temp, _, _, _ = test2(model, X, y)
        print(epoch, batches, train_acc_temp, train_error_temp)
        train_acc_count += train_acc_temp * count_temp
        train_loss_count += train_error_temp * count_temp
        count += count_temp
        ## TESTING ##

        if batches % args.n_steps == 0:
            seeds = parse_range('1-100')
            # seeds = parse_range(str(index) + '-' + str(index + BATCH_SIZE-1)) #
            # index = index + 1 + BATCH_SIZE
            import random
            images = generate_images(network_pkl=network, seeds=seeds, truncation_psi=random.uniform(0, 1), noise_mode='const',
                                     outdir='./',
                                     translate=parse_vec2('0,0'), rotate=0,
                                     class_idx=None)  # pylint: disable=no-value-for-parameter #0.7
            swapped_images = []
            i = 0
            while i < len(images) - 1:
                # for i in range(len(images), 2):
                #     swapped = simswap(images[i][..., ::-1], images[i + 1][..., ::-1], spNorm, model_simswap, app, net)
                # print("Swapped: ", swapped)
                if swap_model == 'sberswap':
                    swapped = sberswap(images[i], images[i + 1], model_sberswap, handler, netArc,
                                       G_sberswap, app_sberswap, mode) #[..., ::-1]
                elif swap_model == 'simswap':
                    swapped = simswap(images[i], images[i + 1], spNorm, model_simswap, app, net, mode)
                if swapped is not None:
                    swapped_images.append(swapped)
                else:
                    print("No face was detected in the image thus skipping over this image. ")

                i += 1
            swapped_images = np.array(swapped_images)
            images = np.array(images)
            X_test = np.append(swapped_images, images, axis=0)
            y_test = np.append(np.zeros(len(swapped_images)), np.ones(len(images)))

            X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
            y_test = torch.Tensor(y_test)

            test_acc, test_error, _, scores, _, _ = test2(model, X_test, y_test)

            train_error = train_loss_count / float(count)
            train_acc = train_acc_count / float(count)

            print("EPOCH: %d, Train Accuracy: %.4f, Train Error: %.6f, Test Accuracy: %.4f, Test Error: %.6f" % (epoch,
                                                                                                                 train_acc,
                                                                                                                 train_error,
                                                                                                                 test_acc,
                                                                                                                 test_error))
            torch.save(model, 'models_' + swap_model + '/' + str(epoch) + '_' + args.save_model_suffix + '.pth')
            del X_test, y_test

            count = 0
            train_loss_count = 0
            train_acc_count = 0
            epoch += 1
            batches = 0
            np.savetxt('scores_val_' + swap_model + '.csv', scores, delimiter=',')

        batches += 1

elif mode == 'finetune' or mode == 'train_real':

    if swap_model == 'simswap':
        epoch = 26
        filename = 'models_' + swap_model + '/' + str(epoch) + '.pth'
    else:
        epoch = 48
        filename = 'models_' + swap_model + '/' + str(epoch) + '.pth'

    if mode == 'finetune':
        model = torch.load(filename, map_location=device).to(device)

    def _worker_init_fn(id):
        # Worker process should inherit its affinity from parent
        affinity = os.sched_getaffinity(0)
        print(f"Process  Worker {id} set affinity to: {affinity}")

    transformed_dataset = DatasetFinetune(args.finetune_dataset, swap_model)
    dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)#,  num_workers=5, persistent_workers=True, worker_init_fn=_worker_init_fn, pin_memory=False) #, pin_memory=True,  prefetch_factor=3, , num_workers=2
    # print(len(transformed_dataset))
    epochs = 100
    # print("here")
    for epoch in range(epochs):

        count = 0
        train_acc_count = 0
        train_loss_count = 0

        for idx, (X_batch, y_batch) in enumerate(dataloader):
            # print('count', count)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            model.zero_grad()
            y_pred = model(X_batch)
            # print(type(y_pred), type(y_batch))
            loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss_count += loss.item()
            count += BATCH_SIZE
            # print(y_pred)
            # print(y_batch)
            # print(torch.argmax(y_pred,1) == y_batch)

            train_acc_count += int(np.count_nonzero((torch.argmax(y_pred, 1) == y_batch).detach().cpu().numpy()))
            print(train_acc_count / float(count), train_loss_count / float(count))

        print( train_acc_count / float(len(dataloader)), train_loss_count / float(len(dataloader)))

        # train_acc_temp, train_error_temp, count_temp, _ = test(model, dataloader)
        # print(train_acc_temp, train_error_temp)
        # train_acc_count += train_acc_temp * count_temp
        # train_loss_count += train_error_temp * count_temp

        # count += count_temp

        torch.save(model, 'models/' + mode + '_' + str(swap_model) + '_' + str(finetune_dataset) + '_' + str(
            epoch) + '_' + args.save_model_suffix + '.pth')

elif mode == 'test2' or mode == 'train_test2':

    if args.full_test_model_path is not None:
        filename = args.full_test_model_path
    elif test_swap_model == 'simswap':

        epoch = 26
        filename = 'models_' + test_swap_model + '/' + str(epoch) + '.pth'
    else:
        epoch = 48
        filename = 'models_' + test_swap_model + '/' + str(epoch) + '.pth'

    model = torch.load(filename, map_location=device).to(device)
    criterion = nn.CrossEntropyLoss()
    test_dataset = args.test_dataset  # 'celeba-hq'
    if test_dataset == 'ffhq':

        ffhq_path = '/scratch/aj3281/ffhq-dataset/test/'
        images = []
        for file in glob.glob(os.path.join(ffhq_path + "*/*.png")):
            # print(file)
            I = cv2.imread(file)
            images.append(I)
    elif test_dataset == 'lfw':
        from sklearn.datasets import fetch_lfw_pairs
        from sklearn.datasets import fetch_lfw_people
        import sklearn
        # sklearn.datasets.get_data_home('/scratch/aj3281/')

        lfw_people = fetch_lfw_people(data_home='/scratch/aj3281/', color=True)
        print(lfw_people.images.shape)
        lfw_shape = lfw_people.images.shape
        # resize_factor = min(1024/float(lfw_shape[0]), 1024/float(lfw_shape[1]))
        # lfw_people
        images = []
        for img in lfw_people.images:
            I = cv2.resize(img, (1024, 1024)) * 255
            I = I.astype(np.uint8)
            # print(np.max(I))
            images.append(I)

    elif test_dataset == 'celeba-hq':

        # Images (9999, 1024, 1024, 3)
        # Swapped (9289, 1024, 1024, 3)
        # X_test (19288, 1024, 1024, 3)

        path = '/scratch/aj3281/celebA-HQ-dataset-download/data1024x1024/test/'
        images = []
        for file in glob.glob(os.path.join(path, "20*.jpg")):
            # print(I)
            I = cv2.imread(file)
            images.append(I)

    elif test_dataset == 'adfes':  # Shape: (576, 720, 3)

        adfes_path1 = '/scratch/aj3281/Still_pictures*'
        # adfes_path2 = '/scratch/aj3281/'
        images = []
        for file in glob.glob(
                os.path.join(adfes_path1 + "/*/*.jpg")):  # + glob.glob(os.path.join(adfes_path2 + "*/*.png")):
            # print(file)
            I = cv2.imread(file)
            # print(I.shape)

            I = cv2.resize(I, (1024, 1024))
            images.append(I)

        for file in glob.glob(os.path.join(adfes_path1 + "/*/*/*.jpg")):
            # print(file)
            I = cv2.imread(file)
            I = cv2.resize(I, (1024, 1024))
            images.append(I)

    elif test_dataset == 'IJCB2017':  # (2000, 3008, 3)
        path = '/vast/aj3281/FOCS/*'
        images = []
        for file in glob.glob(os.path.join(path + "/*.jpg")):  # + glob.glob(os.path.join(adfes_path2 + "*/*.png")):

            I = cv2.imread(file)
            print(I.shape)
            I = cv2.resize(I, (1024, 1024))
            images.append(I)
    elif test_dataset == 'pasc':
        print("here")
        path = '/vast/aj3281/PaSC/'
        images = []
        for file in glob.glob(os.path.join(path + "/*.jpg")):  # + glob.glob(os.path.join(adfes_path2 + "*/*.png")):

            I = cv2.imread(file)
            print(I.shape)
            I = cv2.resize(I, (1024, 1024))
            images.append(I)

    swapped_images = []

    i = 0
    while i < len(images) - 1:
        if swap_model == 'sberswap':
            swapped = sberswap(images[i], images[i + 1], model_sberswap, handler, netArc, G_sberswap, app_sberswap, mode)
        elif swap_model == 'simswap':
            swapped = simswap(images[i], images[i + 1], spNorm, model_simswap, app, net, mode)

        # print("Swapped: ", swapped)
        if swapped is not None:
            swapped_images.append(swapped)
        else:
            print("No face was detected - Test set")
        i += 1

    swapped_images = np.array(swapped_images)
    images = np.array(images)
    print("Images", images.shape)
    print("Swapped", swapped_images.shape)
    X_test = np.append(swapped_images, images, axis=0)

    y_test = np.append(np.zeros(len(swapped_images)), np.ones(len(images)))
    del swapped_images, images

    print("X_test", X_test.shape)

    X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
    y_test = torch.Tensor(y_test)
    # print("X_test", X_test.shape)
    # print(y_test)

    test_acc, test_error, _, scores, _, _ = test2(model, X_test, y_test)
    print("Test Accuracy: %.4f, Test Error: %.6f" % (test_acc, test_error))

    np.savetxt('scores_' + str(test_dataset) + '_' + swap_model + '_' + test_swap_model + '.csv',
               np.concatenate((np.expand_dims(y_test.cpu().numpy(),1), scores[:,1]), axis=1), delimiter=',')





elif mode == 'test' or mode == 'train_test':

    if args.full_test_model_path is not None:
        filename = args.full_test_model_path
    elif test_swap_model == 'simswap':

        epoch = 26
        filename = 'models_' + test_swap_model + '/' + str(epoch) + '.pth'
    else:
        epoch = 48
        filename = 'models_' + test_swap_model + '/' + str(epoch) + '.pth'

    model = torch.load(filename, map_location=device).to(device)
    criterion = nn.CrossEntropyLoss()
    test_dataset = args.test_dataset  # 'celeba-hq'

    if test_dataset == 'ffhq':

        ffhq_path = '/scratch/aj3281/ffhq-dataset/test/'
        path_list = glob.glob(os.path.join(ffhq_path + "*/*.png"))

    elif test_dataset == 'lfw':
        from sklearn.datasets import fetch_lfw_pairs
        from sklearn.datasets import fetch_lfw_people
        import sklearn

        # sklearn.datasets.get_data_home('/scratch/aj3281/')

        lfw_people = fetch_lfw_people(data_home='/scratch/aj3281/', color=True)
        print(lfw_people.images.shape)
        lfw_shape = lfw_people.images.shape
        # resize_factor = min(1024/float(lfw_shape[0]), 1024/float(lfw_shape[1]))
        # lfw_people
        images = []
        for img in lfw_people.images:
            I = cv2.resize(img, (1024, 1024)) * 255
            I = I.astype(np.uint8)
            # print(np.max(I))
            images.append(I)

    elif test_dataset == 'celeba-hq':

        # Images (9999, 1024, 1024, 3)
        # Swapped (9289, 1024, 1024, 3)
        # X_test (19288, 1024, 1024, 3)

        path = '/scratch/aj3281/celebA-HQ-dataset-download/data1024x1024/test/'
        path_list = glob.glob(os.path.join(path, "*.jpg"))


    elif test_dataset == 'adfes':  # Shape: (576, 720, 3)

        adfes_path1 = '/scratch/aj3281/Still_pictures*'
        path_list = glob.glob(os.path.join(adfes_path1 + "/*/*.jpg")) + glob.glob(os.path.join(adfes_path1 + "/*/*/*.jpg"))

    test(path_list, model)



    # elif test_dataset == 'IJCB2017':  # (2000, 3008, 3)
    #     path = '/vast/aj3281/FOCS/*'
    #     images = []
    #     for file in glob.glob(os.path.join(path + "/*.jpg")):  # + glob.glob(os.path.join(adfes_path2 + "*/*.png")):
    #
    #         I = cv2.imread(file)
    #         print(I.shape)
    #         I = cv2.resize(I, (1024, 1024))
    #         images.append(I)
    #
    # elif test_dataset == 'pasc':
    #     print("here")
    #     path = '/vast/aj3281/PaSC/'
    #     images = []
    #     for file in glob.glob(os.path.join(path + "/*.jpg")):  # + glob.glob(os.path.join(adfes_path2 + "*/*.png")):
    #
    #         I = cv2.imread(file)
    #         print(I.shape)
    #         I = cv2.resize(I, (1024, 1024))
    #         images.append(I)


    # print(count)
    # print(TP)
    # loss = loss / float(count)
    # accuracy = TP / float(count)
    #
    # scores = np.squeeze(scores, axis=1)
    # # test_acc, test_error, _, scores = test(model, X_test, y_test)
    # print("Test Accuracy: %.4f, Test Error: %.6f" % (accuracy, loss))
    #
    #
    # np.savetxt('scores_' + str(test_dataset) + '_' + swap_model + '_' + test_swap_model + '_' + args.full_test_model_path.split('/')[-1].split('.')[0]  +'.csv',
    #            np.concatenate((np.expand_dims(y_test, 1), scores[:,1]), axis=1), delimiter=',')
    #
    #




    # images = []
    # finetune_dataset = args.finetune_dataset #'celeba-hq'
    # if finetune_dataset == 'ffhq':
    #
    #     ffhq_path = '/scratch/aj3281/ffhq-dataset/'
    #
    #     for file in glob.glob(os.path.join(ffhq_path + "1*/*.png")):
    #         print(file)
    #         I = cv2.imread(file)
    #         images.append(I)
    # if finetune_dataset == 'celeba-hq':
    #
    #     path = '/scratch/aj3281/celebA-HQ-dataset-download/data1024x1024/'
    #
    #     for file in glob.glob(os.path.join(path, "1*.jpg")):
    #         I = cv2.imread(file)
    #         images.append(I)
    #
    # images = images[:len(images)//2]
    #
    # swapped_images = []
    #
    # i = 0
    # while i < len(images) - 1:
    #     # for i in range(len(images), 2):
    #     # try:
    #     if swap_model == 'sberswap':
    #         swapped = sberswap(images[i][..., ::-1], images[i + 1][..., ::-1], model_sberswap, handler, netArc, G_sberswap, app_sberswap)
    #     elif swap_model == 'simswap':
    #         swapped = simswap(images[i][..., ::-1], images[i + 1][..., ::-1], spNorm, model_simswap, app, net)
    #
    #     # swapped = simswap(images[i], images[i + 1], spNorm, model_simswap, app, net)
    #
    #     # print("Swapped: ", swapped)
    #     if swapped is not None:
    #
    #         swapped_images.append(swapped[..., ::-1])
    #     else:
    #         print("No face was detected - Test set")
    #
    #     i += 1
    # swapped_images = np.array(swapped_images)
    # images = np.array(images)
    # print("Images", images.shape)
    # print("Swapped", swapped_images.shape)
    # X = np.append(swapped_images, images, axis=0)
    #
    # y = np.append(np.zeros(len(swapped_images)), np.ones(len(images)))
    # del swapped_images, images
    #
    # temp = list(zip(X, y))
    # random.shuffle(temp)
    # X, y = zip(*temp)
    # # train_acc, train_error = train(model, X, y)
    # X = torch.Tensor(np.array(X)).permute(0, 3, 1, 2)#.to(device)
    # y = torch.Tensor(y)#.to(device)
    # print(X.shape)
    #
    # dataset = TensorDataset(X, y)  # create your datset
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
