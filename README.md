# You don't need no dataset: Leveraging synthetic datasets for training deep classifiers. 



This package contains modules to generate fair datasets from a biased StyleGAN2 model and use these for training classifiers. As an example, this package supports face-swap detection. 


# Zero-Shot Racially Balanced Dataset Generation Using StyleGAN2

[Paper (arXiv)](https://arxiv.org/pdf/2305.07710)

## Setup 

Our work heavily depends on the StyleGAN2 github repository. It can be cloned and extracted as follows. 

```
$ git clone https://github.com/NVlabs/stylegan3 && mv styleGAN3/* ./
```


## Fair Dataset Generation 



To generate data via the proposed latent space exploration approach you will have to run the script independently for each racial groups you are interested in. The codebase currently supoorts 6 racial groups - Blacks, Indians, Asians, Hispanic, White, Middle Eastern - based on the Deepface ethnicity classifier. 

The mode parameter can be generate - for generating sampling in the W space or generate_z for generating samples in the Z space. The start and end iter help in controlling parallel runs of the script. The refer to the seed values. Like mentioned earlier the script supports the following racial groups - ```["asian", "white", "middle_eastern", 'black', 'latino', 'indian']```

```
$ python3 main_latent_exploration.py --target_race <racial group> --mode <generate_z, generate> --start_iter 0 --end_iter 1000 
```


You can also generate samples using random rejection sampling. 

```
$ python3 main_latent_exploration.py --target_race <racial group> --mode rejection_sampling
```

### Upcoming 
Support for generating combinations of demographics. 

# A Dataless FaceSwap Detection Approach Using Synthetic Images

[Paper](https://ieeexplore.ieee.org/iel7/10007927/10007928/10007967.pdf)


This package contains the codebase for the paper titled "A Dataless FaceSwap Detection Approach Using Synthetic Images" that was accepted at IJCB 2022. The paper proposes a privacy preserving approach to detect faceswaps using of faces that don't exist. We make use of synthetic images generated using StyleGAN3 and show that it has various other benefits such as reduction in bias, learning more generalizable features and also generalizability to unseen faceswap models and datasets. 


## Setup 

We currently support two models to generate faceswaps - SimSwap and Sberswap. These need to cloned and extracted in this repository for it to work. You can do so using the commands below. However, we write our own face cropper and thus the insightace_func needs to be removed from the directory. 

```
$ git clone https://github.com/neuralchen/SimSwap
$ rm SimSwap/insightface_func && mv SimSwap/* ./
$ git clone https://github.com/ai-forever/sber-swap
$ rm sber-swap/insightface_func && mv sber-swap/* ./
```

We also use the StyleGAN3 github repository for generating synthetic images. It can similarly be cloned and extracted as follows: 

```
$ git clone https://github.com/NVlabs/stylegan3 && mv styleGAN3/* ./
```


## Training using Syntheic Images 

To train the faceswap detection model - XceptionNet - using synthetic images, run the following command on your terminal.

```
$ python3 main.py --mode train --swap_model {simswap or sberswap} --batch_size 12 --n_steps 2000 --save_model_suffix synthetic
```


## Training using Real data 

```
$ python3 main.py --mode train_real_gpu --swap_model {simswap or sberswap} --batch_size 12 --n_steps 2000 --save_model_suffix real
```


## Testing

You can test the models using your own trained models or use the pretrained models provided by us. For either of them you test the models on the FFHQ dataset, Celeba-HQ, ADFES or any of the subsets of the FaceForensics++ dataset. You need to download and extract the respective dataset in this working repository. 

```
$ python3 main.py --mode test --test_dataset {ffhq, celeba-hq, adfes, ff-neuraltextures, ff-face2face, ff-faceswap, ff-faceshifter or ff-google} --full_test_model_path /path/to/the/test/model
```



If you use this package please consider citiing the corresponding paper. 

```
   @INPROCEEDINGS{Jain_IJCB_2022,
            author = {Jain, Anubhav and Memon, Nasir and Togelius, Julian},
            title = {A Dataless FaceSwap Detection Approach Using Synthetic Images},
         booktitle = {International Joint Conference on Biometrics (IJCB 2022)},
            year = {2022},
            note = {Accepted for Publication in IJCB2022},
   }
```

