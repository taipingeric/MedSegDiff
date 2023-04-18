import torch
from torchvision import transforms as T
import imgaug
import pandas as pd
import cv2
import os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
# from segmentation_models_pytorch.encoders import get_preprocessing_fn

class UALDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, config, aug=False):
        data_dir = config['data_dir']
        self.img_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.df = pd.read_csv(df_path)
        self.config = config
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.aug = aug
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=(-15, 15),
                shear=(-10, 10),
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            ),
        ])
        self._init_img_preprocess_fn(config)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['img']))[:, :, ::-1]
        mask = cv2.imread(os.path.join(self.mask_dir, row['mask']))[:, :, 0]
        
        img = self.preprocess(img)
        mask = self.preprocess_mask(mask)

        if self.aug:
            img, mask = self.seq(image=img, segmentation_maps=mask)

        return self._to_torch_tensor(img, mask)
    
    def preprocess(self, img):
        img = cv2.resize(img, (self.img_w, self.img_h))
        return img
    
    def preprocess_mask(self, mask):
        mask = imgaug.augmentables.segmaps.SegmentationMapsOnImage(mask.astype(np.int8), 
                                                                   shape=mask.shape)
        mask = mask.resize((self.img_h, self.img_w))
        return mask

    def _init_img_preprocess_fn(self, config):
        model_type = config['model']['name']
        transform = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # if model_type == 'smp':
        #     raise AssertionError('Not implemented model type preprocess fn')
        #     # encoder_name = config['model'][model_type]['encoder_name']
        #     # transform = get_preprocessing_fn(encoder_name, pretrained='imagenet')
        # else:
        #     transform = T.Compose([
        #         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ])
        self.transform = transform

    def _to_torch_tensor(self, img, mask):
        model_type = self.config['model']['name']

        if model_type == 'smp':
            img = img / 255.
            img = torch.tensor(img, dtype=torch.float)
            img = self.transform(img).permute(2, 0, 1)
        else:
            raise ValueError('Not implemented model type preprocess fn')

        if mask:
            mask = mask.get_arr()  # to np
            mask = torch.tensor(mask, dtype=torch.long)
            return img, mask
        else:
            return img