import numpy as np

imagefolder = '/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images'
kpt_path = '/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/landmarks'
dict = {}
kpt = np.load(kpt_path)[0]

import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from decalib.utils.config import cfg
from decalib.utils import lossfunc


class GetKptLoss:
    def __init__(self, K, image_size, scale, image_path, kpt_path, trans_scale=0, isTemporal=False, isEval=False,
                 isSingle=False, ):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.imagepath = image_path
        self.kptpath = kpt_path

        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def getKpt(self, idx):
        images_list = [];
        kpt_list = [];
        image_path = self.imagepath
        kpt_path = self.kptpath

        image = imread(image_path) / 255.
        kpt_pc = np.load(kpt_path)
        kpt = np.load(kpt_path)[0]

        ### crop information
        tform = self.crop(image, kpt)
        ## crop
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_kpt = np.dot(tform.params,
                             np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

        images_list.append(cropped_image.transpose(2, 0, 1))
        kpt_list.append(cropped_kpt)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3

        return kpt_array

    def crop(self, image, kpt):
        # 围绕landmark中心裁剪图像
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform


get_org_kpt = GetKptLoss(image_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images',
                        kpt_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images')
org_kpt = get_org_kpt.getKpt()

get_render1_kpt = GetKptLoss(image_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images',
                        kpt_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images')
render1_kpt = get_render1_kpt.getKpt()

get_render2_kpt = GetKptLoss(image_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images',
                        kpt_path='/mnt/f/Data_Set/VGGFaceDataset/vgg_face_dataset/images')
render2_kpt = get_render2_kpt.getKpt()

render1_loss = lossfunc.weighted_landmark_loss(render1_kpt, org_kpt) * cfg.loss.lmk
render2_loss = org_loss = lossfunc.weighted_landmark_loss(render2_kpt, org_kpt) * cfg.loss.lmk

print('render1_loss:' , render1_loss)
print('render2_loss:' , render2_loss)