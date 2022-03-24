import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import numpy as np
from numpy import ndarray
from skimage import measure
import cv2
from DDT.ImageSet import ImageSet
import os.path as osp
import os

from PIL import Image

import pathlib

import densecrf


class DDT(object):
    def __init__(self, use_cuda=False):
        if not torch.cuda.is_available():
            self.use_cuda=False
        else:
            self.use_cuda=use_cuda

        if self.use_cuda:
            self.pretrained_feature_model = (models.vgg19(pretrained=True).features).cuda()
        else:
            self.pretrained_feature_model = models.vgg19(pretrained=True).features

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.totensor = transforms.ToTensor()

    def fit(self, traindir):
        train_dataset = ImageSet(traindir, resize=1000)

        descriptors = np.zeros((1, 512))

        for index in range(len(train_dataset)):
            image = train_dataset[index]
            h, w = image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, h, w)
            if self.use_cuda:
                image=image.cuda()

            output = self.pretrained_feature_model(image)[0, :]
            output = output.view(512, output.shape[1] * output.shape[2])
            output = output.transpose(0, 1)
            descriptors = np.vstack((descriptors, output.detach().cpu().numpy().copy()))
            del output

        descriptors = descriptors[1:]

        descriptors_mean = sum(descriptors)/len(descriptors)
        descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)
        pca = PCA(n_components=1)
        pca.fit(descriptors)
        trans_vec = pca.components_[0]
        return trans_vec, descriptors_mean_tensor

    def co_locate(self, testdir, savedir, sal_dir, trans_vector, descriptor_mean_tensor):
        test_dataset = ImageSet(testdir, resize=1000)
        if self.use_cuda:
            descriptor_mean_tensor = descriptor_mean_tensor.cuda()
        for index in range(len(test_dataset)):
            image = test_dataset[index]
            origin_image = image.copy()
            origin_height, origin_width = origin_image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, origin_height, origin_width)
            if self.use_cuda:
                image = image.cuda()
            featmap = self.pretrained_feature_model(image)[0, :]
            h, w = featmap.shape[1], featmap.shape[2]
            featmap = featmap.view(512, -1).transpose(0, 1)
            featmap -= descriptor_mean_tensor.repeat(featmap.shape[0], 1)
            features = featmap.detach().cpu().numpy()
            del featmap

            P = np.dot(trans_vector, features.transpose()).reshape(h, w)

            mask = self.max_conn_mask(P, origin_height, origin_width)

            mask_3 = np.concatenate(
                (np.zeros((2, origin_height, origin_width), dtype=np.uint16), mask * 255), axis=0)
            mask_3 = np.transpose(mask_3, (1, 2, 0))
            mask_3[mask_3[:, :, 2] > 254, 0] = 255
            mask_3[mask_3[:, :, 2] > 254, 1] = 255
            mask_3[mask_3[:, :, 2] > 254, 2] = 255
            mask_3 = np.array(mask_3, dtype=np.float32)

            file = pathlib.Path(testdir / test_dataset.get_file_name(index))
            cosal_map = densecrf.apply_crf(file, mask_3, savedir)
            
            sal_map = pathlib.Path(sal_dir / test_dataset.get_file_name(index)).with_suffix(".png")
            
            I  = Image.open(sal_map)
            Iq = np.asarray(I)

            Iq = cv2.resize(Iq, (cosal_map.shape[1], cosal_map.shape[0]), interpolation = cv2.INTER_AREA)

            res = np.multiply(Iq, cosal_map)
            res = Image.fromarray(res)
            res = res.convert('RGB')
            
            res.save(pathlib.Path(savedir / test_dataset.get_file_name(index)).with_suffix(".png"), 'PNG')            
            

    def max_conn_mask(self, P, origin_height, origin_width):
        h, w = P.shape[0], P.shape[1]
        highlight = np.zeros(P.shape)
        for i in range(h):
            for j in range(w):
                if P[i][j] > 0:
                    highlight[i][j] = 1

        labels = measure.label(highlight, connectivity=1, background=0)
        props = measure.regionprops(labels)
        max_index = 0
        for i in range(len(props)):
            if props[i].area > props[max_index].area:
                max_index = i
        max_prop = props[max_index]
        highlights_conn = np.zeros(highlight.shape)
        for each in max_prop.coords:
            highlights_conn[each[0]][each[1]] = 1

        highlight_big = cv2.resize(
            highlights_conn,
            (origin_width, origin_height),
            interpolation=cv2.INTER_NEAREST
        )

        highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)
        return highlight_big



