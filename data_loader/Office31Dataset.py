import torch.utils.data
import pandas as pd
import time
import numpy as np
import sys
import torch
import numbers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


sys.path.append('..')
import conf

opt = conf.Office31_Opt
WIN_LEN = opt['seq_len']

################################################ Image Transform Methods #####################################################

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h_off = random.randint(0, img.shape[1] - self.size)
        w_off = random.randint(0, img.shape[2] - self.size)
        img = img[:, h_off:h_off + self.size, w_off:w_off + self.size]
        return img


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    def __init__(self, mean=None, meanfile=None):
        if mean:
            self.mean = mean
        else:
            arr = np.load(meanfile)
            self.mean = torch.from_numpy(arr.astype('float32') / 255.0)[[2, 1, 0], :, :]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.sub_(m)
        return tensor


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = (img.shape[1], img.shape[2])
        th, tw = self.size
        w_off = int((w - tw) / 2.)
        h_off = int((h - th) / 2.)
        img = img[:, h_off:h_off + th, w_off:w_off + tw]
        return img


def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1

    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])


def image_test_10crop(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = [
        transforms.Compose([
            ResizeImage(resize_size), ForceFlip(),
            PlaceCrop(crop_size, start_first, start_first),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size), ForceFlip(),
            PlaceCrop(crop_size, start_last, start_last),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size), ForceFlip(),
            PlaceCrop(crop_size, start_last, start_first),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size), ForceFlip(),
            PlaceCrop(crop_size, start_first, start_last),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size), ForceFlip(),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_first, start_first),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_last, start_last),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_last, start_first),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_first, start_last),
            transforms.ToTensor(),
            normalize
        ]),
        transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize
        ])
    ]
    return data_transforms


################################################ Image Loading #####################################################

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, dataset, transform_tag="train", target_transform=None, mode='RGB'):

        self.imgs = dataset
        if transform_tag == "train":
            self.transform = image_train()
        elif transform_tag == "test":
            if conf.args.test_10crop:
                self.transform = image_test_10crop()
            else:
                self.transform = image_test()
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def get_num_domains(self):
        return 1

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Office31Dataset(Dataset):
    # load static files

    def __init__(self, file='./image_dataset/data/office/amazon_list.txt',
                 labels=None, domain=None, activity=None):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            domain: condition on user-phone combination
            activity: condition on action
            complementary: is it complementary dataset for given conditions? (used for "multi" case)

        """

        st = time.time()
        self.domain = domain
        self.activity = activity

        self.dataset = make_dataset(open(file).readlines(), labels)

        self.num_domains = 1

        ppt = time.time()
        #
        # self.preprocessing()
        print('Loading data done.\tPreprocessing:{:f}\tTotal Time:{:f}'.format(time.time() - ppt,
                                                                        time.time() - st))

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return self.num_domains

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        return self.dataset[idx]


if __name__ == '__main__':
    pass