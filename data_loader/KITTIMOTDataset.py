import os
import warnings
import torch.utils.data
import pandas as pd
import time
import numpy as np
import sys
import conf
from PIL import Image
# from skimage.transform import resize
from utils.augmentations import AUGMENTATION_TRANSFORMS
from utils.transforms import DEFAULT_TRANSFORMS
import torch.nn.functional as F
import copy
opt = conf.KITTI_MOT_Opt


class KITTIMOTDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform=None):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source
        self.img_shape = opt['img_size']
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        self.file_path = opt['file_path']
        self.max_objects = 50
        self.class_index = [opt['classes'].index(i) for i in opt['sub_classes']]

        self.tgt_or_val = None

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.sub_path1 = 'origin'
            self.sub_path2 = 'train'

        elif domains[0] == '2d_detection':

            self.sub_path1 = '2d_detection'
            self.sub_path2 = ''
        elif domains[0].startswith('rain'):
            self.sub_path1=domains[0].split('-')[0]
            self.sub_path2=domains[0].split('-')[1]+'mm'
            # self.sub_path1 = 'rain'
            # self.sub_path2 = domains[0].lstrip('rain-')
        elif domains[0] == 'half1':
            self.sub_path1 = 'origin'
            self.sub_path2 = 'train'
        elif domains[0] == 'half2':
            self.sub_path1 = 'origin'
            self.sub_path2 = 'train'

        # self.min_size = self.img_size - 3 * 32
        # self.max_size = self.img_size + 3 * 32


        if transform == 'def':
            self.transform = DEFAULT_TRANSFORMS
        elif transform == 'aug':
            self.transform = AUGMENTATION_TRANSFORMS
        else:
            raise NotImplementedError



        self.preprocessing()

    def preprocessing(self):

        path = f'{self.file_path}/{self.sub_path1}/{self.sub_path2}/'
        # ['0000', '0001']

        for sequence_name in sorted(os.listdir(path)):
            if conf.args.dataset == 'kitti_mot_test':
                if sequence_name not in ['0000', '0001']:
                    continue
            if self.domains[0] == 'half1':
                if sequence_name not in ['0000', '0001']:
                # if sequence_name not in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']:
                    continue
            elif self.domains[0] == 'half2':
                # if sequence_name not in ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']:
                if sequence_name not in ['0011', '0012']:
                    continue
            elif self.domains[0].endswith('tgt'):
                if sequence_name not in ['0010','0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']:
                    continue
            elif self.domains[0].endswith('val'):
                if sequence_name not in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009']:
                    continue

            txtnpng = sorted(os.listdir(os.path.join(path, sequence_name)))
            print(f'total_len is : {len(txtnpng) // 2}')
            for i in range(len(txtnpng) // 2):
                img_path = os.path.join(path, sequence_name, str(i).zfill(6) + ".png")
                img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)




                # ---------
                #  Image
                # ---------
                '''
                h, w, _ = img.shape
                dim_diff = np.abs(h - w)
                # Upper (left) and lower (right) padding
                pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                # Determine padding
                pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
                # Add padding
                input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
                padded_h, padded_w, _ = input_img.shape
                # Resize and normalize
                input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect')
                # Channels-first
                input_img = np.transpose(input_img, (2, 0, 1))
                # As pytorch tensor
                # input_img = torch.from_numpy(input_img).float()
                if (input_img.shape != (3, 416, 416)):
                    print(f"shape is : {input_img.shape}sth wrong....")
                '''

                # ---------
                #  Label
                # ---------

                label_path = os.path.join(path, sequence_name, str(i).zfill(6) + ".txt")
                labels = None
                if os.path.exists(label_path):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels = np.loadtxt(label_path).reshape(-1, 5)

                    '''
                    tmp_filtered_labels = np.array([i for i in labels if i[0] in self.class_index])
                    # print(labels.shape)
                    if labels.shape == (0,5) or len(tmp_filtered_labels) == 0:
                        labels = None
                    else:
                        # print("normal")
                        labels = []
                        for elem in tmp_filtered_labels: #Label number from zero: ex) 3, 5, 6... -> 0, 1, 2...
                            elem_tmp_filtered_labels = elem
                            elem_tmp_filtered_labels[0] = self.class_index.index(elem_tmp_filtered_labels[0])
                            labels.append(elem_tmp_filtered_labels)
                        # print(labels)
                        labels = np.array(labels)

                        # Extract coordinates for unpadded + unscaled image
                        x1 = w * (labels[:, 1] - labels[:, 3] / 2)  # []
                        y1 = h * (labels[:, 2] - labels[:, 4] / 2)  # []
                        x2 = w * (labels[:, 1] + labels[:, 3] / 2)  # []
                        y2 = h * (labels[:, 2] + labels[:, 4] / 2)  # []

                        # Adjust for added padding
                        x1 += pad[1][0]
                        y1 += pad[0][0]
                        x2 += pad[1][0]
                        y2 += pad[0][0]
                        
                        # Calculate ratios from coordinates
                        labels[:, 1] = ((x1 + x2) / 2) / padded_w
                        labels[:, 2] = ((y1 + y2) / 2) / padded_h
                        labels[:, 3] *= w / padded_w
                        labels[:, 4] *= h / padded_h
                
                # Fill matrix
                filled_labels = np.zeros((self.max_objects, 5))
                if labels is not None:
                    filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
                '''





                # filled_labels = torch.from_numpy(filled_labels)
                self.features.append(img)
                self.class_labels.append(labels)

                # assuming that single domain is passed as list
                self.domain_labels.append(0) #TODO: change this if domain label is required

        # self.features = np.array(self.features, dtype=np.float)
        # self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        self.domain_labels = torch.utils.data.TensorDataset(torch.from_numpy(self.domain_labels))

        # self.datasets.append(torch.utils.data.TensorDataset(
        #     torch.from_numpy(self.features).float(),
        #     torch.from_numpy(self.class_labels).float(),
        #     torch.from_numpy(self.domain_labels).float(),
        # ))
        # print(f'count_num is : {self.count_num}')
        # self.dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.features).float(),
        #                                               torch.from_numpy(self.class_labels),
        #                                               torch.from_numpy(self.domain_labels))
        # self.dataset = [torch.from_numpy(self.features).float(),
        #                                               torch.from_numpy(self.class_labels),
        #                                               torch.from_numpy(self.domain_labels)
        # self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features).float(),
        #                                                     torch.from_numpy(self.class_labels),
        #                                                     torch.from_numpy(self.domain_labels)))
        #
        # self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.features)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.get_datasets


    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, boxes, dl= self.features[idx], self.class_labels[idx], self.domain_labels[idx]
        img, boxes = self.transform((img, copy.deepcopy(boxes)))#deepcopy is required to modify list

        img = F.interpolate(img.unsqueeze(0), size=self.img_shape, mode="nearest").squeeze(0) #resize
        return img, boxes, dl[0]# dl is tuple of size 1


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def collate_fn(batch):
    # Drop invalid images
    batch = [data for data in batch if data is not None]

    imgs, bb_targets, dls = list(zip(*batch))

    # Selects new image size every tenth batch
    # if self.multiscale and self.batch_count % 10 == 0:
    #     self.img_size = random.choice(
    #         range(self.min_size, self.max_size + 1, 32))

    # Resize images to input shape
    # imgs = torch.stack([resize(img, 416) for img in imgs])

    # Add sample index to targets
    for i, boxes in enumerate(bb_targets):
        boxes[:, 0] = i
    bb_targets = torch.cat(bb_targets, 0)

    return torch.stack(imgs), bb_targets, dls

if __name__ == '__main__':
    pass
