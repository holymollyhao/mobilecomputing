from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import math
from .HHARDataset import HHARDataset
from .DSADataset import DSADataset
from .ICHARDataset import ICHARDataset
from .ICSRDataset import ICSRDataset
from .Office31Dataset import Office31Dataset
from .Office31Dataset import ImageList
from .WESADDataset import WESADDataset
from .HARTHDataset import HARTHDataset
from .RealLifeHARDataset import RealLifeHARDataset
from .ExtraSensoryDataset import ExtraSensoryDataset
from .KITTIMOTDataset import KITTIMOTDataset
from .KITTIMOTDataset import collate_fn
from .KITTISOTDataset import KITTISOTDataset
from .OpportunityDataset import OpportunityDataset
from .GAITDataset import GaitDataset
from .PAMAP2Dataset import PAMAP2Dataset
from .CIFAR10Dataset import CIFAR10Dataset
from .CIFAR100Dataset import CIFAR100Dataset
from .ImageNetDataset import ImageNetDataset
from .VLCSDataset import VLCSDataset
from .OfficeHomeDataset import OfficeHomeDataset
from .PACSDataset import PACSDataset
from .DogwalkDataset import DogwalkDataset

import os
import pickle
import re
import random
import copy

import conf


def keep_order_split(entire_data, train_size, valid_size, test_size):
    all_indices = [i for i in range(len(entire_data))]

    valid_indices = random.sample(all_indices, valid_size)

    for i in sorted(valid_indices, reverse=True):  # reverse is required
        all_indices.pop(i)

    test_indices = random.sample(all_indices, test_size)

    for i in sorted(test_indices, reverse=True):
        all_indices.pop(i)

    valid_data = torch.utils.data.Subset(entire_data, valid_indices)
    test_data = torch.utils.data.Subset(entire_data, test_indices)
    train_data = torch.utils.data.Subset(entire_data, all_indices)

    return train_data, valid_data, test_data


def split_data(entire_data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(entire_data) * valid_split)
    test_size = math.floor(len(entire_data) * test_split)

    train_size = len(entire_data) - valid_size - test_size
    # if valid_size > 0 and test_size > 0 : # this meets when it's mostly target condition
    #     if train_size < 20: # for 'src = all' && meta learning cases, source should have at least 5 supports and queries, while target has at least 10 shots for evaluation.
    #         gap = 20 - train_size
    #         train_size = 20
    #         valid_size -= gap # decrease valid size
    assert (train_size >= 0 and valid_size >= 0 and test_size >= 0)

    # train_data, valid_data, test_data = torch.utils.data.random_split(entire_data,
    #                                                                   [train_size, valid_size, test_size])

    train_data, valid_data, test_data = keep_order_split(entire_data, train_size, valid_size, test_size)

    if len(entire_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data


def datasets_to_dataloader(datasets, batch_size, concat=True, shuffle=True, drop_last=False):
    if concat:
        data_loader = None
        if len(datasets):
            if type(datasets) is torch.utils.data.dataset.Subset:
                datasets = [datasets]
            if sum([len(dataset) for dataset in datasets]) > 0:  # at least one dataset has data
                data_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last, pin_memory=False, collate_fn=collate_fn if conf.args.dataset in ['kitti_mot','kitti_mot_test'] else None)
        return data_loader
    else:
        data_loaders = []
        for dataset in datasets:
            if len(dataset) == 0:
                continue
            else:
                data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                drop_last=drop_last, pin_memory=False, collate_fn=collate_fn if conf.args.dataset in ['kitti_mot','kitti_mot_test'] else None))
        if dataset in ["office31"]:
            if len(data_loaders) > 0:
                return data_loaders[0]
            else:
                return None
        return data_loaders


def load_cache(dataset, cond, data_file_path, transform=None):
    root = '/mnt/sting/twkim/WWW/cached_data/'
    dir = root + str(dataset) + '/'
    filename = re.sub("[^a-zA-Z0-9 \n]", '_', str(cond) + '_' + str(data_file_path))
    if transform:
        filename += '_' + transform + '.pkl'
    else:
        filename += '.pkl'
    cache_path = dir + filename

    if os.path.isfile(cache_path):
        print(f'Cache hit:{cache_path}')
        return torch.load(cache_path)
    else:
        print(f'Cache miss:{cache_path}')
        return None


def save_cache(loaded_data, dataset, cond, data_file_path, transform=None):
    root = '/mnt/sting/twkim/WWW/cached_data/'
    dir = root + str(dataset) + '/'
    filename = re.sub("[^a-zA-Z0-9 \n]", '_', str(cond) + '_' + str(data_file_path))
    if transform:
        filename += '_' + transform + '.pkl'
    else:
        filename += '.pkl'
    cache_path = dir + filename
    if not os.path.exists(dir):
        os.makedirs(dir)
    return torch.save(loaded_data, cache_path, pickle_protocol=4)


def domain_data_loader(dataset, domains, file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf,
                       test_max_rows=np.inf, valid_split=0, test_split=0, separate_domains=False, is_src=True,
                       num_source=9999):
    entire_datasets = []
    train_datasets = []

    valid_datasets = []
    test_datasets = []
    st = time.time()

    if domains is not None:
        if domains == 'src':
            processed_domains = conf.args.opt['src_domains']
        elif isinstance(domains, (list,)):
            processed_domains = domains
        else:
            processed_domains = [domains]
    elif is_src:

        if conf.args.validation:
            processed_domains = sorted(list(set(conf.args.opt['src_domains']) - set([conf.args.tgt]))) # Leave-one-user-out
        else:
            processed_domains = conf.args.opt['src_domains']
    else:#not called?
        if conf.args.validation:
            processed_domains = conf.args.opt['src_domains']
        else:
            processed_domains = conf.args.opt['tgt_domains']

    ##-- load dataset per each domain
    print('Domains:{}'.format(processed_domains))


    loaded_data = None

    if dataset in ['hhar']:
        # hhar_model_device_user ex) hhar.gear.gear1.a, hhar.nexus..c

        cond1 = sorted(list(set([i.split('.')[0] for i in processed_domains])))
        cond2 = sorted(list(set([i.split('.')[1] for i in processed_domains])))

        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = HHARDataset(file=file_path, users=cond1, models=cond2, devices=None)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['ichar']:
        # ichar_domain ex) metasense_acitivity.PH0007-jskim

        cond = processed_domains
        is_ecdf = True if 'ecdf' in conf.args.model else False

        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = ICHARDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['icsr']:
        # icsr_domain ex) icsr.PH0007-jskim

        cond = processed_domains

        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = ICSRDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['wesad']:

        cond = processed_domains

        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = WESADDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['harth']:


        # if 'split' in file_path:
        #     if is_src:
        #         processed_domains = [d+'_back' for d in processed_domains] # back is source
        #         # processed_domains = [d+'_thigh' for d in processed_domains] # thigh is source
        #     else:
        #         processed_domains = [d+'_thigh' for d in processed_domains] # thigh is target
        #         # processed_domains = [d+'_back' for d in processed_domains] # back is target
        cond = processed_domains
        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = HARTHDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['reallifehar']:

        cond = processed_domains

        loaded_data = load_cache(dataset, processed_domains, file_path)
        if not loaded_data:
            loaded_data = RealLifeHARDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, processed_domains, file_path)

    elif dataset in ['extrasensory']:

        cond = processed_domains

        reduced_name = [domain[:7] for domain in processed_domains]

        loaded_data = load_cache(dataset, reduced_name, file_path)
        if not loaded_data:
            loaded_data = ExtraSensoryDataset(file=file_path, domains=cond, max_source=num_source)
            save_cache(loaded_data, dataset, reduced_name, file_path)

    elif dataset in ['kitti_mot', 'kitti_mot_test']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        transform = 'def' if is_src else 'def'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        if not loaded_data:
            loaded_data = KITTIMOTDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['kitti_sot', 'kitti_sot_test']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        if not loaded_data:
            loaded_data = KITTISOTDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['cifar10']:

        cond = processed_domains

        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)

        if not loaded_data:
            loaded_data = CIFAR10Dataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['cifar100']:

        cond = processed_domains

        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)

        if not loaded_data:
            loaded_data = CIFAR100Dataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['imagenet']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        # transform = 'src' if is_src else 'val'
        transform = 'none'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        #
        if not loaded_data:
            loaded_data = ImageNetDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['vlcs']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        # transform = 'src' if is_src else 'val'
        transform = None

        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        if not loaded_data:
            loaded_data = VLCSDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path)
    elif dataset in ['officehome']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        # transform = 'src' if is_src else 'val'
        transform = None
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        if not loaded_data:
            loaded_data = OfficeHomeDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path)
    elif dataset in ['pacs']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        # transform = 'src' if is_src else 'val'
        transform = None
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        if not loaded_data:
            loaded_data = PACSDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path)
    elif dataset in ['dogwalk']:

        cond = processed_domains

        # transform = 'aug' if is_src else 'def'
        # transform = 'src' if is_src else 'val'
        transform = None
        # loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        # if not loaded_data:
        loaded_data = DogwalkDataset(file=file_path, domains=cond, max_source=num_source)
            # save_cache(loaded_data, dataset, processed_domains, file_path)

    if separate_domains:
        for train_data in loaded_data.get_datasets_per_domain():
            entire_datasets.append(train_data)
    else:
        train_data = loaded_data
        entire_datasets.append(train_data)

    ##-- split each dataset into train, valid, and test
    for train_data in entire_datasets:
        total_len = len(train_data)
        train_data, valid_data, test_data = split_data(train_data, valid_split, test_split, train_max_rows,
                                                       valid_max_rows, test_max_rows)
        if dataset in ['office31']:
            # apply transforms to images
            if len(train_data) > 0:
                train_data = ImageList(train_data, transform_tag='train')
            if len(valid_data) > 0:
                valid_data = ImageList(valid_data, transform_tag='test')
            if len(test_data) > 0:
                test_data = ImageList(test_data, transform_tag='test')

        train_datasets.append(train_data)
        valid_datasets.append(valid_data)
        test_datasets.append(test_data)

        print('#Multi?:{:d} data_loader len:{:d} Train: {:d}\t# Valid: {:d}\t# Test: {:d}'.format(
            True if domains == ['rest'] else False, total_len, len(train_data), len(valid_data),
            len(test_data)))

    train_datasets = train_datasets[:num_source]
    valid_datasets = valid_datasets[:num_source]
    test_datasets = test_datasets[:num_source]

    print('# Time: {:f} secs'.format(time.time() - st))

    if separate_domains:
        # actual batch size is multiplied by num_class
        train_data_loaders = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=False, drop_last=True, shuffle=False if conf.args.src_sep_noshuffle else True)
        # if 'PN' in conf.args.method:
        #     test_batch_size = 5
        # else:
        valid_data_loaders = datasets_to_dataloader(valid_datasets, batch_size=32,
                                                    concat=False)  # set validation batch_size = 32 to boost validation speed
        if is_src:
            test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False,
                                                       drop_last=True)
        else:
            test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False,
                                                       shuffle=False)
        # assert (len(train_data_loaders) == len(valid_data_loaders) == len(test_data_loaders))
        data_loaders = []
        for i in range(len(train_data_loaders)):
            data_loader = {
                'train': train_data_loaders[i],
                'valid': valid_data_loaders[i] if len(valid_data_loaders) == len(train_data_loaders) else None,
                'test': test_data_loaders[i] if len(valid_data_loaders) == len(train_data_loaders) else None,
                'num_domains': len(train_data_loaders)
            }
            data_loaders.append(data_loader)
        print('num_domains:' + str(len(train_data_loaders)))

        return data_loaders
    else:
        if is_src:
            train_data_loader = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=True,
                                                       drop_last=True,
                                                       shuffle=True)  # Drop_last for avoding one-sized minibatches for batchnorm layers
        else:
            train_data_loader = datasets_to_dataloader(train_datasets, batch_size=1, concat=True,
                                                       drop_last=False,
                                                       shuffle=False)
        valid_data_loader = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=True,
                                                   shuffle=False)
        test_data_loader = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=True, shuffle=False)

        data_loader = {
            'train': train_data_loader,
            'valid': valid_data_loader,
            'test': test_data_loader,
            'num_domains': sum([dataset.dataset.get_num_domains() for dataset in train_datasets]),
        }
        print('num_domains:' + str(data_loader['num_domains']))
        return data_loader


if __name__ == '__main__':
    pass
