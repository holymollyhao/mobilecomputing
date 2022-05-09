import argparse
import os
import sys
import re

from main import get_path as get_path
import numpy as np
from sklearn.preprocessing import minmax_scale
# import torch.utils.tensorboard as tf
import torch.utils.tensorboard as tf  # use pytorch tensorboard; conda install -c conda-forge tensorboard
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import statistics
from conf import *
args = None


def load_epoch(file_path):
    found = False
    for file in os.listdir(file_path):
        if 'events' in file:
            file_path = file_path + '/' + file
            found = True
            break
    if found:
        for e in tf.compat.v1.train.summary_iterator(file_path):
            for v in e.summary.value:
                if v.tag == 'args/epoch':
                    return str(v.tensor.string_val[0].decode('utf-8'))

    return str(0)


def load_result_file(file_path):
    result = ''

    f = open(file_path)
    lines = f.readlines()

    for line in lines:
        matchObj = re.match('.+\s([\d\.]+)', line)
        result += matchObj.group(1) + '\t'
    result += '\n'

    f.close()

    return result


is_plot = False

def validate_file(filename):
    result = None
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        # print(line)
        matchObj = re.match('Result from\s([\de\-\.]+)\spaths', line)
        if matchObj:
            num_files = int(matchObj.group(1))
            if 'harth' in filename:
                num_domains = len(HARTHOpt['tgt_domains'])
            elif 'extrasensory' in filename:
                num_domains = len(ExtraSensoryOpt['tgt_domains'])
            elif 'reallifehar' in filename:
                num_domains = len(RealLifeHAROpt['tgt_domains'])
            elif 'cifar100' in filename:
                num_domains = len(CIFAR100Opt['tgt_domains'])
            elif 'cifar10' in filename:
                num_domains = len(CIFAR10Opt['tgt_domains'])
            elif 'kitti_sot' in filename:
                num_domains = len(KITTI_SOT_Opt['tgt_domains'])
            elif 'vlcs' in filename:
                num_domains = len(VLCSOpt['tgt_domains'])
            elif 'pacs' in filename:
                num_domains = len(PACSOpt['tgt_domains'])
            else:
                raise NotImplementedError
            assert num_files == num_domains, print(f'Failed to fetch data from {num_domains} files. {num_files} files exist.')
            break

    f.close()
    return result

def read_file(filename):
    result = None
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        # print(line)
        matchObj = re.match('avg_acc:\s([\de\-\.]+)', line)
        if matchObj:
            result = matchObj.group(1)
            break

    f.close()
    return result


def load_result_file(file_path):
    result = ''

    f = open(file_path)
    lines = f.readlines()

    for line in lines:
        matchObj = re.match('.+\s([\d\.]+)', line)
        result += matchObj.group(1) + '\t'
    result += '\n'

    f.close()

    return result


def get_avg_acc(file_path):
    result = 0

    f = open(file_path)
    lines = f.readlines()

    for line in lines:
        matchObj = re.match('.+\s([\d\.]+)', line)
        result += float(matchObj.group(1))

    f.close()

    return result / len(lines)


def mp_work(path):
    # print(f'Current file:\t{path}')

    if args.eval_type == 'avg_acc':
        tmp_dict = {}
        tmp_dict[path] = get_avg_acc(path + '/accuracy.txt')
        return tmp_dict


# log_icsr_FT_all_dist3_ep1_uex50_mem500_mtCBRS.txt

# .*${dataset}.*scaled/${method}/.*${date}.*dist${dist}_.*ep${ep}_.*uex${uex}_.*mem${mem}_.*mt${mt}_s0.*

def main(args):
    log_path = './eval_logs/'

    is_online = True
    datasets = [
        # 'cifar10',
        # 'cifar100',
        'vlcs',
        'pacs',
        # 'kitti_sot',
        # 'harth',
        # 'extrasensory',
        # 'reallifehar',

        # 'hhar',
        # 'wesad',
        # 'ichar',
        # 'icsr'
    ]
    methods = [

        # 'Src',
        # 'TT_BATCH_STATS',
        # 'PseudoLabel',
        # 'TENT',
        # 'T3A',
        # 'COTTA',
        'Ours',

    ]

    dists = [
        '0',
        '1',
    ]
    cases = [
        # 'case1',
        'case2',
        'case3',
        'case4',
        'case5',
        'case6',
        # 'case7',
        # 'case8',

        'case9',
        'case10',
        'case11',
    ]

    lrs = [
        '0.01',
        '0.001',
        '0.0001',
    ]

    mts = [
        # '0.1',
        # '0.05',
        '0.01',
        # '0.001',
    ]
    ks = [
        # '3',
        '4',
        # '5',
    ]
    ablation=True
    if not ablation:
        result_dic = {}

        for dataset in datasets:
            result_dic[dataset] = {}
            for method in methods:
                result_dic[dataset][method] = {}
                for dist in dists:
                    # result_dic[dataset][method][dist] = {}
                    # for lr in lrs:
                    #     result_dic[dataset][method][dist][lr] = {}
                    #     for bn in bns:
                    #         result_dic[dataset][method][dist][lr][bn] = {}
                    # for case in cases:
                    if method == 'Src':
                        result_dic[dataset][method] = {}
                        target = f'log_{dataset}_{method}.txt'
                    else:
                        target = f'log_{dataset}_{method}_{dist}.txt'
                        # target = f'log_{dataset}_{method}_{dist}_{case}.txt'
                        # target = f'log_{dataset}_{method}_{dist}_{case}_{lr}_{bn}.txt'

                    pattern = re.compile(target)

                    found = 0
                    print(f'Pattern:\t{pattern}', end='\t')
                    for file in os.listdir(log_path):
                        if pattern.match(file):
                            file = log_path + file
                            print(f'Matched:\t{file}')
                            validate_file(file)
                            value = read_file(file)
                            if method == 'Src':
                                result_dic[dataset][method] = float(value)
                            else:
                                result_dic[dataset][method][dist] = float(value)
                                # result_dic[dataset][method][dist][case] = float(value)
                                # result_dic[dataset][method][dist][lr][bn][case] = float(value)
                            found += 1

                            if found == 0:
                                print(f'Cannot find a matched file for:\ttarget')
                            elif found > 1:
                                print(f'Multiple files found for:\ttarget')

        print('\n\n\n\n')
        print(datasets)
        # for lr in lrs:
        #     for bn in bns:
        #         print(f'LR:{lr} BN:{bn}')
        # for case in cases:
        #     print(case, end='\t')
        for method in methods:
            print(f'{method}', end='\t')

            for dist in dists:
                for dataset in datasets:
                # if dist == '0':
                #     print(f'---------Real distribution--------')
                # elif dist == '1':
                #
                #     print(f'---------Random distribution---------')

                    if method == 'Src':
                        print(result_dic[dataset][method], end='\t')
                    else:
                        print(result_dic[dataset][method][dist], end='\t')
                        # print(result_dic[dataset][method][dist][case], end='\t')
                        # print(result_dic[dataset][method][dist][lr][bn][case], end='\t')
                print('',end='\t')
                #         print()
            print()
        print('\n\n\n')
    else:

        result_dic = {}
        for dataset in datasets:
            result_dic[dataset] = {}
            for method in methods:
                result_dic[dataset][method] = {}
                for dist in dists:
                    result_dic[dataset][method][dist] = {}
                    for mt in mts:
                        result_dic[dataset][method][dist][mt] = {}
                        for k in ks:
                            result_dic[dataset][method][dist][mt][k] = {}
                            for case in cases:
                                target = f'log_{dataset}_{method}_{dist}_{mt}_{k}_{case}.txt'
                                pattern = re.compile(target)

                                found = 0
                                print(f'Pattern:\t{pattern}', end='\t')
                                for file in os.listdir(log_path):
                                    if pattern.match(file):
                                        file = log_path + file
                                        print(f'Matched:\t{file}')
                                        validate_file(file)
                                        value = read_file(file)

                                        result_dic[dataset][method][dist][mt][k][case] = float(value)
                                        found += 1

                                        if found == 0:
                                            print(f'Cannot find a matched file for:\ttarget')
                                        elif found > 1:
                                            print(f'Multiple files found for:\ttarget')

        print('\n\n\n\n')
        print(datasets)

        for mt in mts:
            print(f'MT:{mt}')
            for k in ks:
                print(f'k:{k}')
                for case in cases:
                    print(case, end='\t')
                    for method in methods:
                        print(f'{method}', end='\t')

                        for dist in dists:
                            for dataset in datasets:
                                # if dist == '0':
                                #     print(f'---------Real distribution--------')
                                # elif dist == '1':
                                #
                                #     print(f'---------Random distribution---------')

                                print(result_dic[dataset][method][dist][mt][k][case], end='\t')
                            print('', end='\t')
                            #         print()
                    print()
        print('\n\n\n')


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--regex', type=str, default='', help='train condition regex')
    parser.add_argument('--get_last_epoch', action='store_true', help='Get last epoch accuracy?')
    parser.add_argument('--directory', type=str, default='',
                        help='which directory to search through? ex: ichar/FT_FC')
    parser.add_argument('--target', type=str, default='accuracy',
                        help='what is the target objective? [accuracy, loss, f1, auroc] ')
    parser.add_argument('--eval_type', type=str, default='avg_acc',
                        help='what type of evaluation? in [result, log, estimation, dtw, avg_acc]')
    parser.add_argument('--early_stop', action='store_true', help='Early stop or best acc?')
    parser.add_argument('--tolerance', type=int, default=0, help='Early stop tolerance')

    parser.add_argument('--scaling', action='store_true', help='minmax scale [0,1]')

    ### Methods ###

    parser.add_argument('--method', default=[], nargs='*',
                        help='method to evaluate, e.g., dev, iwcv, entropy, etc.')

    return parser.parse_args()


if __name__ == '__main__':
    import time

    st = time.time()
    # args = parse_arguments(sys.argv[1:])
    main(args)
    # print(f'time:{time.time() - st}')
