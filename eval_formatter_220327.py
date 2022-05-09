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
        matchObj = re.match('Result from:\s([\de\-\.]+)\spaths', line)
        if matchObj:
            num_files = matchObj.group(1)
            assert(num_files == 8), print(f'Failed to fetch data from 8 files. Only {num_files} files exist.')
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
        'harth',
        'extrasensory',
        'reallifehar',
        # 'hhar',
        # 'wesad',
        # 'ichar',
        # 'icsr'
    ]
    methods = [
        # 'FT_all',
        'Src',
        # 'SHOT',
        # 'CDAN',
        # 'FeatMatch',
        # 'TT_SINGLE_STATS',
        # 'TT_BATCH_STATS',
        # 'TT_BATCH_PARAMS',
        # 'TT_WHOLE',
        # 'TENT',
        # 'TENT_STATS',
        # 'Ours',
        # 'VOTE',
    ]
    method_config = {
        'Src': {
            'uex': [
                None
            ],
            'eps':
                [
                    None
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },
        'FT_all': {
            'uex': [
                '50',
                '100',
                '99999'
            ],
            'eps':
                [
                    '1', '5'
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },

        'SHOT': {
            'uex': [
                # '50',
                # '100',
                '200',
                # '99999'
            ],
            'eps':
                [
                    '1',
                    # '5'
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },

        'CDAN': {
            'uex': [
                # '50',
                # '100',
                '200',
                # '99999'
            ],
            'eps':
                [
                    '1',
                    # '5'
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },
        'FeatMatch': {
            'uex': [
                # '50',
                # '100',
                '200',
                # '99999'
            ],
            'eps':
                [
                    '1',
                    # '5'
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },
        'TT_SINGLE_STATS': {

            'uex': [
                '1',
            ],
            'eps':
                [
                    '1',
                ],
            'momentums': [
                "0.1",
                "0.01",
                "0.001",
                "0.0001",
                "0.00001",
                "0.000001"
            ],
            'mts': [
                None
            ]
        },
        'TT_BATCH_STATS': {

            'uex': [
                '16',
                '32',
                '64',
            ],
            'eps':
                [
                    '1',
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },
        'TENT': {

            'uex': [
                '16',
                '32',
                '64'
            ],
            'eps':
                [
                    '1',
                    '5',
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },
        'TENT_STATS': {

            'uex': [
                '16',
                '32',
                '64'
            ],
            'eps':
                [
                    '1',
                ],
            'momentums': [
                0.01,
                0.0001
            ],
            'mts': [
                None
            ]
        },
        'Ours': {

            'uex': [
                '16',
                '32',
                '64',
                # '128',
                # '256',
                # '512'
            ],
            'eps':
                [
                    '1',
                ],
            'momentums': [
                0.1,
                0.01,
                # 0.0001
            ],
            'mts': [
                # 'CBRS',
                'CBFIFO',
                # 'Diversity',
                # 'Reservoir',
            ]
        },
        'VOTE': {

            'uex': [
                '16',
                '32',
                '64',
            ],
            'eps':
                [
                    '1',
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },

        'Default': {
            'uex': [
                '16',
                '32',
                '64'
            ],
            'eps':
                [
                    '1',
                    '5'
                ],
            'momentums': [
                None
            ],
            'mts': [
                None
            ]
        },

    }

    mems = ['200', '500']

    mts = ['FIFO', 'CBRS']
    dists = [
        '0',
        '1',
    ]

    result_dic = {}
    if is_online:  ################################################################################## Online

        for dataset in datasets:
            result_dic[dataset] = {}
            for method in methods:
                if method in method_config:
                    eps = method_config[method]['eps']
                    uex_range = method_config[method]['uex']
                    momentums =  method_config[method]['momentums']
                    mts = method_config[method]['mts']
                else:
                    eps = method_config['Default']['eps']
                    uex_range = method_config['Default']['uex']
                    momentums =  method_config['Default']['momentums']
                    mts =  method_config['Default']['mts']
                result_dic[dataset][method] = {}
                for dist in dists:
                    if method != 'Src':
                        result_dic[dataset][method][dist] = {}
                    for ep in eps:
                        if method != 'Src':
                            result_dic[dataset][method][dist][ep] = {}
                        for uex in uex_range:
                            if method != 'Src':
                                result_dic[dataset][method][dist][ep][uex] = {}
                            for mt in mts:

                                if method != 'Src':
                                    result_dic[dataset][method][dist][ep][uex][mt] = {}
                                for momentum in momentums:
                                    if method != 'Src':
                                        result_dic[dataset][method][dist][ep][uex][mt][momentum] = {}

                                    if method == 'Src':
                                        result_dic[dataset][method] = {}
                                        target = f'log_{dataset}_{method}.txt'
                                    elif method == 'TT_SINGLE_STATS':
                                        result_dic[dataset][method][dist][momentum] = {}
                                        target = f'log_{dataset}_{method}_{dist}_{momentum}.txt'
                                    elif method == 'TENT_STATS':
                                        target = f'log_{dataset}_{method}_{dist}_{ep}_{uex}_{momentum}.txt'
                                    elif method == 'Ours':
                                        target = f'log_{dataset}_{method}_{dist}_{ep}_{uex}_{mt}_{momentum}.txt'
                                    else:
                                        target = f'log_{dataset}_{method}_{dist}_{ep}_{uex}.txt'

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
                                            elif method == 'TT_SINGLE_STATS':
                                                result_dic[dataset][method][dist][momentum] = float(value)
                                            elif method in ['TENT_STATS']:
                                                result_dic[dataset][method][dist][ep][uex][momentum] = float(value)
                                            elif method in ['Ours']:
                                                result_dic[dataset][method][dist][ep][uex][mt][momentum] = float(value)
                                            else:
                                                result_dic[dataset][method][dist][ep][uex] = float(value)
                                            found += 1

                                    if found == 0:
                                        print(f'Cannot find a matched file for:\ttarget')
                                    elif found > 1:
                                        print(f'Multiple files found for:\ttarget')

        print('\n\n\n\n')
        for dist in dists:
            if dist == '0':
                print(f'---------Real distribution--------')
            elif dist == '1':

                print(f'---------Random distribution---------')
            for dataset in datasets:
                print(f'############{dataset}############')
                for method in methods:
                    print(f'{method}')

                    if method in method_config:
                        eps = method_config[method]['eps']
                        uex_range = method_config[method]['uex']
                        momentums = method_config[method]['momentums']
                    else:
                        eps = method_config['Default']['eps']
                        uex_range = method_config['Default']['uex']
                        momentums = method_config['Default']['momentums']

                    if method == 'Src':
                        print(result_dic[dataset][method], end='\t')
                        print()
                    elif method == 'TT_SINGLE_STATS':
                        for momentum in momentums:
                            print(f'{momentum}', end='\t')
                            print(result_dic[dataset][method][dist][momentum], end='\t')
                            print()
                    elif method == 'TENT_STATS':
                        for momentum in momentums:
                            for ep in eps:
                                # print(f'Epoch {ep}', end='\t')
                                for uex in uex_range:
                                    print(f'Mom{momentum} Epoch {ep} Every {uex}', end='\t')
                                    print(result_dic[dataset][method][dist][ep][uex][momentum], end='\t')
                                    print()
                    elif method == 'Ours':
                        for mt in mts:
                            for momentum in momentums:
                                for ep in eps:
                                    # print(f'Epoch {ep}', end='\t')
                                    for uex in uex_range:
                                        print(f'Epoch {ep} Memory{mt} Mom{momentum} Every {uex}', end='\t')
                                        print(result_dic[dataset][method][dist][ep][uex][mt][momentum], end='\t')
                                        print()
                    else:
                        for ep in eps:
                            # print(f'Epoch {ep}', end='\t')
                            for uex in uex_range:
                                print(f'Epoch {ep} Every {uex}', end='\t')
                                print(result_dic[dataset][method][dist][ep][uex], end='\t')
                                print()
        print()
        print('\n\n\n')

    else:  ################################################################################## Offline

        for dataset in datasets:
            result_dic[dataset] = {}
            for method in methods:
                result_dic[dataset][method] = {}
                for dist in dists:
                    result_dic[dataset][method][dist] = {}
                    for ep in ['50']:
                        target = f'log_{dataset}_{method}_dist{dist}_ep{ep}.txt'
                        pattern = re.compile(target)

                        found = 0
                        print(f'Pattern:\t{pattern}', end='\t')
                        for file in os.listdir(log_path):
                            if pattern.match(file):
                                file = log_path + file
                                print(f'Matched:\t{file}')
                                value = read_file(file)
                                result_dic[dataset][method][dist][ep] = float(value)
                                found += 1

                        if found == 0:
                            print(f'Cannot find a matched file for:\ttarget')
                        elif found > 1:
                            print(f'Multiple files found for:\ttarget')

        print('\n\n\n\n')

        for dataset in datasets:
            print(f'###################################{dataset}###################################')
            for method in methods:
                print(f'***********************************{method}*************************************')
                for dist in dists:
                    if dist == '0':
                        print(f'---------------------------Real distribution---------------------------')
                    elif dist == '1':
                        print(f'---------------------------Random distribution---------------------------')
                    elif dist == '2':
                        print(f'---------------------------Sorted distribution---------------------------')
                    print(result_dic[dataset][method][dist][ep], end='\t')
                    print()


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
