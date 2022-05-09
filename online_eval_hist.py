import argparse
import os
import pickle
import sys
import re

from main import get_path as get_path
import numpy as np
from sklearn.preprocessing import minmax_scale
# import torch.utils.tensorboard as tf
import torch.utils.tensorboard as tf  # use pytorch tensorboard; conda install -c conda-forge tensorboard
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from tqdm import tqdm
import statistics

args = None

def process_path(path):
    strip_path = path[1:].split('/')[1:]
    return strip_path

def check_num_inconsistency(arr1, arr2):
    assert len(arr1) == len(arr2), "two arrays have different length!"
    cnt = 0
    for i in range(len(arr1)):
        if(arr1[i] != arr2[i]):
            cnt += 1
    return cnt

def show_plt(path):
    split_path_list = process_path(path)
    json_file = open(os.path.join(path, 'online_eval.json'), 'r')
    json_data = json.load(json_file)

    try :
        print(os.path.join(path, 'logbnstats.pickle'))
        pickle_file = open(os.path.join(path, 'logbnstats.pickle'), 'rb')
        pickle_data = pickle.load(pickle_file)
    except :
        raise NotImplementedError("logbnstats not in specified path")

    dataset_dict = {
        'harth': {
            'num_class': 12,
            'src_path': "220317_minmax_scaling_all_split_win50_ep50_s0",
        },
        'extrasensory': {
            'num_class': 5,
            'src_path': "220317_selectedfeat_woutloc_std_scaling_all_win5_ep50_s0",
        },
        'reallifehar': {
            'num_class': 4,
            'src_path': "220317_reallifehar_acc_minmax_scaling_all_win400_overlap_ep50_s0",
        }
    }

    src_path = os.path.join(split_path_list[0], split_path_list[1], "Src", split_path_list[3], dataset_dict[split_path_list[1]]['src_path'])
    json_src_file = open(os.path.join(src_path, 'online_eval.json'), 'r')
    json_src_data = json.load(json_src_file)

    src_accuracy = round(json_src_data['accuracy'][-1], 2)
    tgt_accuracy = round(json_data['accuracy'][-1], 2)

    # last_epoch, last_item = sorted(json_data.items(), reverse=True, key= lambda x: int(x[0]))[0]
    last_epoch = len(json_data['gt'])
    last_item = json_data
    # print(sorted(json_data.items(), reverse=True))
    (x_list, y_list1, y_list2) = ([i for i in range(1, int(last_epoch) + 1)], last_item['pred'], last_item['gt'])


    # get list of ys
    num_of_bns = 0
    if(split_path_list[2] != 'Src'):
        num_of_bns = len(pickle_data['mean'][0])
        list_of_mean_input_list = [[] for i in range(num_of_bns)]
        list_of_mean_output_list = [[] for i in range(num_of_bns)]
        list_of_running_mean = [[list_bn[i] for list_bn in pickle_data['running_mean']] for i in range(num_of_bns)]
        list_of_running_var = [[list_bn[i] for list_bn in pickle_data['running_var']] for i in range(num_of_bns)]

        for elem in pickle_data['mean']:
            for i in range(len(elem)):
                list_of_mean_input_list[i].append(elem[i][0])
                list_of_mean_output_list[i].append(elem[i][1])
    additional_figs = num_of_bns * 4

    title = split_path_list[2] + '_' + split_path_list[3]

    print("last epoch is : " + str(last_epoch))

    plt.figure(figsize=(50, 50))

    plt.suptitle(f'Accuracy difference is : {round(float(tgt_accuracy-src_accuracy),2)}', fontsize=25)

    plt.subplot(4 + additional_figs, 1, 1)
    # plt.figure(figsize=(30, 6))
    plt.step(x_list, y_list1)
    plt.step(x_list, y_list2, '-.', color='orange')
    # label each graph
    plt.legend(['Prediction', 'Ground Truth'])
    plt.yticks([i for i in range(dataset_dict[split_path_list[1]]['num_class'])])
    plt.title('Both')

    plt.subplot(4 + additional_figs, 1, 2)
    # plt.figure(figsize=(30, 6))
    plt.step(x_list, y_list1)
    plt.yticks([i for i in range(dataset_dict[split_path_list[1]]['num_class'])])
    plt.title('Prediction')

    plt.subplot(4 + additional_figs, 1, 3)
    # plt.figure(figsize=(30, 6))
    plt.step(x_list, y_list2, color='orange')
    plt.yticks([i for i in range(dataset_dict[split_path_list[1]]['num_class'])])
    plt.title('Ground truth')

    if(split_path_list[2] != 'Src'):
        for i in range(num_of_bns):
            print(len(x_list))
            print(len(list_of_mean_input_list[i]))
            plt.subplot(4 + additional_figs, 1, 4 + i * 4)
            plt.plot([j for j in range(1, len(list_of_mean_input_list[i]) + 1)], list_of_mean_input_list[i], color='gray')
            plt.title(f'{i}th batchnorm Mean values - input')

            plt.subplot(4 + additional_figs, 1, 4 + i * 4 + 1)
            plt.plot([j for j in range(1, len(list_of_mean_output_list[i]) + 1)], list_of_mean_output_list[i], color='gray')
            plt.title(f'{i}th batchnorm Mean values - output')

            plt.subplot(4 + additional_figs, 1, 4 + i * 4 + 2)
            plt.plot([j for j in range(1, len(list_of_running_mean[i]) + 1)], list_of_running_mean[i], color='gray')
            plt.title(f'{i}th batchnorm RUNNING MEAN values')

            plt.subplot(4 + additional_figs, 1, 4 + i * 4 + 3)
            plt.plot([j for j in range(1, len(list_of_running_var[i]) + 1)], list_of_running_var[i], color='gray')
            plt.title(f'{i}th batchnorm RUNNING VAR values')

    # plt.subplot(4, 1, 4)
    # plt.figure(figsize=(30, 6))
    # plt.step(x_list, y_list3, color='green')
    # plt.title('Ground truth')

    #add title for plot
    print(check_num_inconsistency(json_data['pred'], json_data['gt']))
    print(check_num_inconsistency(json_src_data['pred'], json_src_data['gt']))

    save_path = f'/home/twkim/git/WWW/hist_log/{split_path_list[1]}/{split_path_list[2]}/{split_path_list[4]}/'
    print(f'save path is  : {save_path}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print( title + '_' + str(round(float(tgt_accuracy-src_accuracy), 2)))

    plt.savefig(save_path + title + '_' + str(round(float(tgt_accuracy-src_accuracy), 2)) + '.png')
    plt.show()


def main(args):
    # 220220_online_cdan_99999sample_dist0_beta0.3_thr0_ep1_uex100_mem200_mtCBRS_s0
    # .*${dataset}.*/${method}/.*${date}.*dist${dist}_.*ep${ep}_.*uex${uex}_.*mem${mem}_.*mt${mt}_s0.* 2>&1

    # for CDAN and other algorithms
    # .*${dataset}.*/${method}/.*${date}.*dist${dist}_.*ep${ep}_.*uex${uex}_.*mem${mem}_.*mt${mt}_s0.* 2>&1
    # pattern_of_path = '.*reallifehar.*/VOTE/.*220313-5.*dist0_.*uex32.*'

    # best-working in harth
    pattern_of_path = '.*harth.*/TENT_UNIFORM_MEMORY/.*/.*220323_99999sample_.*dist0.*'
    # pattern_of_path = '.*harth.*/Src/.*/220317_minmax_scaling_all_split_win50_ep50_s0.*'

    # best-working in extrasensory
    # pattern_of_path = '.*extrasensory.*/TT_SINGLE_STATS/.*/220317_99999sample_dist0_ep1_uex1_mem1_bnm0.001_s0.*'
    # pattern_of_path = '.*extrasensory.*/Src/.*/220317_selectedfeat_woutloc_std_scaling_all_win5_ep50_s0.*'
    # pattern_of_path = '.*extrasensory.*/Src/.*/tgt_A5A30F76-581E-4757-97A2-957553A2C6AA.*/220317_selectedfeat_woutloc_std_scaling_all_win5_ep50_s0.*'
    # pattern_of_path = '.*extrasensory.*/TT_SINGLE_STATS.*/tgt_A5A30F76-581E-4757-97A2-957553A2C6AA.*/220317_99999sample_dist0_ep1_uex1_mem1_bnm0.001_s0.*'



    # best-working in reallifehar
    # pattern_of_path = '.*reallifehar.*/TT_SINGLE_STATS/.*/220317_99999sample_dist0_ep1_uex1_mem1_bnm0.001_s0.*'
    # pattern_of_path = '.*reallifehar.*/Src/.*/220317_reallifehar_acc_minmax_scaling_all_win400_overlap_ep50_s0.*'

    # for Src Data
    # pattern_of_path = '.*harth.*/Src/.*ep50_.*s0.*'


    root = './log'

    path_list = []

    print(pattern_of_path)
    pattern_of_path = re.compile(pattern_of_path)
    plt.show()
    for (path, dir, files) in os.walk(root):
        if pattern_of_path.match(path):
            if not path.endswith('/cp'):  # ignore cp/ dir
                path_list.append(path)

    print(f'Result from {len(path_list)} paths:')

    for path in path_list:
        print(path)
        show_plt(path)


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--regex', type=str, default='', help='train condition regex')

    return parser.parse_args()

import csv
if __name__ == '__main__':
    import time

    st = time.time()
    args = parse_arguments(sys.argv[1:])
    main(args)
    print(f'time:{time.time() - st}')
