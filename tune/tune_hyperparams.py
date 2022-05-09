import os
import sys

sys.path.append('../')
sys.path.append('./')
import conf
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
# from utils.entropy_loss import *
import math
# from utils.augmentation import *
import random
import pandas as pd
import re
import multiprocessing as mp
import time
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.utils import pin_in_object_store, get_pinned_object
from ray.tune.suggest.hyperopt import HyperOptSearch

# log_suffix = '220309-6_kdh'
# log_suffix = '220315-6_single-stats-adaptive_harth'
# log_suffix = '220315-7_single-stats-adaptive_extrasensory'
# log_suffix = '220315-8_kdh_extrasensory'
# log_suffix = '220316_single-stats-adaptive_harth_simple-adaptive'
# log_suffix = '220319_cbrs-psuedo_eloss-dloss'
# log_suffix = '220319-2_cbrs-psuedo_eloss-dloss0'

log_suffix = None
method = None
dataset = None

memory = None
model = None
checkpoint_path = None
tgts = None

seed = 0
run_epoch = 100
ray_num_samples = 1  # ray samples
HARTH_domains = conf.HARTHOpt['src_domains'] #leave-one-user-out validation
ExtraSensory_domains = conf.ExtraSensoryOpt['src_domains'] #leave-one-user-out validation
RealLifeHar_domains = conf.RealLifeHAROpt['src_domains'] #leave-one-user-out validation
KITTI_MOT_domains  = conf.KITTI_MOT_Opt['val_domains']
# KITTI_SOT_domains  = conf.KITTI_SOT_Opt['val_domains']
KITTI_SOT_domains  = conf.KITTI_SOT_Opt['tgt_domains']
CIFAR10_domains  = conf.CIFAR10Opt['tgt_domains']

os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
device = torch.device("cuda:{:d}".format(0) if torch.cuda.is_available() else "cpu")

NUM_GPUS = 8


def main(args):
    global dataset, method, log_suffix, model, checkpoint_path, tgts, NUM_GPUS_PER_PROCESS
    dataset = args.dataset
    method = args.method
    log_suffix = args.log_suffix
    NUM_GPUS_PER_PROCESS = args.num_gpus_per_process

    if dataset == 'harth':
        model = 'HARTH_model'
        checkpoint_path = "220327-2_harth_minmax_scaling_all_split_win50_val"  # src-back, tgt-thigh ########### SELECTED'
        tgts = HARTH_domains
    elif dataset == 'extrasensory':
        model = 'ExtraSensory_model'
        checkpoint_path = "220327-2_extrasensory_selectedfeat_woutloc_std_scaling_all_win5_val"  ############ SELECTED
        tgts = ExtraSensory_domains
    elif dataset == 'reallifehar':
        model = 'RealLifeHAR_model'
        checkpoint_path = "220327-2_reallifehar_acc_minmax_scaling_all_win400_overlap0_val"  ############ SELECTED
        tgts = RealLifeHar_domains
    elif dataset == 'kitti_mot':
        model = 'KITTI_MOT_model'
        # checkpoint_path = "log/kitti_mot/Src/tgt_rain-100/220414-2_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_darknet_ep100_uex64_s0/cp/cp_last.pth.tar"  ############ SELECTED
        checkpoint_path = "log/kitti_mot/Src/tgt_rain-100-tgt/220417-1_kitti_mot_src-2d_obj_tgt-rain-100_lr1e-4_scratch_noaug_ep100_uex64_s0/cp/cp_last.pth.tar"  ############ SELECTED
        tgts = KITTI_MOT_domains
    elif dataset == 'kitti_sot':
        model = 'resnet50'
        # checkpoint_path = "log/kitti_sot/Src/src_2d_detection/tgt_2d_detection/220419-1_kitti_sot_src-2d_detection_tgt-2d_detection_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
        checkpoint_path = "log/kitti_sot/Src/src_original/tgt_original/220419-2_kitti_sot_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar" ####### Selected
        # checkpoint_path = "log/kitti_sot/Src/src_original-val/tgt_original-val/220419-3_kitti_sot_src-original-val_tgt-original-val_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
        tgts = KITTI_SOT_domains
    elif dataset == 'cifar10':
        model = 'wideresnet28-10'

        #      checkpoint_path="log/cifar10/Src/src_original/tgt_original/220421-1_cifar10_src-original_tgt-original_lr1e-4_pretrained_ep100_uex64_s0/cp/cp_last.pth.tar"
        # checkpoint_path = "log/cifar10/Src/src_original/tgt_original/220421-2_cifar10_src-original_tgt-original_lr1e-4_scratch_ep100_uex64_s0/cp/cp_last.pth.tar"  ### SELECTED
        checkpoint_path = "wideresnet28-10"  ### SELECTED
        tgts = CIFAR10_domains


    tune_with_ray(args)


def trainable_subprocess(config):
    import shlex
    import subprocess

    LOG_SUFFIX = "trainable_subprocess"
    os.chdir('/home/tsgong/git/WWW/')

    common_str = ''

    for param_name in config:
        if param_name in ['use_learned_stats', 'iabn']:
            common_str += f"--{param_name} " if config[param_name] else ''
        elif param_name in ['update_every_x']:
            common_str += f'--{param_name} {config[param_name]} '
            common_str += f'--{"memory_size"} {config[param_name]} ' # assert memsize = update_every_x
        else:
            common_str += f'--{param_name} {config[param_name]} '
    # print(common_str)

    runs = []
    for i in range(len(tgts)):
        # {i // NUM_GPUS_PER_PROCESS}
        run = f'python main.py ' \
              f'--gpu_idx {i % NUM_GPUS_PER_PROCESS} ' \
              f'--dataset {dataset} ' \
              f'--method {method} ' \
              f'--tgt {tgts[i]} ' \
              f'--log_suffix {LOG_SUFFIX} ' \
              f'--model {model} ' \
              f'--remove_cp ' \
              f'--online ' \
              f'--validation ' \
              f'--nsample 99999 ' \
              f'--tgt_train_dist 0 '
        if dataset in ['kitti_mot', 'kitti_sot', 'cifar10']:
            run+= f'--load_checkpoint_path {checkpoint_path} --validation ' + common_str
        else:
            run+= f'--load_checkpoint_path log/{dataset}/Src/tgt_{tgts[i]}/{checkpoint_path}_ep50_s0/cp/cp_last.pth.tar --validation ' + common_str


        runs.append(run)

    # print(shlex.split(t))
    plist = []
    for i in range(len(tgts)):
        plist.append(subprocess.Popen(shlex.split(runs[i]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      universal_newlines=True))

    result_dict = [[] for _ in range(run_epoch)]
    done = False
    current_epoch = 1

    while not done:
        for i, p in enumerate(plist):
            line = p.stdout.readline()
            # print(line, end='') ############################################################## uncomment this to debug

            matchObj = re.match('.*Online Eval.+Epoch:(\d+).+Accuracy:([\de\-\.]+).+', line)  # grab output

            if matchObj:
                epoch = int(matchObj.group(1))  # 1~100
                acc = float(matchObj.group(2))
                if epoch > 0:  # ignore epoch 0
                    # print('===========================================')
                    result_dict[epoch - 1].append(acc)
                    # print(result_dict)
                    # print('===========================================')

        ### report result
        target = result_dict[current_epoch - 1]
        if len(target) == len(plist):
            # print('###########AVG##############')
            avg = sum(target) / len(target)
            tune.report(acc=avg)
            current_epoch += 1

        ### Check done

        # 1. check if p is dead
        done = True
        for i, p in enumerate(plist):
            if p.poll() is None:
                done = False
                break

        if done == True:
            tune.report(dead=True)
        # 2. check if data is collected
        # done = True
        # for ele in result_dict:
        #     if len(ele) == len(plist):
        #         continue
        #     else:
        #         done = False
        #         break


def tune_with_ray(args):
    ## NOTE that directly passing heavy objects such as dataloader, fe, cc will slow down the training.
    # we cannot use "self" object in trainable, but variables outer loop can be used

    ray.init()

    # Shared default values. Specify if it's not in the main.
    config = {
    }

    grid_search = True
    if grid_search:
        ray_num_samples = 1
        search_alg = None
        search = tune.grid_search
    else:
        ray_num_samples = 100
        search_alg = HyperOptSearch(
            metric="acc", mode="max",
            random_state_seed=seed)
        search = tune.choice

    if method == 'Ours':
        # config['lr'] = search([0.1, 0.01, 0.001, 0.0001])
        config['lr'] = search([0.0001])
        # config['temperature'] = search([0.1, 0.5, 1, 4, 10, 20])
        config['temperature'] = search([1])
        # config['loss_scaler'] = search([0, 0.01, 0.1, 0.5, 1, 2, 10, 20, 100])
        config['loss_scaler'] = search([1])
        # config['loss_scaler'] = search([0])
        # config['bn_momentum'] = search([0.1, 0.01, 0.001, 0.0001])
        config['bn_momentum'] = search([0.01])
        # config['bn_momentum'] = tune.choice([0.1])

        # config['update_every_x'] = tune.choice([64])
        config['update_every_x'] = search([64])
        # config['update_every_x'] = search([10, 15, 20, 32, 64])

        # config['iabn_k'] = search([3, 4 , 5, 6, 7, 8, 9, 10, 15, 30, 50, 100, 1000, 10000, 100000])
        # config['iabn_k'] = search([7, 8, 9, 15, 30, 50, 100, 1000, 10000, 100000])
        config['iabn_k'] = search([10, 15, 30, 50, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e8])

        # config['memory_size'] = tune.choice([64])
        config['memory_type'] = tune.choice(['CBReservoir'])
        config['use_learned_stats'] = tune.choice([True])
        config['iabn'] = tune.choice([True])

    elif method == 'TT_SINGLE_STATS':
        ray_num_samples = 1
        search_alg = None
        config['bn_momentum']=tune.grid_search([0.1, 0.01, 0.001, 0.0001])

    elif method == 'TT_BATCH_STATS':
        ray_num_samples = 1
        # config['update_every_x'] = tune.grid_search([10, 15, 20, 32, 64])
        config['update_every_x'] = tune.grid_search([64])
        config['bn_momentum'] = tune.grid_search([0.1, 0.01, 0.001, 0.0001, 0.00001])
        config['use_learned_stats'] = tune.choice([True])
        search_alg = None

    elif method == 'TENT':
        ray_num_samples = 1
        search_alg = None

        config['lr'] = tune.grid_search([0.1, 0.01, 0.001, 0.0001])
        config['update_every_x'] = tune.choice([64])
        # config['memory_size'] = tune.choice([64])
        # config['use_learned_stats'] = tune.choice([True])

    elif method == 'COTTA':
        search_alg = None

        config['lr'] = search([0.1, 0.01, 0.001, 0.0001])
        config['aug_threshold'] = search([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
        config['restoration_factor'] = search([0.1, 0.01, 0.001, 0.0001])
        config['ema_factor'] = search([0.99, 0.995, 0.999, 0.9995, 0.9999])
        # augmentation confidence threshold is difficult to tune as it uses 5% percentile from source, we rather use the given value
        config['update_every_x'] = tune.choice([64])
        # config['memory_size'] = tune.choice([64])

        ###### Ours
        # config['bn_momentum'] = search([0.1, 0.01, 0.001, 0.0001, 0.00001])
        # config['memory_type'] = tune.choice(['CBFIFO'])
        # config['use_learned_stats'] = tune.choice([True])
    else:
        raise NotImplementedError

    scheduler = ASHAScheduler(
        metric="acc",
        mode="max",
        max_t=run_epoch,
        grace_period=run_epoch,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=['acc'])
    if dataset == 'kitti_mot':
        num_cpus_per_process = 8
    else:
        num_cpus_per_process = 1
    result = tune.run(trainable_subprocess,
                      resources_per_trial={"cpu": num_cpus_per_process, "gpu": NUM_GPUS_PER_PROCESS},
                      search_alg=search_alg,
                      # defaulting to 1 CPU, 0 GPU per trial By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
                      config=config,
                      num_samples=ray_num_samples,
                      scheduler=scheduler,
                      local_dir='~/git/WWW/ray_results',
                      fail_fast=False,  # To stop the entire Tune run as soon as any trial errors
                      name=log_suffix,
                      resume='AUTO',  # AUTO ERRORED_ONLY
                      progress_reporter=reporter,
                      max_failures=3)

    best_trial = result.get_best_trial("acc", "max", "last")

    f = open('/home/tsgong/git/WWW/ray_results/' + f'{log_suffix}.txt', 'w')

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial last acc: {}".format(
        best_trial.last_result["acc"]))
    f.write("Best trial config: {}\n".format(best_trial.config))
    f.write("Best trial last acc: {}\n".format(
        best_trial.last_result["acc"]))


def set_seed():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='',
                        help='dataset')
    parser.add_argument('--method', type=str, default='',
                        help='method')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='method')
    parser.add_argument('--num_gpus_per_process', type=int, default=4,
                        help='method')

    args = parser.parse_args()
    set_seed()
    main(args)
