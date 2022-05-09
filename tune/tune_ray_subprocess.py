import os
import sys

sys.path.append('../')
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


class Object(object):
    pass


conf = Object()
conf.args = Object()
conf.args.dataset = ["hhar_scaled", "wesad_scaled", "metasense_activity_scaled", "metasense_speech_scaled"]
# conf.args.estimator_target_method = "FT_all"
conf.args.estimator_target_method = ["FT_all", "SHOT"]
conf.args.gpu_idx = 0
conf.args.seed = 0
conf.args.epoch = 100
conf.args.num_samples = 500  # ray samples
# conf.args.log_suffix = '220309-6_kdh'
# conf.args.log_suffix = '220315-6_single-stats-adaptive_harth'
# conf.args.log_suffix = '220315-7_single-stats-adaptive_extrasensory'
# conf.args.log_suffix = '220315-8_kdh_extrasensory'
# conf.args.log_suffix = '220316_single-stats-adaptive_harth_simple-adaptive'
# conf.args.log_suffix = '220319_cbrs-psuedo_eloss-dloss'
conf.args.log_suffix = '220319-2_cbrs-psuedo_eloss-dloss0'

os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

NUM_GPUS = 8
NUM_GPUS_PER_PROCESS = 2


def main():
    # init dataloader
    tune_with_ray()


def trainable_subprocess(config):
    import shlex
    import subprocess

    conf.args = config['args']
    EPOCH = conf.args.epoch

    LOG_SUFFIX = "trainable_subprocess"
    os.chdir('/home/tsgong/git/WWW/')

    common_str = f"--lr {config['lr']} --epoch {config['epoch']} --update_every_x {config['update_every_x']} --memory_size {config['update_every_x']} --temperature {config['temperature']} --loss_scaler {config['loss_scaler']} --bn_momentum {config['bn_momentum']}"
    # common_str = f"--bn_momentum_start {config['bn_momentum_start']} --bn_momentum_end {config['bn_momentum_end']} --bn_momentum_decay {config['bn_momentum_decay']}"

    tgts = ['S006', 'S010', 'S015', 'S017', 'S020', 'S023', 'S025', 'S026']
    # tgts = ['098A72A5-E3E5-4F54-A152-BBDA0DF7B694', '0A986513-7828-4D53-AA1F-E02D6DF9561B', '61976C24-1C50-4355-9C49-AAE44A7D09F6', '74B86067-5D4B-43CF-82CF-341B76BEA0F4', 'A5A30F76-581E-4757-97A2-957553A2C6AA', 'B9724848-C7E2-45F4-9B3F-A1F38D864495', 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC', 'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B']

    runs = []
    for i in range(8):
        # run = f'python main.py --gpu_idx {i//(NUM_GPUS//NUM_GPUS_PER_PROCESS)} --dataset extrasensory --method TT_SINGLE_STATS_ADAPTIVE --tgt {tgts[i]} --log_suffix {LOG_SUFFIX} --model ExtraSensory_model ' \
        #       f'--remove_cp --online --use_learned_stats --nsample 99999 --tgt_train_dist 0 --epoch 1 --update_every_x 1 --memory_size 1 --load_checkpoint_path log/extrasensory/Src/tgt_{tgts[i]}/220314-4_selectedfeat_woutloc_std_scaling_all_win5_ep50_s0/cp/ ' + common_str

        run = f'python main.py --gpu_idx {i // (NUM_GPUS // NUM_GPUS_PER_PROCESS)} --dataset harth --method Ours --tgt {tgts[i]} --log_suffix {LOG_SUFFIX} --model HARTH_model --memory_type CBRS ' \
              f'--remove_cp --online --use_learned_stats --nsample 99999 --tgt_train_dist 0 --epoch 1 --update_every_x 1 --memory_size 1 --load_checkpoint_path log/harth/Src/tgt_{tgts[i]}/220314-4_minmax_scaling_all_split_win50_ep50_s0/cp/ ' + common_str
        # run = f'python main.py --gpu_idx {i//(NUM_GPUS//NUM_GPUS_PER_PROCESS)} --dataset extrasensory --method TT_BATCH_PARAMS --tgt {tgts[i]} --log_suffix {LOG_SUFFIX} --model ExtraSensory_model ' \
        #       f'--remove_cp --online --use_learned_stats --nsample 99999 --tgt_train_dist 0 --seed 0 --load_checkpoint_path log/extrasensory/Src/tgt_{tgts[i]}/220314-4_selectedfeat_woutloc_std_scaling_all_win5_ep50_s0/cp/ ' + common_str

        runs.append(run)

    # print(shlex.split(t))
    plist = []
    for i in range(8):
        plist.append(subprocess.Popen(shlex.split(runs[i]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      universal_newlines=True))

    result_dict = [[] for _ in range(EPOCH)]
    done = False
    current_epoch = 1

    while not done:
        for i, p in enumerate(plist):
            line = p.stdout.readline()
            # print(line, end='') # uncomment this to debug

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


def tune_with_ray():
    ## NOTE that directly passing heavy objects such as dataloader, fe, cc will slow down the training.
    # we cannot use "self" object in trainable, but variables outer loop can be used

    ray.init()

    config = {
        'lr': tune.choice([0.1, 0.01, 0.001, 0.0001, 0.00005, 0.00001]),
        'update_every_x': tune.choice([16, 32, 64, 128]),
        'epoch': tune.choice([1, 5, 10]),
        'temperature': tune.choice([0.1, 0.5, 1, 4, 10, 20]),
        # 'loss_scaler': tune.choice([0, 0.01, 0.1, 0.5, 1, 2, 10, 20, 100]),
        'loss_scaler': tune.choice([0]),
        'bn_momentum': tune.choice([0.1, 0.01, 0.001, 0.0001]),

        # 'bn_momentum_start': tune.choice(
        #     [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]),
        # 'bn_momentum_end': tune.choice(
        #     [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]),
        # # sampling in log space and rounding to increments of 0.000005
        # 'bn_momentum_decay': tune.qloguniform(0.5, 1, 0.01),

        'args': conf.args,
        'path': os.getcwd() + '/',
        'seed': conf.args.seed,
    }

    current_best_params = [{
        #
        # 'bn_momentum_start': 0.01,
        # 'bn_momentum_end': 0.00001,
        # 'bn_momentum_decay': 0.95,

        'lr': 0.001,
        'update_every_x': 64,
        'epoch': 1,
        'temperature': 1,
        'loss_scaler': 0,
        'bn_momentum': 0.01,
    }]

    scheduler = ASHAScheduler(
        metric="acc",
        mode="max",
        max_t=conf.args.epoch,
        grace_period=100,
        reduction_factor=2)

    hyperopt_search = HyperOptSearch(
        metric="acc", mode="max",
        points_to_evaluate=current_best_params,
        random_state_seed=conf.args.seed)

    reporter = CLIReporter(
        metric_columns=['acc'])
    result = tune.run(trainable_subprocess,
                      resources_per_trial={"cpu": 1, "gpu": NUM_GPUS_PER_PROCESS},
                      search_alg=hyperopt_search,
                      # defaulting to 1 CPU, 0 GPU per trial By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
                      config=config,
                      num_samples=conf.args.num_samples,
                      scheduler=scheduler,
                      local_dir='~/git/WWW/ray_results',
                      fail_fast=False,  # To stop the entire Tune run as soon as any trial errors
                      name=conf.args.log_suffix,
                      resume='AUTO',
                      progress_reporter=reporter)

    best_trial = result.get_best_trial("acc", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial last acc: {}".format(
        best_trial.last_result["acc"]))


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed()
    main()
