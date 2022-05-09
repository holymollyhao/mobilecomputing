import subprocess
import multiprocessing as mp
import itertools
import argparse
import re
import os
import numpy as np
import random
import torch
import conf
import sys

manager = mp.Manager()

gpus_available = manager.list([0, 0, 0, 0, 0, 0, 0, 0])  # use only GPUs with low temp

visible_devices = []
for i in range(torch.cuda.device_count()):
    visible_devices.append(str(i))
    gpus_available[i] = 1

torch.cuda.device_count()
visible_devices = ','.join(visible_devices)
print("visible_devices: {:s}".format(visible_devices))

from conf import HHAROpt
from conf import DSAOpt
from conf import ICHAROpt
from conf import ICSROpt
from conf import PAMAP2Opt
from conf import WESADOpt
from conf import OpportunityOpt
from conf import GaitOpt
from conf import HARTHOpt
from conf import RealLifeHAROpt
from conf import ExtraSensoryOpt
from conf import KITTI_MOT_Opt
from conf import KITTI_SOT_Opt
from conf import CIFAR10Opt
from conf import CIFAR100Opt
from conf import VLCSOpt
from conf import OfficeHomeOpt
from conf import PACSOpt


# gpus_available[0] =0
# gpus_available[1] =0
# gpus_available[2] =0
# gpus_available[3] =0
# gpus_available[4] =0
# gpus_available[5] =0
# gpus_available[6] =0
# gpus_available[7] =0


def init(l):
    global lock
    lock = l


def gpu_work(idx):
    global gpus_available

    lock.acquire()
    gpu_idx = gpus_available.index(max(gpus_available))
    gpus_available[gpu_idx] -= 1
    lock.release()

    newjob = re.sub('--gpu_idx 0', '--gpu_idx ' + str(gpu_idx), jobs_agg[idx])
    print(newjob)
    subprocess.call(newjob, shell=True)

    lock.acquire()
    gpus_available[gpu_idx] += 1
    lock.release()


def sub_parser_handler(args, script_args):
    jobs = []
    for method in args.method:
        test_domain_combinations = []
        selected_domain = 'src_domains' if script_args.validation else 'tgt_domains'

        if args.dataset in ['hhar', 'hhar', 'hhar_feature']:
            src_domains = HHAROpt['src_domains']
            test_domain_combinations = HHAROpt[selected_domain]

        elif args.dataset in ['ichar', 'ichar_feature']:

            src_domains = ICHAROpt['src_domains']
            test_domain_combinations = ICHAROpt[selected_domain]

        elif args.dataset in ['icsr', 'icsr_feature']:

            src_domains = ICSROpt['src_domains']
            test_domain_combinations = ICSROpt[selected_domain]

        elif args.dataset in ['wesad']:

            src_domains = WESADOpt['src_domains']
            test_domain_combinations = WESADOpt[selected_domain]

        elif args.dataset in ['harth']:

            src_domains = HARTHOpt['src_domains']
            test_domain_combinations = HARTHOpt[selected_domain]

        elif args.dataset in ['reallifehar']:

            src_domains = RealLifeHAROpt['src_domains']
            test_domain_combinations = RealLifeHAROpt[selected_domain]

        elif args.dataset in ['extrasensory']:

            src_domains = ExtraSensoryOpt['src_domains']
            test_domain_combinations = ExtraSensoryOpt[selected_domain]
        elif args.dataset in ['kitti_mot', 'kitti_mot_test']:

            src_domains = KITTI_MOT_Opt['src_domains']
            test_domain_combinations = KITTI_MOT_Opt[selected_domain]

        elif args.dataset in ['kitti_sot', 'kitti_sot_test']:

            src_domains = KITTI_SOT_Opt['src_domains']
            test_domain_combinations = KITTI_SOT_Opt[selected_domain]
        elif args.dataset in ['cifar10']:

            src_domains = CIFAR10Opt['src_domains']
            test_domain_combinations = CIFAR10Opt[selected_domain]
        elif args.dataset in ['cifar100']:

            src_domains = CIFAR100Opt['src_domains']
            test_domain_combinations = CIFAR100Opt[selected_domain]
        elif args.dataset in ['vlcs']:
            src_domains = VLCSOpt['src_domains']
            test_domain_combinations = VLCSOpt[selected_domain]

        elif args.dataset in ['officehome']:

            src_domains = OfficeHomeOpt['src_domains']
            test_domain_combinations = OfficeHomeOpt[selected_domain]

        elif args.dataset in ['pacs']:
            src_domains = PACSOpt['src_domains']
            test_domain_combinations = PACSOpt[selected_domain]

        elif args.dataset in ['dsa', 'dsa_feature']:

            users = DSAOpt['users']
            positions = DSAOpt['positions']

            test_domain_combinations = list(itertools.product(users, positions))
            test_domain_combinations = [user + '.' + position for user, position in test_domain_combinations]

        elif args.dataset in ['opportunity']:

            # users = OpportunityOpt['users']
            users = ['']
            positions = OpportunityOpt['positions']

            test_domain_combinations = list(itertools.product(users, positions))
            test_domain_combinations = [user + '.' + position for user, position in test_domain_combinations]

        elif args.dataset in ['gait']:

            # users = GaitOpt['users']
            users = ['']
            positions = GaitOpt['positions']

            test_domain_combinations = list(itertools.product(users, positions))
            test_domain_combinations = [user + '.' + position for user, position in test_domain_combinations]


        elif args.dataset in ['pamap2', 'pamap2_feature']:
            users = PAMAP2Opt['users']
            # users = ['']
            # positions = PAMAP2Opt['positions']
            positions = ['']
            test_domain_combinations = list(itertools.product(users, positions))
            test_domain_combinations = [user + '.' + position for user, position in test_domain_combinations]

        ########################################################################################

        if args.method[0] == 'Estimator':
            cmd_str = 'CUDA_VISIBLE_DEVICES={:s} python main.py ' \
                      '--gpu_idx 0 ' \
                      '--dataset {:s} ' \
                      '--method {:s} ' \
                      '--epoch {:d} ' \
                      '--log_suffix {:s} '.format(visible_devices, args.dataset, method,
                                                  args.epoch, args.log_suffix)

            jobs.append(cmd_str)

        for domain in test_domain_combinations:
            print(visible_devices)
            print(args.dataset)
            cmd_str = 'CUDA_VISIBLE_DEVICES={:s} python main.py ' \
                      '--gpu_idx 0 ' \
                      '--dataset {:s} ' \
                      '--method {:s} ' \
                      '--tgt {:s} ' \
                      '--log_suffix {:s} '.format(visible_devices, args.dataset, method, domain, args.log_suffix)

            jobs.append(cmd_str)


    #### arguments with single input



    if script_args.validation:
        for i in range(len(jobs)):
            jobs[i] += ' --validation'

    if args.model:
        for i in range(len(jobs)):
            jobs[i] += ' --model ' + args.model


    ## add boolean args
    for k, v in vars(args).items():
        if isinstance(v, bool) and v==True:
            for i in range(len(jobs)):
                jobs[i] += f' --{k}'




    ## add list args
    if len(args.nsample) > 0:
        nsample_combinations = list(itertools.product(jobs, args.nsample))
        jobs = [job + ' --nsample ' + lr for job, lr in nsample_combinations]

        for i in range(len(jobs)):
            nsample = re.search('--nsample ([0-9\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # \S+: anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)',
                             '--log_suffix ' + log_suffix.group(1) + '_' + nsample.group(1) + 'sample', jobs[i])

    if len(args.num_source) > 0:
        nsource_combinations = list(itertools.product(jobs, args.num_source))
        jobs = [job + ' --num_source ' + n for job, n in nsource_combinations]

        for i in range(len(jobs)):
            nsource = re.search('--num_source ([0-9\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)',
                             '--log_suffix ' + log_suffix.group(1) + '_' + nsource.group(1) + 'source', jobs[i])

    if len(args.lr) > 0:
        lr_combinations = list(itertools.product(jobs, args.lr))
        jobs = [job + ' --lr ' + lr for job, lr in lr_combinations]

        for i in range(len(jobs)):
            lr = re.search('--lr ([0-9\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_' + lr.group(1),
                             jobs[i])

    if len(args.tgt_train_dist) > 0:
        combinations = list(itertools.product(jobs, args.tgt_train_dist))
        jobs = [job + ' --tgt_train_dist ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--tgt_train_dist ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_dist' + found.group(1),
                             jobs[i])

    if len(args.cls_par) > 0:
        combinations = list(itertools.product(jobs, args.cls_par))
        jobs = [job + ' --cls_par ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--cls_par ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_beta' + found.group(1),
                             jobs[i])

    if len(args.threshold) > 0:
        combinations = list(itertools.product(jobs, args.threshold))
        jobs = [job + ' --threshold ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--threshold ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_thr' + found.group(1),
                             jobs[i])


    if len(args.epoch) > 0:
        combinations = list(itertools.product(jobs, args.epoch))
        jobs = [job + ' --epoch ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--epoch ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_ep' + found.group(1),
                             jobs[i])

    if len(args.update_every_x) > 0:
        combinations = list(itertools.product(jobs, args.update_every_x))
        jobs = [job + ' --update_every_x ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--update_every_x ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_uex' + found.group(1),
                             jobs[i])



    if len(args.memory_size) > 0:
        combinations = list(itertools.product(jobs, args.memory_size))
        jobs = [job + ' --memory_size ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--memory_size ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_mem' + found.group(1),
                             jobs[i])

    if len(args.memory_type) > 0:
        combinations = list(itertools.product(jobs, args.memory_type))
        jobs = [job + ' --memory_type ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--memory_type ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_mt' + found.group(1),
                             jobs[i])


    if len(args.bn_momentum) > 0:
        combinations = list(itertools.product(jobs, args.bn_momentum))
        jobs = [job + ' --bn_momentum ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--bn_momentum ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_bnm' + found.group(1),
                             jobs[i])

    if len(args.bn_momentum_start) > 0:
        combinations = list(itertools.product(jobs, args.bn_momentum_start))
        jobs = [job + ' --bn_momentum_start ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--bn_momentum_start ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_bnms' + found.group(1),
                             jobs[i])

    if len(args.bn_momentum_end) > 0:
        combinations = list(itertools.product(jobs, args.bn_momentum_end))
        jobs = [job + ' --bn_momentum_end ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--bn_momentum_end ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_bnme' + found.group(1),
                             jobs[i])

    if len(args.bn_momentum_decay) > 0:
        combinations = list(itertools.product(jobs, args.bn_momentum_decay))
        jobs = [job + ' --bn_momentum_decay ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--bn_momentum_decay ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_bnmd' + found.group(1),
                             jobs[i])

    if len(args.temperature) > 0:
        combinations = list(itertools.product(jobs, args.temperature))
        jobs = [job + ' --temperature ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--temperature ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_T' + found.group(1),
                             jobs[i])

    if len(args.loss_scaler) > 0:
        combinations = list(itertools.product(jobs, args.loss_scaler))
        jobs = [job + ' --loss_scaler ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--loss_scaler ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_ls' + found.group(1),
                             jobs[i])


    if len(args.aug_threshold) > 0:
        combinations = list(itertools.product(jobs, args.aug_threshold))
        jobs = [job + ' --aug_threshold ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--aug_threshold ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_at' + found.group(1),
                             jobs[i])


    if len(args.restoration_factor) > 0:
        combinations = list(itertools.product(jobs, args.restoration_factor))
        jobs = [job + ' --restoration_factor ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--restoration_factor ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_rf' + found.group(1),
                             jobs[i])


    if len(args.ema_factor) > 0:
        combinations = list(itertools.product(jobs, args.ema_factor))
        jobs = [job + ' --ema_factor ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--ema_factor ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_em' + found.group(1),
                             jobs[i])

    if len(args.iabn_k) > 0:
        combinations = list(itertools.product(jobs, args.iabn_k))
        jobs = [job + ' --iabn_k ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--iabn_k ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_k' + found.group(1),
                             jobs[i])


    if len(args.sigma2_b_thres) > 0:
        combinations = list(itertools.product(jobs, args.sigma2_b_thres))
        jobs = [job + ' --sigma2_b_thres ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--sigma2_b_thres ([0-9a-zA-Z\.\-]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_thr' + found.group(1),
                             jobs[i])
    if len(args.skip_thres) > 0:
        combinations = list(itertools.product(jobs, args.skip_thres))
        jobs = [job + ' --skip_thres ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--skip_thres ([0-9a-zA-Z\.\-]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_sth' + found.group(1),
                             jobs[i])








    ###########################################################################
    #### seed naming comes last
    if len(args.seed) > 0:

        combinations = list(itertools.product(jobs, args.seed))
        jobs = [job + ' --seed ' + item for job, item in combinations]

        for i in range(len(jobs)):
            found = re.search('--seed ([0-9a-zA-Z\.]+)', jobs[i])
            log_suffix = re.search('--log_suffix (\S+)', jobs[i])  # anything but whitespace
            jobs[i] = re.sub('--log_suffix (\S+)', '--log_suffix ' + log_suffix.group(1) + '_s' + found.group(1),
                             jobs[i])

    ##### after making all log suffix

    if args.load_checkpoint_suffix:
        # if args.method in ['FT_FC' in args.method or 'FT_all' in args.method or 'Src' in args.method or 'CDAN' or 'SHOT' in args.method:
        for i in range(len(jobs)):

            # log_suffix = re.search('--log_suffix (\S+)', jobs[i]).group(1)
            # seed = re.search('(_s\d+)', log_suffix).group(1)
            # print(src,tgt)

            if args.load_checkpoint_suffix == 'darknet':
                jobs[i] += ' --load_checkpoint_path ' + 'log/imagenet/darknet53.conv.74'
            elif args.load_checkpoint_suffix == 'yolov3-kitti':
                jobs[i] += ' --load_checkpoint_path ' + 'log/imagenet/yolov3-kitti.weights'
            elif args.load_checkpoint_suffix.startswith('res') or args.load_checkpoint_suffix.startswith('wideres'):
                jobs[i] += ' --load_checkpoint_path ' + args.load_checkpoint_suffix
            elif args.load_checkpoint_suffix.endswith('.pth.tar'):
                jobs[i] += ' --load_checkpoint_path ' + args.load_checkpoint_suffix
            else:
                tgt = re.search('--tgt (\S+)', jobs[i]).group(1)
                jobs[
                    i] += ' --load_checkpoint_path log/' + args.dataset + '/Src/' + 'tgt_' + tgt + '/' + args.load_checkpoint_suffix + '/cp/cp_last_fused.pth.tar'


    return jobs


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def set_seed(seed):
#     random.seed(seed)

if __name__ == '__main__':

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--script', default=[], nargs='*',
                             help='script list')
    main_parser.add_argument('--num_concurrent', type=int, default=2,
                             help='number of concurrent processes per gpu')
    main_parser.add_argument('--log_as_file', action='store_true',
                             help='Write results as file')
    main_parser.add_argument('--script_seed', type=int, default=0,
                             help='random seed for script')
    main_parser.add_argument('--validation', action='store_true', help='Use validation data instead of test data for hyperparameter tuning')

    sub_parser = argparse.ArgumentParser()
    sub_parser.add_argument('--dataset', type=str, default='',
                            help='Dataset to be used, in [hhar, hhar, dsa, opportunity, gait, pamap2, ichar, icsr, mnistall]')
    sub_parser.add_argument('--model', type=str, default='HHAR_model',
                            help='Which model to use')
    sub_parser.add_argument('--method', default=[], nargs='*',
                            help='Dataset to be used, in [Src, DANN-general-L-personal-U]')
    sub_parser.add_argument('--src', type=str, default=None,
                            help='rest'
                                 'all (metasense extension)')
    sub_parser.add_argument('--log_suffix', type=str, default='',
                            help='log_suffix')
    sub_parser.add_argument('--load_checkpoint_suffix', type=str, default=None,
                            help='load checkpoint with the same log_suffix')
    sub_parser.add_argument('--lr', default=[], nargs='*',
                            help='task-learning rates')
    sub_parser.add_argument('--nsample', default=[], nargs='*',
                            help='How many samples do you want use for train')
    sub_parser.add_argument('--nvalsample', default=[], nargs='*',
                            help='How many samples do you want use for validation')
    sub_parser.add_argument('--num_source', default=[], nargs='*',
                            help='number of available sources')
    sub_parser.add_argument('--remove_cp', action='store_true',
                            help='Remove checkpoints after evaluation')
    sub_parser.add_argument('--tsne', action='store_true',
                            help='draw tsne for target validation set')
    sub_parser.add_argument('--seed', default=[], nargs='*',
                            help='random seed')

    sub_parser.add_argument('--tgt_train_dist', default=[], nargs='*',
                            help='0: random selection'
                                 '1: uniform selection'
                                 '2: constant selection'
                            )

    ### SHOT ###
    sub_parser.add_argument('--cls_par', default=[], nargs='*', help='balancing hyperparameter')
    sub_parser.add_argument('--threshold', default=[], nargs='*', help='threshold for determining clusters')

    ### WWW ###
    sub_parser.add_argument('--online', action='store_true', help='training via online learning?')
    sub_parser.add_argument('--update_every_x', default=[], nargs='*',
                            help='number of target samples used for every update')
    sub_parser.add_argument('--epoch', default=[], nargs='*', help='How many epochs do you want to use for train')
    sub_parser.add_argument('--memory_size', default=[], nargs='*',
                            help='number of previously trained data to be used for training')
    sub_parser.add_argument('--memory_type', default=[], nargs='*',
                        help='FIFO, CBRS')


    ### TT ###
    sub_parser.add_argument('--bn_momentum', default=[], nargs='*', help='momentum for tt_single_stats')
    sub_parser.add_argument('--bn_momentum_start', default=[], nargs='*', help='momentum for tt_single_stats')
    sub_parser.add_argument('--bn_momentum_end', default=[], nargs='*', help='momentum for tt_single_stats')
    sub_parser.add_argument('--bn_momentum_decay', default=[], nargs='*', help='momentum for tt_single_stats')
    #CoTTA
    sub_parser.add_argument('--ema_factor', default=[], nargs='*',
                        help='hyperparam for CoTTA')
    sub_parser.add_argument('--restoration_factor', default=[], nargs='*',
                        help='hyperparam for CoTTA')
    sub_parser.add_argument('--aug_threshold', default=[], nargs='*',
                        help='hyperparam for CoTTA')


    ## Ours
    sub_parser.add_argument('--dsbn', action='store_true', help='Apply domain-specific batch norm')
    sub_parser.add_argument('--use_learned_stats', action='store_true', help='Use learned stats for tt_batch_params')
    sub_parser.add_argument('--adapt_then_eval', action='store_true', help='Evaluation after adaptation - unrealistic and causing additoinal latency, but common in TTA.')

    sub_parser.add_argument('--src_sep', action='store_true', help='Separate domains for source')
    sub_parser.add_argument('--src_sep_noshuffle', action='store_true', help='Separate domains for source')
    sub_parser.add_argument('--temperature', default=[], nargs='*',
                            help='temperature for KDLoss')
    sub_parser.add_argument('--loss_scaler', default=[], nargs='*',
                            help='loss_scaler for entropy_loss')
    sub_parser.add_argument('--no_optim', action='store_true', help='no optimization')
    sub_parser.add_argument('--update_all', action='store_true', help='update all parameters')
    sub_parser.add_argument('--iabn', action='store_true', help='replace bn with iabn layer')
    sub_parser.add_argument('--iabn_k', default=[], nargs='*', help='k for iabn')
    sub_parser.add_argument('--use_in', action='store_true',
                            help='use IN stats instead of BN stats in IABN for threshold')
    sub_parser.add_argument('--sigma2_b_thres', default=[], nargs='*',
                            help='sigma2_b threshold to discard adjustment')
    sub_parser.add_argument('--skip_thres', default=[], nargs='*',
                            help='skip threshold to discard adjustment')

    sub_parser.add_argument('--fuse_model', action='store_true', default=False, help='fusing and training models')

    main_args = main_parser.parse_args()

    set_seed(main_args.script_seed)  # set seed

    sub_arg_list = []
    for script in main_args.script:
        args = sub_parser.parse_args(script.split(' '))
        sub_arg_list.append(args)

    for i in range(len(gpus_available)):
        gpus_available[i] *= main_args.num_concurrent
    num_gpus = sum([1 if i > 0 else 0 for i in gpus_available])

    print('num_gpus:{:d}'.format(num_gpus))
    print('gpus_available:' + str(gpus_available))

    jobs_agg = []
    for args in sub_arg_list:
        jobs = sub_parser_handler(args, main_args)
        jobs_agg += jobs
    # jobs_agg = sorted(jobs_agg)

    if main_args.log_as_file:
        raw_log_path = 'raw_logs/'
        if not os.path.exists(raw_log_path):  # create if it doesn't exist
            oldumask = os.umask(0)
            os.makedirs(raw_log_path, 0o777)
            os.umask(oldumask)

        for i in range(len(jobs_agg)):
            log_suffix = re.search('--log_suffix (\S+)', jobs_agg[i]).group(1)
            jobs_agg[i] += ' 2>&1 | tee {:s} '.format(
                raw_log_path + args.dataset + '_' + log_suffix + '_job' + str(i) + '.txt')

    print('Number of TOTAL jobs: ' + str(len(jobs_agg)))
    for job in jobs_agg:
        print(job)

    print('===========================')
    l = mp.Lock()
    pool = mp.Pool(processes=num_gpus * main_args.num_concurrent, initializer=init, initargs=(l,))
    pool.map(gpu_work, range(len(jobs_agg)), chunksize=1)
    pool.close()
    pool.join()
