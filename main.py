# -*- coding: utf-8 -*-
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import time

import conf

import os

from tensorboard_logger import Tensorboard

import torchvision

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model



def get_path():
    path = 'log/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type
    path += conf.args.method + '/'

    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    if conf.args.tgt:
        path += 'tgt_' + conf.args.tgt + '/'

    path += conf.args.log_suffix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyper parameters #################
    if 'dogwalk' == conf.args.dataset:
        opt = conf.DogwalkOpt
    elif 'dogwalk_win100' == conf.args.dataset:
        opt = conf.DogwalkOpt
    elif 'dogwalk_all' == conf.args.dataset:
        opt = conf.DogwalkAllOpt
    elif 'dogwalk_all_win100' == conf.args.dataset:
        opt = conf.DogwalkAll_WIN100_Opt
    elif 'dogwalk_all_win5' == conf.args.dataset:
        opt = conf.DogwalkAll_WIN5_Opt


    conf.args.opt = opt
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr

    # if 'icsr' in conf.args.dataset and conf.args.method in ['Src', 'Tgt', 'Src_Tgt', 'FT_FC', 'FT_all']:  # prevent high fluctuation for non-meta learning method
    #     conf.args.opt['learning_rate'] = conf.args.opt['learning_rate'] * 0.1

    model = None
    if conf.args.model == "Dogwalk_model":
        import models.Dogwalk_model as model
    elif conf.args.model == "Dogwalk_model_win100":
        import models.Dogwalk_model_win100 as model
    elif conf.args.model == "Dogwalk_model_win5":
        import models.Dogwalk_model_win5 as model


    # import modules after setting the seed
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    from learner.tent import TENT
    from learner.lame import LAME

    result_path, checkpoint_path, log_path = get_path()
    tensorboard = Tensorboard(log_path)

    ########## Dataset loading ############################
    source_data_loader = None
    target_data_loader = None
    learner = None

    if conf.args.method in ['Src']:
        learner_method = DNN
    elif 'TENT' in conf.args.method:
        learner_method = TENT
    elif 'LAME' in conf.args.method:
        learner_method = LAME
    else:
        raise NotImplementedError

    if conf.args.method in ['Src', 'TENT', 'T3A', 'LAME', 'LAME_vote', 'TENT_vote', 'LAME_vote_gt']:

        '''
        src_train = source_data_loader['train'].dataset
        tgt_train = target_data_loader['train'].dataset
        tgt_test = target_data_loader['test'].dataset
        state = {}
        state['src_train'] = src_train
        state['tgt_train'] = tgt_train
        state['tgt_test'] = tgt_test
        exit(0)
        '''

        print('##############Source Data Loading...##############')
        source_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.src,
                                                            conf.args.opt['file_path'],
                                                            batch_size=conf.args.opt['batch_size'],
                                                            valid_split=0,  # to be used for the validation
                                                            test_split=0,
                                                            separate_domains=conf.args.src_sep, is_src=True,
                                                            num_source=conf.args.num_source)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(conf.args.dataset, conf.args.tgt,
                                                            conf.args.opt['file_path'],
                                                            batch_size=conf.args.opt['batch_size'],
                                                            valid_split=0,
                                                            test_split=0,
                                                            separate_domains=False, is_src=False,
                                                            num_source=conf.args.num_source)
        print(source_data_loader)
        print(len(source_data_loader))
        learner = learner_method(model, tensorboard=tensorboard, source_dataloader=source_data_loader,
                                 target_dataloader=target_data_loader, write_path=log_path)


    #################### Training #########################

    since = time.time()

    # make dir if doesn't exist
    if not os.path.exists(result_path):
        oldumask = os.umask(0)
        os.makedirs(result_path, 0o777)
        os.umask(oldumask)
    if not os.path.exists(checkpoint_path):
        oldumask = os.umask(0)
        os.makedirs(checkpoint_path, 0o777)
        os.umask(oldumask)
    for arg in vars(conf.args):
        tensorboard.log_text('args/' + arg, getattr(conf.args, arg), 0)
    script = ' '.join(sys.argv[1:])
    tensorboard.log_text('args/script', script, 0)

    if conf.args.online == False:

        start_epoch = 1
        best_acc = -9999
        best_epoch = -1
        test_epoch = 1

        for epoch in range(start_epoch, conf.args.epoch + 1):
            learner.train(epoch)


        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result(is_train_offline=True) # eval with final model

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    elif conf.args.online == True:

        current_num_sample = 1
        num_sample_end = conf.args.nsample
        best_acc = -9999
        best_epoch = -1

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        finished = False
        while not finished and current_num_sample < num_sample_end:

            ret_val = learner.train_online(current_num_sample)

            if ret_val == FINISHED:
                break
            elif ret_val == SKIPPED:
                pass
            elif ret_val == TRAINED:
                pass
            current_num_sample += 1


        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result()

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    tensorboard.close()

    if conf.args.remove_cp:
        best_path = checkpoint_path + 'cp_best.pth.tar'
        last_path = checkpoint_path + 'cp_last.pth.tar'
        try:
            os.remove(best_path)
            os.remove(last_path)
        except Exception as e:
            pass


def parse_arguments(argv):
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ###MANDATORY###

    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used')

    parser.add_argument('--model', type=str, default='HHAR_model',
                        help='Which model to use')

    parser.add_argument('--method', type=str, default='',
                        help='method to be used')

    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgt', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    ###Optional###
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nsample', type=int, default=99999,
                        help='How many samples do you want use for train')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Suffix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')
    parser.add_argument('--tsne', action='store_true',
                        help='draw tsne for target validation set')
    parser.add_argument('--data_gen', action='store_true',
                        help='generate training data with source for training estimator')

    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')
    parser.add_argument('--online', action='store_true', help='training via online learning?')
    parser.add_argument('--update_every_x', type=int, default=1, help='number of target samples used for every update')
    parser.add_argument('--memory_size', type=int, default=1,
                        help='number of previously trained data to be used for training')
    parser.add_argument('--validation', action='store_true', help='Use validation data instead of test data for hyperparameter tuning')
    parser.add_argument('--src_sep', action='store_true', help='Separate domains for source')
    parser.add_argument('--memory_type', type=str, default='FIFO', help='FIFO, CBRS')

    #### Distribution ####
    parser.add_argument('--tgt_train_dist', type=int, default=0,
                        help='0: real selection'
                             '1: random selection'
                             '2: sorted selection'
                             '3: uniform selection'
                             '4: dirichlet distribution'
                        )

    parser.add_argument('--shuffle_instances', action='store_true', help='whether to shuffle classes or data instances')
    parser.add_argument('--dummy', action='store_true', default=False, help='do nothing')
    parser.add_argument('--log_percentile', action='store_true', help='percentile logging process')

    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()

