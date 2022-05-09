import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist
import math
import conf
from .dnn import DNN
from torch.utils.data import DataLoader
from .tt_whole import TT_WHOLE

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TT_SINGLE_STATS_ADAPTIVE(TT_WHOLE):


    def __init__(self, *args, **kwargs):
        super(TT_SINGLE_STATS_ADAPTIVE, self).__init__(*args, **kwargs)

        self.equal_counter = 1
        self.prev_cls = -1

    def train_online(self, current_num_sample):
        """
        Train the model
        """
        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        self.mem.add_instance(current_sample)
        current_class = cls[current_num_sample - 1].detach().cpu().numpy()
        if current_class==self.prev_cls:
            self.equal_counter+=1
        else:
            self.equal_counter=1

        online_adapt_momentum = conf.args.bn_momentum_start * (conf.args.bn_momentum_decay**(self.equal_counter-1))
        online_adapt_momentum = online_adapt_momentum if online_adapt_momentum > conf.args.bn_momentum_end else conf.args.bn_momentum_end
        conf.args.bn_momentum = online_adapt_momentum
        # assert(conf.args.bn_momentum_start >= conf.args.bn_momentum_end)

        self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        self.prev_cls = current_class

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED


        ################################ TT_Single_stats: Mazankiewicz, A., Böhm, K., & Bergés, M. (2020). Incremental Real-Time Personalization In Human Activity Recognition Using Domain Adaptive Batch Normalization. ArXiv, 4(4). https://doi.org/10.1145/3432230
        assert (conf.args.epoch == 1)  # should be 1 for tta?
        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

