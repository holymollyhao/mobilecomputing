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


class VOTE(TT_WHOLE):


    def train(self, epoch):
        raise NotImplementedError

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

        # eval_online called every single time

        assert (conf.args.update_every_x == conf.args.memory_size) # == update batch_size
        assert (conf.args.epoch == 1)  # should be 1

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data
                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        self.evaluation_online(current_num_sample, '', self.mem.get_memory())
        self.batch_instances = []
        ################################ TT_Single_stats: Mazankiewicz, A., Böhm, K., & Bergés, M. (2020). Incremental Real-Time Personalization In Human Activity Recognition Using Domain Adaptive Batch Normalization. ArXiv, 4(4). https://doi.org/10.1145/3432230

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
