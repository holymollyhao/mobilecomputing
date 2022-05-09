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
from .tt_whole import TT_WHOLE
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TT_BATCH_PARAMS(TT_WHOLE):

    def __init__(self, *args, **kwargs):
        super(TT_BATCH_PARAMS, self).__init__(*args, **kwargs)

        # turn on grad for BN params only
        for model in [self.feature_extractor, self.class_classifier]:

            for param in model.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False

            for module in model.modules():

                if isinstance(module, nn.BatchNorm1d):
                    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                    if conf.args.use_learned_stats:
                        module.track_running_stats = True
                        pass
                    else:
                        # With below, this module always uses the test batch statistics (no momentum)
                        module.track_running_stats = False
                        module.running_mean = None
                        module.running_var = None

                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

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
        self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                # self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models

        self.feature_extractor.train()
        self.class_classifier.train()

        if len(feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)

        dataset = torch.utils.data.TensorDataset(feats, cls, dls)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        entropy_loss = HLoss()
        kd_loss = KDLoss(temp_factor=conf.args.temperature)

        for e in range(conf.args.epoch):

            for batch_idx, labeled_data in enumerate(data_loader):
                feats, cls, _ = labeled_data
                feats = feats.to(device)
                _ = cls.to(device)

                if conf.args.dsbn:
                    raise NotImplementedError
                else:

                    preds_of_data = self.feature_extractor(feats)

                # loss = kd_loss(preds_of_data[:len(preds_of_data)//2], preds_of_data[len(preds_of_data)//2:]) #+ entropy_loss(preds_of_data)
                kdloss = kd_loss(preds_of_data[::2], preds_of_data[1::2])
                eloss = entropy_loss(preds_of_data)
                loss = kdloss + conf.args.loss_scaler * eloss
                # loss = kd_loss(preds_of_data[:len(preds_of_data)//2], preds_of_data[len(preds_of_data)//2:])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
