import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF

import math
import conf
from .dnn import DNN
from torch.utils.data import DataLoader

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

DEBUG = True


class FeatMatch(DNN):
    def __init__(self, *args, **kwargs):
        super(FeatMatch, self).__init__(*args, **kwargs)
        self.source_train_procesing()
        self.cur_best_acc = 0.0
        self.optimizer1 = optim.Adam([{'params': self.feature_extractor.parameters()}], lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])
        self.optimizer2 = optim.Adam([{'params': self.class_classifier.parameters()}], lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])

        # print(summary(self.feature_extractor.cuda(), (6, 32)))

    def train(self, epoch):
        """
        Train the model
        """

        # setup models
        self.feature_extractor.train()
        self.class_classifier.train()

        class_loss_sum = 0.0
        total_iter = 0

        # load src, tgt data
        src_feats, src_cls, src_dls = self.source_train_set
        tgt_feats, tgt_cls, tgt_dls = self.target_train_set

        # set batch size
        # batch_size = 125
        print(f'src_dls is : {src_dls}')

        if len(tgt_feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()
        # conf.args.opt['batch_size']
        src_dataset = torch.utils.data.TensorDataset(src_feats, src_cls, src_dls)
        src_data_loader = DataLoader(src_dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=False,
                                     drop_last=True, pin_memory=False)

        tgt_dataset = torch.utils.data.TensorDataset(tgt_feats, tgt_cls, tgt_dls)
        tgt_data_loader = DataLoader(tgt_dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=True, pin_memory=False)


        num_iter = len(src_data_loader)
        total_iter += num_iter
        cur_best = 0.0
        num_steps = 1

        for i in range(num_steps):
            src_list = self.get_single_random_batch(src_dataset, conf.args.opt['batch_size'])
            src_feats = torch.concat([elem[0].unsqueeze(0) for elem in src_list])
            src_cls = torch.concat([elem[1].unsqueeze(0) for elem in src_list])

            # print(f'current input size is : {src_feats.size()}')
            # print(f'current cls size is : {src_cls.size()}')
            # print(f'current cls size is : {src_cls[0]}')

            src_feats = src_feats.to(device)
            src_cls = src_cls.to(device)

            tgt_list = self.get_single_random_batch(tgt_dataset, conf.args.opt['batch_size'])
            tgt_feats = torch.concat([elem[0].unsqueeze(0) for elem in tgt_list])
            tgt_cls = torch.concat([elem[1].unsqueeze(0) for elem in tgt_list])

            tgt_feats = tgt_feats.to(device)
            tgt_cls = tgt_cls.to(device)

            src_feature = self.feature_extractor(src_feats)
            tgt_feature = self.feature_extractor(tgt_feats)

            class_loss_of_labeled_data, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                                  self.class_criterion,
                                                                                  src_feature,
                                                                                  src_cls)

            # class_loss_of_target_data, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
            #                                                                       self.class_criterion,
            #                                                                       tgt_feature,
            #                                                                       tgt_cls)


            class_loss = class_loss_of_labeled_data
            class_loss_sum += float(class_loss * src_feats.size(0))
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            class_loss.backward(retain_graph=True)

            self.set_classifier_requires_grad(False)
            mmd_loss = self.MMD(src_feature.to(device), tgt_feature.to(device), "rbf")
            mmd_loss.backward()
            # class_loss_of_target_data.backward()
            self.set_classifier_requires_grad(True)

            self.optimizer1.step()
            self.optimizer2.step()


        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        print("validation")
        print(self.validation(epoch))
        avg_loss = class_loss_sum / total_iter
        return avg_loss


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
        src_feats, src_cls, src_dls = self.source_train_set
        tgt_feats, tgt_cls, tgt_dls = self.target_train_set

        current_sample = tgt_feats[current_num_sample - 1], tgt_cls[current_num_sample - 1], tgt_dls[current_num_sample - 1]
        self.mem.add_instance(current_sample)
        self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models
        self.feature_extractor.train()
        self.class_classifier.train()

        class_loss_sum = 0.0
        total_iter = 0

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)

        if len(feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()

        tgt_dataset = torch.utils.data.TensorDataset(feats, cls, dls)
        src_dataset = torch.utils.data.TensorDataset(src_feats, src_cls, src_dls)
        num_iter = 1
        num_steps = conf.args.num_steps

        for e in range(conf.args.epoch):
            for i in range(num_steps):
                total_iter += num_iter
                src_list = self.get_single_random_batch(src_dataset, min(len(tgt_dataset), conf.args.opt['batch_size']))
                src_feats = torch.concat([elem[0].unsqueeze(0) for elem in src_list])
                src_cls = torch.concat([elem[1].unsqueeze(0) for elem in src_list])

                src_feats = src_feats.to(device)
                src_cls = src_cls.to(device)

                tgt_list = self.get_single_random_batch(tgt_dataset, min(len(tgt_dataset), conf.args.opt['batch_size']))
                tgt_feats = torch.concat([elem[0].unsqueeze(0) for elem in tgt_list])
                tgt_cls = torch.concat([elem[1].unsqueeze(0) for elem in tgt_list])

                tgt_feats = tgt_feats.to(device)
                tgt_cls = tgt_cls.to(device)

                src_feature = self.feature_extractor(src_feats)
                tgt_feature = self.feature_extractor(tgt_feats)

                class_loss_of_labeled_data, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                                      self.class_criterion,
                                                                                      src_feature,
                                                                                      src_cls)

                class_loss = class_loss_of_labeled_data
                class_loss_sum += float(class_loss * src_feats.size(0))
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                class_loss.backward(retain_graph=True)

                self.set_classifier_requires_grad(False)
                mmd_loss = self.MMD(src_feature.to(device), tgt_feature.to(device), "rbf")
                mmd_loss.backward()
                # class_loss_of_target_data.backward()
                self.set_classifier_requires_grad(True)

                self.optimizer1.step()
                self.optimizer2.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        self.previous_train_loss = avg_loss

        return TRAINED

    # additional fucntions
    def source_train_procesing(self):
        # print(f'source_dataloader is currently : {len(self.source_dataloader[0])}')

        features = [x[0] for x in self.source_dataloader['train'].dataset]
        cl_labels = [x[1] for x in self.source_dataloader['train'].dataset]
        do_labels = [x[2] for x in self.source_dataloader['train'].dataset]
        self.source_train_set = (torch.stack(features),
                                 torch.stack(cl_labels),
                                 torch.stack(do_labels))

    def set_classifier_requires_grad(self, bool):
        for param in self.class_classifier.parameters():
            param.requires_grad = bool

    # https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    def MMD(self, x, y, kernel):
        """Emprical maximum mean discrepancy. The lower the result
           the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25, 30, 100]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)


    def get_single_random_batch(self, dataset, batch_size):
        ret_list = []
        print(f'len of dataset is : {len(dataset)}')
        print(f'batch_size is : {batch_size}')
        for index in random.sample(range(0, len(dataset)), batch_size):
            # print(dataset[index][2])
            ret_list.append(dataset[index])
        return ret_list
