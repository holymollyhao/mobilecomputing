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


class ChangedFeatMatch(DNN):
    def __init__(self, *args, **kwargs):
        super(ChangedFeatMatch, self).__init__(*args, **kwargs)
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

            # src_feats & src_cls creation
            src_feats = torch.concat([elem[0].unsqueeze(0) for elem in src_list])
            src_cls = torch.concat([elem[1].unsqueeze(0) for elem in src_list])
            src_feats = src_feats.to(device)
            src_cls = src_cls.to(device)

            tgt_list = self.get_single_random_batch(tgt_dataset, conf.args.opt['batch_size'])

            # tgt_feats & tgt_cls creation
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
            domain_batch_list = self.get_random_domain_batch(src_dataset, src_dls, 5)
            combination_list = list(itertools.combinations(range(len(domain_batch_list)), 2))
            for (index1,index2) in combination_list:
                domain1_feats = torch.concat([elem[0].unsqueeze(0) for elem in domain_batch_list[index1]])
                domain2_feats = torch.concat([elem[0].unsqueeze(0) for elem in domain_batch_list[index2]])

                domain1_feature = self.feature_extractor(domain1_feats.to(device))
                domain2_feature = self.feature_extractor(domain2_feats.to(device))

                # gradient update procedure
                mmd_loss = self.MMD(domain1_feature.to(device), domain2_feature.to(device), "rbf", len(combination_list))
                mmd_loss.backward(retain_graph=True)

            mmd_loss = self.MMD(src_feature.to(device), tgt_feature.to(device), "rbf", 1.0)
            mmd_loss.backward()
            self.set_classifier_requires_grad(True)

            self.optimizer1.step()
            self.optimizer2.step()


        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        print("validation")
        print(self.validation(epoch))
        avg_loss = class_loss_sum / total_iter
        return avg_loss

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
    def MMD(self, x, y, kernel, division_const):
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

        # return torch.mean(XX + YY - 2. * XY)
        return torch.div(torch.mean(XX + YY - 2. * XY), division_const)


    def get_single_random_batch(self, dataset, batch_size):
        ret_list = []
        # print(f'len of dataset is : {len(dataset)}')
        # print(f'batch_size is : {batch_size}')
        for index in random.sample(range(0, len(dataset)), batch_size):
            ret_list.append(dataset[index])
        return ret_list

    def get_random_domain_batch(self, dataset, src_dls, batch_size):
        ret_list = [[]] * len(conf.args.opt['src_domains'])
        bool_list = [False] * len(conf.args.opt['src_domains'])

        domain_list = src_dls.tolist()
        # print(f'domain_list is : {domain_list}')
        idx_list = []
        valid_domain = []
        for i in range(len(conf.args.opt['src_domains'])):
            try:
                idx_list.append(domain_list.index(i))
            except ValueError:
                idx_list.append(-1)
            if idx_list[-1] != -1:
                valid_domain.append(i)
        # print(f'valid domain is : {valid_domain}')
        for i in range(len(valid_domain) - 1):
            if( idx_list[valid_domain[i+1]] - idx_list[valid_domain[i]] >= batch_size):
                for index in random.sample(range(idx_list[valid_domain[i]], idx_list[valid_domain[i+1]]), batch_size):
                    ret_list[i].append(dataset[index])

        if(len(dataset) - idx_list[valid_domain[-1]]>=batch_size):
            for index in random.sample(range(idx_list[-1], len(dataset)), batch_size):
                ret_list[-1].append(dataset[index])
        # print(ret_list)
        return ret_list

