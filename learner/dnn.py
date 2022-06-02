import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
import math
import conf
from copy import deepcopy
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
import json
import pickle
from torch.optim.lr_scheduler import StepLR
import os
from collections import OrderedDict
from utils import memory
from utils.logging import *
import torchvision

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

class l2_distance_list():
    def __init__(self, length):
        self.data = []
        self.max_len = length
    def append(self, item):
        self.data.append(item)
        if len(self.data) > self.max_len:
            self.data = self.data[1:]
        self.data.append(item)
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
    def clear(self):
        self.data = []


class DNN():
    def __init__(self, model, tensorboard, source_dataloader, target_dataloader, write_path):
        self.device = device
        self.tensorboard = tensorboard

        # init dataloader
        self.source_dataloader = source_dataloader

        if conf.args.src_sep:
            self.num_src_domains = len(self.source_dataloader)
            self.source_iters = [iter(dataloader['train']) for dataloader in self.source_dataloader]

        self.target_dataloader = target_dataloader

        if conf.args.dataset in ['cifar10', 'cifar100', 'vlcs', 'officehome', 'pacs'] and conf.args.tgt_train_dist == 0:
            self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        else:
            self.tgt_train_dist = conf.args.tgt_train_dist
        self.target_data_processing()

        self.write_path = write_path

        ################## Init & prepare model###################
        self.conf_list = []

        # Load model
        self.net = model.Net()
        self.net.to(device)

        if conf.args.load_checkpoint_path:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                weight_decay=conf.args.opt['weight_decay'])

        self.class_criterion = nn.CrossEntropyLoss()
        self.freeze_layers()  # this will call overriden method



        # online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBRS':
            # self.mem = memory.CBRS(capacity=conf.args.memory_size)
            self.mem = memory.CBRS_debug(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'Reservoir':
            self.mem = memory.Reservoir(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'Diversity':
            self.mem = memory.Diversity(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBFIFO':
            self.mem = memory.CBFIFO_debug(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'CBReservoir':
            self.mem = memory.CBReservoir_debug(capacity=conf.args.memory_size)

        self.json = {}
        self.entropy = []
        self.bef_cls = -10
        self.bef_gt_cls = -10
        self.bef_feat_list = l2_distance_list(32)
        self.bef_feat = []
        self.l2_dist_list = []

    def freeze_bn(self): #TODO: what is this?
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_layers(self):
        if 'FT_FC' in conf.args.method:  # transfer learning
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):
            features.append(feat.squeeze(0))# batch size is 1
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        num_class = conf.args.opt['num_class']

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        # real distribution
        if self.tgt_train_dist == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        if self.tgt_train_dist == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        elif self.tgt_train_dist == 2:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

            indices = np.argsort(result_cl_labels)

            result_feats = [result_feats[i] for i in indices]
            result_cl_labels = [result_cl_labels[i] for i in indices]
            result_do_labels = [result_do_labels[i] for i in indices]

        # uniform selection
        elif self.tgt_train_dist == 3:

            samples_per_class = [[] for _ in range(num_class)]

            for c in range(num_class):  # make data for each class
                indices_c = [i for i, x in enumerate(cl_labels) if x == c]
                for i in indices_c:
                    samples_per_class[c].append((features[i], cl_labels[i], do_labels[i]))

            c = 0
            num_added_samples = 0

            while True:
                if len(samples_per_class[c]) > 0:
                    sample = samples_per_class[c].pop(0)
                    result_feats.append(sample[0])
                    result_cl_labels.append(sample[1])
                    result_do_labels.append(sample[2])
                    num_added_samples += 1
                if num_added_samples >= conf.args.nsample or sum(
                        [len(samples_per_class[c]) for c in range(num_class)]) == 0:
                    break
                c = (c + 1) % num_class
                # Note that if it is running out of samples, it just appends the remaining classes

            features = []
            cl_labels = []
            do_labels = []

            for c in range(num_class):  # store back remaining data
                while len(samples_per_class[c]) != 0:
                    sample = samples_per_class[c].pop(0)
                    features.append(sample[0])
                    cl_labels.append(sample[1])
                    do_labels.append(sample[2])
        # dirichlet distribution
        elif self.tgt_train_dist == 4:
            dirichlet_numchunks = conf.args.opt['num_class']
            
            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
            min_size = 0
            N = len(features)

            while min_size < 10:  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
                for k in range(num_class):
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    # balance
                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []
            if conf.args.shuffle_instances:
                # case 2: create temporally correlated toy dataset by shuffling data instances
                cl_labels_np = torch.Tensor(cl_labels).numpy()
                for chunk in idx_batch_cls:
                    data = []
                    for cls in range(num_class):
                        idx = chunk[cls]
                        data.extend([(features[i], cl_labels_np[i], do_labels[i]) for i in idx])
                    np.random.shuffle(data)
                    for feat, cl, dl in data:
                        result_feats.append(feat)
                        result_cl_labels.append(cl)
                        result_do_labels.append(dl)
                        sequence_stats.append(cl)

            else:
                # case 1(Default): create temporally correlated toy dataset by shuffling classes
                for chunk in idx_batch_cls:
                    cls_seq = list(range(num_class))
                    np.random.shuffle(cls_seq)
                    for cls in cls_seq:
                        idx = chunk[cls]
                        result_feats.extend([features[i] for i in idx])
                        result_cl_labels.extend([cl_labels[i] for i in idx])
                        result_do_labels.extend([do_labels[i] for i in idx])
                        sequence_stats.extend(list(np.repeat(cls, len(idx))))

            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        # TODO: manage num_target_train_set..
        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            self.target_train_set = (torch.stack(result_feats),
                                     result_cl_labels,
                                     torch.stack(result_do_labels))


    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.to(device)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):
        self.tensorboard.log_scalar(condition + '/loss_sum', loss_avg, epoch)
        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, loss_avg
        ))

        self.tensorboard.log_scalar(condition + '/loss_sum', loss_avg, epoch)



        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        self.tensorboard.log_scalar(condition + '/' + 'accuracy_class_' + suffix, class_accuracy, epoch)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))
        self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + suffix, cm_class,
                                              conf.args.opt['classes'], epoch)

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()

        class_loss_sum = 0.0

        total_iter = 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            if conf.args.log_percentile:
                assert conf.args.epoch == 1
                self.net.eval()

            for batch_idx, labeled_data in tqdm.tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)
                preds = self.net(feats)
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                if not conf.args.log_percentile:
                    self.optimizer.zero_grad()
                    class_loss.backward()
                    self.optimizer.step()
                else:
                    # print(len(preds))
                    for pred in preds:
                        self.conf_list.append(
                            float(torch.nn.functional.softmax(pred, dim=0).max().cpu()))

            if conf.args.log_percentile:
                print(self.conf_list)
                print(self.conf_list.sort())
                print(f"percentile of current list is : {np.percentile(self.conf_list, [5, 95])}")
        ######################## LOGGING #######################

        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def train_online(self, current_num_sample):

        """
        Train the model
        """
        raise NotImplementedError  # training Src with online is currently not enabled.

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

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models

        self.net.train()
        # self.feature_extractor.train()
        # self.class_classifier.train()

        class_loss_sum = 0.0
        total_iter = 0

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)
        print(len(feats))

        if len(feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()

        dataset = torch.utils.data.TensorDataset(feats, cls, dls)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)
        num_iter = len(data_loader)

        for e in range(conf.args.epoch):

            total_iter += num_iter

            for batch_idx, labeled_data in enumerate(data_loader):
                feats, cls, dls = labeled_data
                feats, cls = feats.to(device), cls.to(device)
                feature_of_labeled_data = self.feature_extractor(feats)
                # compute the class loss of feature_of_labeled_data
                class_loss, _, _ = self.get_loss_and_confusion_matrix(self.class_classifier,
                                                                          self.class_criterion,
                                                                          feature_of_labeled_data,
                                                                          cls)
                class_loss_sum += float(class_loss * feats.size(0))
                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        self.previous_train_loss = avg_loss

        return TRAINED

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')
        self.tensorboard.log_scalar(condition + '/' + name, value, epoch)

    def evaluation(self, epoch, condition):

        #########################################################################################################
        ##############################----- evaluation with target data -----####################################
        #########################################################################################################

        self.net.eval()

        y_true_sum = []
        y_pred_sum = None

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)
            y_true_sum = np.concatenate((y_true_sum, tgt_cls.cpu().numpy()), axis=0)
            if y_pred_sum is None:
                y_pred_sum = y_pred.cpu().numpy()
            else:
                y_pred_sum = np.concatenate((y_pred_sum, y_pred.cpu().numpy()), axis=0)


        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))
        self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + 'test', class_cm_test_data,
                                              conf.args.opt['classes'], epoch)

        self.previous_test_class_accuracy = class_accuracy
        self.previous_test_class_loss_of_test_data = class_loss_of_test_data

        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)
        # self.logger('f1', f1, epoch, condition)
        # self.logger('auroc', macro_roc_auc_ovo, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, condition, current_samples):
        #########################################################################################################
        ##############################----- evaluation with target data -----####################################
        #########################################################################################################
        # evaluation is done as list

        self.net.eval()

        # self.feature_extractor.eval()
        # self.class_classifier.eval()

        with torch.no_grad():

            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples


            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

            raw_y_pred = self.net(feats)
            y_pred = raw_y_pred

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(y_pred, cls)
            pred_label = y_pred.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(cls.cpu(), pred_label.cpu(), labels=labels)

            y_pred = y_pred.max(1, keepdim=False)[1]

            if '_vote' in conf.args.method:
                if 'TENT' not in conf.args.method:
                    raw_y_pred = self.batch_evaluation(feats)

                # for l2 prediction
                l2_printing_list = []
                tmp_featurizer_list = torch.nn.functional.softmax(self.featurizer(feats), dim=1)

                change_idx = []
                index_cls = 0
                pdist = torch.nn.PairwiseDistance(p=2)
                for single_featurized_feat in feats:
                    if len(self.bef_feat) != 0:
                        if(float(torch.sum(pdist(single_featurized_feat, self.bef_feat[0])))>3):
                            change_idx.append(index_cls)
                        self.l2_dist_list.append(float(torch.sum(pdist(single_featurized_feat, self.bef_feat[0]))))
                        self.bef_feat[0] = single_featurized_feat
                    else:
                        self.bef_feat.append(single_featurized_feat)
                    index_cls += 1

                bef_idx = 0
                change_idx.append(-1)

                for i in raw_y_pred:
                    self.entropy.append(float(self.Entropy(i.unsqueeze(0))))

                y_pred = raw_y_pred.argmax(-1)

                for idx in change_idx:
                    max_idx, max_2nd_idx = self.pred_counter(y_pred[bef_idx:idx])
                    if self.bef_gt_cls != y_pred[bef_idx]:
                        if self.bef_cls == max_idx and max_idx != max_2nd_idx:
                            max_idx = max_2nd_idx
                        else:
                            y_revised_pred = self.pred_revision(y_pred[bef_idx:idx], torch.argsort(raw_y_pred)[bef_idx:idx], max_idx)
                            max_idx, _ = self.pred_counter(y_revised_pred)

                    self.bef_cls = max_idx
                    self.bef_gt_cls = cls[bef_idx]
                    y_pred[bef_idx:idx] = torch.zeros_like(y_pred[bef_idx:idx]) + torch.tensor(
                        max_idx)
                    bef_idx = idx

                if 'gt' in conf.args.method:
                    gt, _ = self.pred_counter(cls)
                    y_pred = torch.zeros_like(y_pred) + torch.tensor(gt)

            elif conf.args.method == 'LAME':
                y_pred = self.batch_evaluation(feats).argmax(-1)

            elif conf.args.method == 'T3A':
                z = self.featurizer(feats)
                y_pred = self.batch_evaluation(z)

            try:
                true_cls_list = self.json['gt']
                pred_cls_list = self.json['pred']
                accuracy_list = self.json['accuracy']
                f1_macro_list = self.json['f1_macro']
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cl_labels]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

            # update self.json file
            self.json = {
                'gt': true_cls_list,
                'pred': pred_cls_list,
                'accuracy': accuracy_list,
                'f1_macro': f1_macro_list,
            }
            self.tensorboard.log_scalar(condition + '/online_accuracy', cumul_accuracy, epoch)

        return cumul_accuracy, class_loss_of_test_data, class_cm_test_data

    def dump_eval_online_result(self, is_train_offline=False):

        if is_train_offline:

            feats, cls, dls = self.target_train_set

            for num_sample in range(0, len(feats), conf.args.opt['batch_size']):
                current_sample = feats[num_sample:num_sample+conf.args.opt['batch_size']], cls[num_sample:num_sample+conf.args.opt['batch_size']], dls[num_sample:num_sample+conf.args.opt['batch_size']]
                self.evaluation_online(num_sample + conf.args.opt['batch_size'], '',
                                       [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """
        # self.draw_tsne(epoch)

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss

    def log_previous_test_result(self, epoch):

        self.logger('accuracy', self.previous_test_class_accuracy, epoch, 'test')
        self.logger('loss', self.previous_test_class_loss_of_test_data, epoch, 'test')
        self.logger('f1', self.previous_test_f1, epoch, 'test')
        # self.logger('auroc', self.previous_test_macro_roc_auc_ovo, epoch, 'test')

    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def pred_counter(self, pred_list):
        pred_counter = [0 for i in range(conf.args.opt['num_class'])]
        for pred_cls in pred_list:
            pred_counter[pred_cls] += 1
        pred_counter_idx = np.argsort(pred_counter)
        return pred_counter_idx[-1], pred_counter_idx[-2]

    def pred_revision(self, argfirst_list, rest_list, avoiding_class):
        new_list = []
        for i in range(len(argfirst_list)):
            if argfirst_list[i] == avoiding_class:
                idx = -2
                new_list.append(rest_list[i][idx])
            else:
                idx = -1
                new_list.append(rest_list[i][idx])
        return new_list