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
import models.DSA_model
from .dnn import DNN
from .tt_whole import TT_WHOLE
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


# class SupportSet():
#     def __init__(self, num_class, init_weights):
#
#         self.sup_set = [
#             [init_weights[i]]
#             for i in range(len(init_weights))
#         ]
#
#         # self.mem_size_list setting
#         self.mem_size_list = [int(conf.args.memory_size) // num_class for i in range(num_class)]
#         for i in range(int(conf.args.memory_size) % num_class):
#             self.mem_size_list[i] += 1
#
#         self.num_class = num_class
#
#     def append(self, data):
#         assert len(data) == 3, 'data must be consisted of 3 elements'
#
#         # data must be a list of [feature_tensor, classfier_output, entropy]
#         feature_tensor, classfier_output, entropy = tuple(data)
#         pseudo_label = torch.argmax(classfier_output, axis=-1)
#
#         # append, then remove until it fits the memory requirement
#         self.sup_set[int(pseudo_label)].append(data)
#         while len(self.sup_set[int(pseudo_label)]) > self.mem_size_list[int(pseudo_label)]:
#             self.remove(int(pseudo_label))
#
#     def remove(self, index):
#         self.sup_set[index] = sorted(self.sup_set[index], key=lambda data: float(data[2]))
#         self.sup_set[index].pop()
#
#     def __getitem__(self, item):
#         return self.sup_set[item]
#
#     def support(self):
#         sup_list = []
#         for class_index in range(self.num_class):
#             for iter_index in range(len(self.sup_set[class_index])):
#                 sup_list.append(self.sup_set[class_index][iter_index][0])
#         return torch.stack(sup_list)
#
#     def label(self):
#         label_list = []
#         for class_index in range(self.num_class):
#             for iter_index in range(len(self.sup_set[class_index])):
#                 elem = self.sup_set[class_index][iter_index][1]
#                 label_list.append(
#                     elem
#                 )
#         return torch.stack(label_list)


class T3A(DNN):
    def __init__(self, *args, **kwargs):
        super(T3A, self).__init__(*args, **kwargs)

        init_weight = []
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                init_weight.append(module.weight.data)
                self.classifier = torch.nn.Sequential(module)
        assert (len(init_weight) == 1)
        self.featurizer = torch.nn.Sequential(*(list(self.net.children())[:-1]))

        warmup_supports = init_weight[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1),
                                                         num_classes=conf.args.opt['num_class']).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.is_mem_enough = bool(int(conf.args.memory_size) // conf.args.opt['num_class'])

        if self.is_mem_enough:
            # self.filter_K_list = [int(conf.args.memory_size) // conf.args.opt['num_class'] for i in
            #                       range(conf.args.opt['num_class'])]
            self.filter_K_list = [20 for i in range(conf.args.opt['num_class'])]
        else:
            self.filter_K_list = [1 for i in range(conf.args.opt['num_class'])]

        self.num_classes = conf.args.opt['num_class']
        self.softmax = torch.nn.Softmax(-1)



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

        # Get sample from target
        xs, cls, dls = self.target_train_set
        current_sample = xs[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]

        # Add sample to memory
        self.mem.add_instance(current_sample)

        if not conf.args.adapt_then_eval:
            self.evaluation_online(current_num_sample, '',
                                   [[current_sample[0]], [current_sample[1]], [current_sample[2]]])



        # Skipping depend on "batch size"
        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models
        self.net.eval()

        if len(xs) == 1:  # avoid BN error
            self.net.eval()

        xs, cls, dls = self.mem.get_memory()
        xs, cls, dls = torch.stack(xs), torch.stack(cls), torch.stack(dls)

        dataset = torch.utils.data.TensorDataset(xs, cls, dls)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        with torch.no_grad():
            for batch_idx, labeled_data in enumerate(data_loader):
                xs, cls, _ = labeled_data
                xs = xs.to(device)
                _ = cls.to(device)

                feats = self.featurizer(xs)
                # preds_of_data = self.classifier(feature_of_test_data.view(len(feats),-1)) # (batch_size, feat_dim)

                for index in range(len(feats)):
                    # single_feature
                    # feature_tensor = torch.unsqueeze(feature_of_test_data[index], dim=0).to(device)
                    # classifier_output_tensor = preds_of_data[index]
                    # yhat = torch.nn.functional.one_hot(classifier_output_tensor.unsqueeze(0).argmax(1),
                    #                                    num_classes=conf.args.opt['num_class']).float()
                    # entropy_tensor = - (classifier_output_tensor.unsqueeze(0).softmax(1) * classifier_output_tensor.unsqueeze(0).log_softmax(1)).sum(1)
                    #
                    # # data that needs to append
                    # new_data = [feature_tensor.squeeze(), yhat.squeeze(), entropy_tensor.squeeze()]
                    #
                    # self.support_set.append(new_data)

                    # z = self.featurizer(xs[index].unsqueeze(0)).T.squeeze().unsqueeze(0)

                    if conf.args.model == 'wideresnet28-10':  # rest operations not in the model.modules()
                        z = self.featurizer(xs[index].unsqueeze(0))
                        z = F.avg_pool2d(z, 8)
                        z = z.view(-1, 640)
                    elif conf.args.model == 'resnet18':
                        z = self.featurizer(xs[index].unsqueeze(0))
                        z = F.avg_pool2d(z, 4)
                        z = z.view(z.size(0), -1)
                    else:
                        z = self.featurizer(xs[index].unsqueeze(0)).squeeze().unsqueeze(0)

                    p = self.classifier(z.unsqueeze(0)).squeeze().unsqueeze(0)

                    # pseudo label in one-hot vector
                    yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()

                    # softmax output of p, entorpy
                    ent = softmax_entropy(p)

                    self.supports = self.supports.to(z.device)
                    self.labels = self.labels.to(z.device)
                    self.ent = self.ent.to(z.device)
                    self.supports = torch.cat([self.supports, z])
                    self.labels = torch.cat([self.labels, yhat])
                    self.ent = torch.cat([self.ent, ent])
                    assert(len(self.ent) == len(self.supports) and len(self.supports) == len(self.labels))

        if conf.args.adapt_then_eval:
            print(f'conf.args.adapt_then_eval is true')
            self.evaluation_online(current_num_sample, '', self.mem.get_memory())

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    # according to https://github.com/matsuolab/T3A, adapt_algorithm.py
    def single_evaluation(self, z):
        # by choosing btw ret 1 : my implementation , ret2 : exact copy of T3A source code, can choose which to use in evaluation
        # supports, labels = self.support_set.support().squeeze(), self.support_set.label().squeeze()
        # supports = torch.nn.functional.normalize(supports, dim=1)
        # weights = supports.T @ labels
        # ret1 = z.squeeze() @ torch.nn.functional.normalize(weights, dim=0) # z is 2048,1,1

        if conf.args.model == 'wideresnet28-10':  # additional operation required
            z = F.avg_pool2d(z, 8)
            z = z.view(-1, 640)
        elif conf.args.model == 'resnet18':
            z = F.avg_pool2d(z, 4)
            z = z.view(z.size(0), -1)

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        ret2 = z.T @ torch.nn.functional.normalize(weights, dim=0)
        return ret2.squeeze()

    def batch_evaluation(self, extracted_feat):
        y_pred = []
        for index in range(len(extracted_feat)):
            y_pred.append(torch.argmax(self.single_evaluation(extracted_feat[index])))
        return torch.stack(y_pred)  # should be tensor list

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K_list = self.filter_K_list

        indices = []

        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K_list[i]])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        if not self.is_mem_enough:
            indices = []
            ent_s2 = self.ent
            indices1 = torch.LongTensor(list(range(len(ent_s2))))
            _, indices2 = torch.sort(ent_s2)
            indices.append(indices1[indices2[:conf.args.memory_size]])
            indices = torch.cat(indices)

            self.supports = self.supports[indices]
            self.labels = self.labels[indices]
            self.ent = self.ent[indices]

        return self.supports, self.labels

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)