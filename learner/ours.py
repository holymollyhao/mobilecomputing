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

from utils import memory

from utils.utils import *
from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
from utils.iabn import *

class Ours(TT_WHOLE):

    def __init__(self, *args, **kwargs):
        super(Ours, self).__init__(*args, **kwargs)

        # turn on grad for BN params only
        if conf.args.update_all:
            pass # update all layers
        else:
            for param in self.net.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if conf.args.iabn:
                if isinstance(module, InstanceAwareBatchNorm2d) or isinstance(module, InstanceAwareBatchNorm1d):
                    for param in module.parameters():
                        param.requires_grad = True



        # target = 'alpha'
        # params = list(map(lambda x: x[1], list(filter(lambda kv: target in kv[0] , self.net.named_parameters()))))
        # base_params = list(
        #     map(lambda x: x[1], list(filter(lambda kv: target not in kv[0], self.net.named_parameters()))))
        # self.optimizer = optim.Adam([{'params': base_params}, {'params': params, 'lr': conf.args.opt['learning_rate']*1000}], lr=conf.args.opt['learning_rate'],
        #                             weight_decay=conf.args.opt['weight_decay'])

        # self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
        #                             weight_decay=conf.args.opt['weight_decay'])

        self.real_mem = memory.CBRS(capacity=conf.args.memory_size)
        self.fifo = memory.FIFO(capacity=conf.args.update_every_x) # required for evaluation


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

        with torch.no_grad():

            self.net.eval()
            if conf.args.memory_type in ['Diversity']:
                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)
                logit = self.net(f.unsqueeze(0))
                self.mem.add_instance_with_logit(current_sample, logit=logit)

            elif conf.args.memory_type in ['FIFO', 'Reservoir']:
                self.mem.add_instance(current_sample)

            elif conf.args.memory_type in ['CBFIFO', 'CBReservoir']:
                # '''
                #### CBRS with pseudo label

                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)

                if conf.args.dataset in ['kitti_mot',
                                         'kitti_mot_test']:
                    outputs = self.net(f.unsqueeze(0))
                    outputs = non_max_suppression(outputs, conf_thres=conf.args.opt['conf_thres'],
                                                  iou_thres=conf.args.opt['nms_thres'])

                    pseudo_bincount = np.bincount(outputs[0][:, 5])
                    if len(pseudo_bincount) == 0:
                        pseudo_cls = -1
                    else:
                        if len(np.argwhere(np.amax(pseudo_bincount) == pseudo_bincount).flatten().tolist()) > 1: #multiple max
                            pseudo_cls = random.choice(np.argwhere(np.amax(pseudo_bincount) == pseudo_bincount).flatten().tolist())
                        else:
                            pseudo_cls = np.argmax(pseudo_bincount)


                    true_bincount = np.bincount(c[:, 1].detach().cpu())
                    if len(true_bincount) == 0 :
                        c= -1
                    else:
                        if len(np.argwhere(np.amax(true_bincount) == true_bincount).flatten().tolist()) > 1: #multiple max
                            c = random.choice(np.argwhere(np.amax(true_bincount) == true_bincount).flatten().tolist())
                        else:
                            c = np.argmax(true_bincount)
                else:
                    logit = self.net(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0]

                if pseudo_cls != -1:
                    self.mem.add_instance([f, pseudo_cls, d, c, 0])
                # '''

                #### Original
                if c != -1:
                    self.real_mem.add_instance((f,c,d))
                # self.mem.add_instance(current_sample)

        self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models

        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)

        entropy_loss = HLoss(temp_factor=conf.args.temperature)
        # kd_loss = KDLoss()
        diversity_loss = DiversityLoss(temp_factor=conf.args.temperature)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                eloss=dloss=0

                preds_of_data = self.net(feats)


                if conf.args.no_optim:
                    pass # no optimization
                else:
                    if isinstance(preds_of_data, list): #kitti
                        for i in range(len(preds_of_data)):
                            eloss += entropy_loss(preds_of_data[i].view(-1, 5+conf.args.opt['num_class'])[:, 5:])
                            dloss += diversity_loss(preds_of_data[i].view(-1, 5+conf.args.opt['num_class'])[:, 5:])
                    else:
                        eloss = entropy_loss(preds_of_data)
                        dloss = diversity_loss(preds_of_data)
                    print(f'Eloss:{eloss:.5f}, DLoss:{dloss:.5f}')
                    loss = eloss + conf.args.loss_scaler*dloss


                    ### Self-training with pseudo label
                    '''
                    feature_of_test_data = self.feature_extractor(feats)
                    preds_of_data = self.class_classifier(feature_of_test_data)
                    pseudo_cls = preds_of_data.max(1, keepdim=False)[1]
                    loss = self.class_criterion(preds_of_data, pseudo_cls)
                    '''

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        if conf.args.memory_type in ['CBFIFO', 'CBReservoir']:
            print(f'Pseudo dist:',end='\t')
            self.mem.print_class_dist()
            print(f'Real dist:',end='\t')
            self.mem.print_real_class_dist()
            print(f'Oracle dist:', end='\t')
            self.real_mem.print_class_dist()
        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        '''
        loss_list = []
        ####### Update loss:
        with torch.no_grad():
            self.feature_extractor.eval()
            self.class_classifier.eval()
            feats, cls, dls = self.mem.get_memory()
            feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)
            for feat in feats:
                preds_of_data = self.net(feat.unsqueeze(0))
                eloss = entropy_loss(preds_of_data)
                dloss = diversity_loss(preds_of_data)
                loss = eloss + conf.args.loss_scaler * dloss
                loss_list.append(loss)


        self.mem.update_loss(loss_list)
        '''

        return TRAINED
