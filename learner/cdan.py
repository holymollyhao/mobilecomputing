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

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

DEBUG = True

class CDAN(DNN):
    def __init__(self, *args, **kwargs):
        super(CDAN, self).__init__(*args, **kwargs)

        # preprocess source data
        self.source_train_processing()

        # init model
        self.adversarial_net = self.model.AdversarialNetwork().to(self.device)

        # init criterions
        # no softmax -> cross-entropy loss; else, NLL loss
        # here, we don't have softmax layer when self.get_output
        self.class_criterion = nn.CrossEntropyLoss()
        self.adv_criterion = nn.BCELoss()

        # set hyperparameters
        parameter_list = [{'params': self.feature_extractor.parameters(), "lr_mult": 10, "decay_mult": 2},
                          {'params': self.class_classifier.parameters(), "lr_mult": 10, "decay_mult": 2},
                          {'params': self.adversarial_net.parameters(), "lr_mult": 10, "decay_mult": 2}]
        self.optimizer = optim.SGD(parameter_list,
                                   lr=conf.args.opt['learning_rate'],
                                   momentum=conf.args.opt['momentum'],
                                   weight_decay=conf.args.opt['weight_decay'],
                                   nesterov=True)
        # define random layer
        if conf.args.rand_proj:
            self.random_layer = self.model.RandomLayer().to(self.device)
        else:
            self.random_layer = None

        # set learning rate scheduler
        param_lr = []
        for param_group in self.optimizer.param_groups:
            param_lr.append(param_group["lr"])

        self.schedule_param = {"lr": conf.args.opt['learning_rate'], "gamma": 0.001, "power": 0.75}

        def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
            """Decay learning rate by a factor of 0.1 every lr_dec
            ay_epoch epochs."""
            lr = lr * (1 + gamma * iter_num) ** (-power)
            i = 0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_mult']
                param_group['weight_decay'] = weight_decay * param_group['decay_mult']
                i += 1

            return optimizer
        schedule_dict = {"inv": inv_lr_scheduler}

        self.lr_scheduler = schedule_dict["inv"]

        # for train() function
        self.labeled_src_data = None

    # override functions
    def freeze_layers(self):
        # freeze the FC layers in CDAN (necessary?)
        print('CDAN: I am child')
        for param in self.class_classifier.parameters():
            param.requires_grad = False

    def source_train_processing(self):
        features = [x[0] for x in self.source_dataloader['train'].dataset]
        cl_labels = [x[1] for x in self.source_dataloader['train'].dataset]
        do_labels = [x[2] for x in self.source_dataloader['train'].dataset]

        # reducing the total number of training set?

        self.source_train_set = (torch.stack(features),
                                 torch.stack(cl_labels),
                                 torch.stack(do_labels))

    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        state = {}
        state['epoch'] = epoch
        state['epoch_acc'] = epoch_acc
        state['best_acc'] = best_acc
        state['feature_extractor'] = self.feature_extractor.state_dict()
        state['class_classifier'] = self.class_classifier.state_dict()
        state['adversarial_net'] = self.adversarial_net.state_dict()
        state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, checkpoint_path)

        return

    def load_checkpoint(self, checkpoint_path, is_transfer=False):
        path = checkpoint_path
        checkpoint = torch.load(path, map_location=f'cuda:{conf.args.gpu_idx}')

        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.class_classifier.load_state_dict(checkpoint['class_classifier'])
        if 'adversarial_net' in checkpoint:
            self.adversarial_net.load_state_dict(checkpoint['adversarial_net'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Freeze classifier in CDAN
        for param in self.class_classifier.parameters():
            param.requires_grad = False

        return checkpoint

    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def grl_hook(self, coeff):
        def fun1(grad):
            return -coeff*grad.clone()
        return fun1

    def get_domain_loss(self, input_list, ad_net, entropy=None, coeff=None, random_layer=None):

        softmax_output = input_list[1].detach()
        feature = input_list[0]
        if softmax_output.size(0) % 2 == 1: #avoid odd number of inputs
            softmax_output = softmax_output[:-1]
            feature = feature[:-1]
        if random_layer is not None:
            random_out = random_layer.forward([feature, softmax_output])
            ad_out = ad_net(random_out.view(-1, random_out.size(1)))
        else:
            op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
            ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        batch_size = softmax_output.size(0) // 2
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(self.device)
        if entropy is not None:
            entropy.register_hook(self.grl_hook(coeff))
            entropy = entropy if len(entropy) %2 == 0 else entropy[:-1] #avoid odd number of inputs
            entropy = 1.0 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[feature.size(0) // 2:] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[0:feature.size(0) // 2] = 0
            target_weight = entropy * target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(
                weight).detach().item()
        else:
            return self.adv_criterion(ad_out, dc_target)

    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

    def train(self, epoch):
        """
        Train the model
        """

        # setup models
        self.feature_extractor.train()
        self.class_classifier.train()
        self.adversarial_net.train()

        src_feats, src_cls, src_dls = self.source_train_set
        tgt_feats, tgt_cls, tgt_dls = self.target_train_set

        if len(tgt_feats) == 1: # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()
            self.adversarial_net.eval()

        src_dataset = torch.utils.data.TensorDataset(src_feats, src_cls, src_dls)
        src_data_loader = DataLoader(src_dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=True, pin_memory=False)

        tgt_dataset = torch.utils.data.TensorDataset(tgt_feats, tgt_cls, tgt_dls)
        tgt_data_loader = DataLoader(tgt_dataset, batch_size = conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=True, pin_memory=False)

        transfer_loss_sum = 0.0
        class_loss_sum = 0.0
        total_iter = 0

        num_iter = len(tgt_data_loader)
        total_iter += num_iter

        if epoch == 1:
            self.labeled_src_data = iter(src_data_loader)

        for batch_idx, labeled_tgt_data in enumerate(tgt_data_loader):
            # get iteration index
            i = (epoch - 1) * num_iter + batch_idx

            # lr scheduler
            if conf.args.cdan_lr_schedule:
                self.optimizer = self.lr_scheduler(self.optimizer, i, **self.schedule_param)

            # get data and label
            try:
                src_inputs, src_cls, _ = self.labeled_src_data.next()
            except StopIteration:
                self.labeled_src_data = iter(src_data_loader)
                src_inputs, src_cls, _ = self.labeled_src_data.next()
            src_inputs, src_cls = src_inputs.to(device), src_cls.to(device)
            tgt_inputs, tgt_cls, _ = labeled_tgt_data
            tgt_inputs, tgt_cls = tgt_inputs.to(device), tgt_cls.to(device)

            # compute the feature and output
            features_source = self.net[0](src_inputs)
            logits_source =  self.net[1](features_source)
            features_target = self.net[0](tgt_inputs)
            logits_target = self.net[1](features_target)

            # concatenate
            features = torch.cat((features_source, features_target), dim=0)
            logits = torch.cat((logits_source, logits_target), dim=0)
            softmax_outs = nn.Softmax(dim=1)(logits)

            # CDAN+E
            entropy = self.Entropy(softmax_outs)
            transfer_loss = self.get_domain_loss([features, softmax_outs], self.adversarial_net, entropy,
                                                 self.calc_coeff(i, max_iter=num_iter*conf.args.epoch), self.random_layer)
            transfer_loss_sum += float(transfer_loss * src_inputs.size(0))
            class_loss = self.class_criterion(logits_source, src_cls)
            class_loss_sum += float(class_loss * src_inputs.size(0))
            total_loss = conf.args.loss_tradeoff * transfer_loss + class_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        avg_loss = (class_loss_sum + transfer_loss_sum) / total_iter
        self.log_loss_results('train', epoch=epoch, loss_avg = avg_loss)
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
        self.adversarial_net.train()

        transfer_loss_sum = 0.0
        class_loss_sum = 0.0
        total_iter = 0

        tgt_feats, tgt_cls, tgt_dls = self.mem.get_memory()
        tgt_feats, tgt_cls, tgt_dls = torch.stack(tgt_feats), torch.stack(tgt_cls), torch.stack(tgt_dls)

        if len(tgt_feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()
            self.adversarial_net.eval()

        src_dataset = torch.utils.data.TensorDataset(src_feats, src_cls, src_dls)
        src_data_loader = DataLoader(src_dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=False, pin_memory=False)

        tgt_dataset = torch.utils.data.TensorDataset(tgt_feats, tgt_cls, tgt_dls)
        tgt_data_loader = DataLoader(tgt_dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=False, pin_memory=False)

        num_iter = len(tgt_data_loader)
        labeled_src_data = iter(src_data_loader)

        for e in range(conf.args.epoch):
            total_iter += num_iter

            for batch_idx, labeled_tgt_data in enumerate(tgt_data_loader):
                # get iteration index
                i = e * num_iter + batch_idx

                # lr scheduler
                if conf.args.cdan_lr_schedule:
                    self.optimizer = self.lr_scheduler(self.optimizer, i, **self.schedule_param)

                # get data and label
                try:
                    src_inputs, src_cls, _ = labeled_src_data.next()
                except StopIteration:
                    labeled_src_data = iter(src_data_loader)
                    src_inputs, src_cls, _ = labeled_src_data.next()
                src_inputs, src_cls = src_inputs.to(device), src_cls.to(device)
                tgt_inputs, tgt_cls, _ = labeled_tgt_data
                tgt_inputs, tgt_cls = tgt_inputs.to(device), tgt_cls.to(device)

                if len(src_inputs) == 1 or len(tgt_inputs) == 1: # avoid BN error
                    continue
                if len(src_inputs) == 1 or len(tgt_inputs) == 1: # avoid BN error
                    continue
                # compute the feature and output
                features_source = self.net[0](src_inputs)
                logits_source = self.net[1](features_source)
                features_target = self.net[0](tgt_inputs)
                logits_target = self.net[1](features_target)

                # concatenate
                features = torch.cat((features_source, features_target), dim=0)
                logits = torch.cat((logits_source, logits_target), dim=0)
                softmax_outs = nn.Softmax(dim=1)(logits)

                # CDAN+E
                entropy = self.Entropy(softmax_outs)
                transfer_loss = self.get_domain_loss([features, softmax_outs], self.adversarial_net, entropy,
                                                     self.calc_coeff(i, max_iter=num_iter*conf.args.epoch), self.random_layer)
                transfer_loss_sum += float(transfer_loss * tgt_inputs.size(0))
                class_loss = self.class_criterion(logits_source, src_cls)
                class_loss_sum += float(class_loss * src_inputs.size(0))
                total_loss = conf.args.loss_tradeoff * transfer_loss + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        avg_loss = (class_loss_sum + transfer_loss_sum) / total_iter
        self.log_loss_results('train', epoch=current_num_sample, loss_avg=avg_loss)
        self.previous_train_loss = avg_loss

        return TRAINED




