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


class SHOT(DNN):
    # def __init__(self, model, tensorboard, source_dataloader, target_dataloader, write_path):

    def freeze_layers(self):
        print('I am child')
        # freeze the FC layers in SHOT
        for param in self.class_classifier.parameters():
            param.requires_grad = False

    def load_checkpoint(self, checkpoint_path, is_transfer=False):
        path = checkpoint_path
        checkpoint = torch.load(path, map_location=f'cuda:{conf.args.gpu_idx}')

        # Freeze classifier in SHOT
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.class_classifier.load_state_dict(checkpoint['class_classifier'])
        for param in self.class_classifier.parameters():
            param.requires_grad = False

        return checkpoint

    def pseudo_label(self, target_dataset, epoch):
        with torch.no_grad():
            tgt_inputs, tgt_cls, tgt_dls = target_dataset
            tgt_inputs, tgt_cls = tgt_inputs.to(device), tgt_cls.to(device)
            tgt_features = self.net[0](tgt_inputs)
            tgt_logits = self.net[1](tgt_features)
            tgt_outputs = nn.Softmax(dim=1)(tgt_logits)

        # measure performance (for observation purpose)
        _, tgt_predicts = torch.max(tgt_outputs, 1)
        before_acc = torch.sum(torch.squeeze(tgt_predicts).float() == tgt_cls).item() / float(tgt_cls.size(0))

        # pre-processing for calculating cosine distance
        _tgt_features = torch.cat((tgt_features.cpu(), torch.ones(tgt_features.size(0), 1)), 1)
        _tgt_features = (_tgt_features.t() / torch.norm(_tgt_features, p=2, dim=1)).t()

        # attain the centroid for each class in the target domain (C0)
        _tgt_features = _tgt_features.float().cpu().numpy()
        K = tgt_outputs.size(1)  # number of classes
        _tgt_outputs = tgt_outputs.float().cpu().numpy()
        C0 = _tgt_outputs.transpose().dot(_tgt_features) / (conf.args.epsilon + _tgt_outputs.sum(axis=0)[:, None])
        cls_count = np.eye(K)[tgt_predicts.cpu()].sum(axis=0)
        labelset = np.where(cls_count > conf.args.threshold)
        labelset = labelset[0]

        # obtain the pseudo labels via the nearest centroid classifier
        cos_dist = cdist(_tgt_features, C0[labelset], 'cosine')
        pred_labels = cos_dist.argmin(axis=1)
        pred_labels = labelset[pred_labels]

        # update the target centroids based on the new pseudo labels and update the label
        # updating once gives sufficiently good pseudo labels
        for round in range(1):
            aff = np.eye(K)[pred_labels]
            C = aff.transpose().dot(_tgt_features) / (conf.args.epsilon + aff.sum(axis=0)[:, None])
            cos_dist = cdist(_tgt_features, C[labelset], 'cosine')  # cdist requires at least two samples
            pred_labels = cos_dist.argmin(axis=1)
            pred_labels = labelset[pred_labels]

        # measure performance (for observation purpose)
        after_acc = np.sum(pred_labels == tgt_cls.cpu().float().numpy()) / len(_tgt_features)
        log_str = 'Accuracy = {:.4f}% -> {:.4f}%'.format(before_acc * 100, after_acc * 100)
        print(log_str)
        self.logger("without_pseudo_labels_accuracy", before_acc, epoch, "train")
        self.logger("pseudo_labels_accuracy", after_acc, epoch, "train")

        return pred_labels.astype('int')

    def Entropy(self, input_):
        bs = input_.size(0)
        entropy = -input_ * torch.log(input_ + conf.args.epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.feature_extractor.train()
        self.class_classifier.train()

        feats, cls, dls = self.target_train_set

        if len(feats) == 1:  # avoid BN error
            self.feature_extractor.eval()
            self.class_classifier.eval()

        dataset = torch.utils.data.TensorDataset(feats, cls, dls)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        class_loss_sum = 0.0
        total_iter = 0

        num_iter = len(data_loader)
        total_iter += num_iter

        for batch_idx, labeled_data in enumerate(data_loader):
            feats, cls, _ = labeled_data
            feats, cls = feats.to(device), cls.to(device)

            # pseudo-label algorithm
            if conf.args.cls_par > 0:
                self.feature_extractor.eval()
                self.class_classifier.eval()

                # get pseudo labels by self-supervised learning approach
                pseudo_labels = self.pseudo_label((feats, cls, dls), epoch)
                pseudo_labels = torch.from_numpy(pseudo_labels).to(device)

                self.feature_extractor.train()
                self.class_classifier.train()

            # compute the feature
            logits_target = self.feature_extractor(feats)
            outputs_target = nn.Softmax(dim=1)(logits_target)

            # compute loss (1) : cross entropy with pseudo labels
            if conf.args.cls_par > 0:
                class_loss = conf.args.cls_par * nn.CrossEntropyLoss()(logits_target, pseudo_labels)
            else:
                class_loss = torch.tensor(0.0).to(device)

            # compute loss (2) : L_ent
            L_ent = torch.mean(self.Entropy(outputs_target))

            # compute loss (3) : L_div
            msoftmax = outputs_target.mean(dim=0)
            L_div = torch.sum(msoftmax * torch.log(msoftmax + conf.args.epsilon))

            class_loss += (L_ent + L_div)

            # update model
            class_loss_sum += float(class_loss * feats.size(0))

            self.optimizer.zero_grad()
            class_loss.backward()
            self.optimizer.step()

        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def train_online(self, current_num_sample):
        """
        Train the model
        """

        # setup models

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


        class_loss_sum = 0.0
        total_iter = 0

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), torch.stack(cls), torch.stack(dls)

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
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)

                if conf.args.cls_par > 0:
                    self.feature_extractor.eval()
                    self.class_classifier.eval()

                    # get pseudo labels by self-supervised learning approach
                    pseudo_labels = self.pseudo_label((feats, cls, dls), current_num_sample)
                    pseudo_labels = torch.from_numpy(pseudo_labels).to(device)

                    self.feature_extractor.train()
                    self.class_classifier.train()

                if len(feats) == 1: # avoid BN error
                    continue
                # compute the feature
                logits_target = self.feature_extractor(feats)
                outputs_target = nn.Softmax(dim=1)(logits_target)

                # compute loss (1) : cross entropy with pseudo labels
                if conf.args.cls_par > 0:
                    class_loss = conf.args.cls_par * nn.CrossEntropyLoss()(logits_target, pseudo_labels)
                else:
                    class_loss = torch.tensor(0.0).to(device)

                # compute loss (2) : L_ent
                L_ent = torch.mean(self.Entropy(outputs_target))

                # compute loss (3) : L_div
                msoftmax = outputs_target.mean(dim=0)
                L_div = torch.sum(msoftmax * torch.log(msoftmax + conf.args.epsilon))

                class_loss += (L_ent + L_div)

                # update model
                class_loss_sum += float(class_loss * feats.size(0))
                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        self.previous_train_loss = avg_loss

        return TRAINED
