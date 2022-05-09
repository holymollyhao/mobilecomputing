import conf
from .tt_whole import TT_WHOLE
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class TENT(TT_WHOLE):
    def __init__(self, *args, **kwargs):
        super(TENT, self).__init__(*args, **kwargs)

        # turn on grad for BN params only


        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

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


        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED


        # Evaluate with a batch
        self.evaluation_online(current_num_sample, '', self.mem.get_memory())



        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            # self.feature_extractor.eval()
            # self.class_classifier.eval()
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)



        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        entropy_loss = HLoss()
        # kd_loss = KDLoss()

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)

                if conf.args.dsbn:
                    assert (dls.tolist().count(int(dls[0])) == len(dls)), 'exists different domains!'
                    assert (dls[0] == 0), 'finally sth different! - inside train_online'
                    feature_of_labeled_data = self.feature_extractor(feats, self.num_src_domains)
                    preds_of_data = self.class_classifier(feature_of_labeled_data, self.num_src_domains)
                else:
                    if conf.args.dataset in ['kitti_mot', 'kitti_mot_test']: #TODO: batch size 20, requires 3 grad steps, is it efficient?
                        preds_of_data = self.net(feats)
                    else:

                        preds_of_data = self.net(feats)

                if isinstance(preds_of_data, list): #kitti
                    loss = 0
                    for i in range(len(preds_of_data)):
                        loss += entropy_loss(preds_of_data[i].view(-1, 5+conf.args.opt['num_class'])[:, 5:]) # loss for classes starts from 5




                else:
                    loss = entropy_loss(preds_of_data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED