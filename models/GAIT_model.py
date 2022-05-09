import torch
import torch.nn as nn
import torch.nn.functional as F

import conf
'''
class Object(object):
    pass
conf.args= Object()
conf.args.da = "00000000000"
conf.args.dataset = "pacs"
conf.args.gpu_idx = 0
conf.args.opt={}
conf.args.opt['num_class']=conf.GaitOpt['num_class']
'''


device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
feature_flatten_dim = 256 #128 len or 64???
input_channel_dim = 3
class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()


        self.feature = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 96, kernel_size=6, stride=2),
            nn.BatchNorm1d(96),
            nn.ReLU(True),

            nn.Conv1d(96, 128, kernel_size=6, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Conv1d(256, 256, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
        )

    def forward(self, input):
        out = self.feature(input)

        out = out.view(input.size(0), -1)
        return out


class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()


        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.args.opt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return out

if __name__ == '__main__':

    fe = Extractor()
    cc = Class_Classifier()
    print(sum(p.numel() for p in fe.parameters() if p.requires_grad)+sum(p.numel() for p in cc.parameters() if p.requires_grad))

    feat = fe(torch.zeros((10,3,192)))
    print(feat.shape)
    out = cc(feat)
    print(out.shape)
    pass