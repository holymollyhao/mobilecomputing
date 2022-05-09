import torch
import torch.nn as nn
import torch.nn.functional as F


import conf
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 13824
input_channel_dim = 9


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),


            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),


            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),


            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Conv1d(256, 512, kernel_size=3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )

    def forward(self, input):
        out = self.feature(input)

        out = out.view(input.size(0), -1)
        return out


class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),


            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Linear(256, conf.args.opt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return out

if __name__ == '__main__':
    pass