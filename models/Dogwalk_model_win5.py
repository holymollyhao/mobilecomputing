import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

if conf.args.dataset == 'dogwalk':
    opt = conf.DogwalkOpt
elif conf.args.dataset == 'dogwalk_win100':
    opt = conf.Dogwalk_WIN100_Opt
elif conf.args.dataset == 'dogwalk_all':
    opt = conf.DogwalkAllOpt
elif conf.args.dataset == 'dogwalk_all_win100':
    opt = conf.DogwalkAll_WIN100_Opt
elif conf.args.dataset == 'dogwalk_all_win5':
    opt = conf.DogwalkAll_WIN5_Opt
feature_flatten_dim = 32
input_channel_dim = opt['input_dim']

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            #win5
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
        )
        self.fc = nn.Linear(feature_flatten_dim, 5)

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)
        out = self.fc(out)

        return out

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        bn = nn.BatchNorm1d
        self.feature = nn.Sequential(

            # # win 50
            nn.Conv1d(input_channel_dim, 32, kernel_size=16, stride=2),
            nn.ReLU(True),
            bn(32),

            nn.Conv1d(32, 32, kernel_size=6),
            nn.ReLU(True),
            bn(32),

            nn.Conv1d(32, 64, kernel_size=6),
            nn.ReLU(True),
            bn(64),

            nn.Conv1d(64, 64, kernel_size=6),
            nn.ReLU(True),
            bn(64),
            nn.MaxPool1d(2),

            # win5
            # nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            # nn.ReLU(True),
            # bn(32),
            #
            # nn.Conv1d(32, 32, kernel_size=3),
            # nn.ReLU(True),
            # bn(32),
        )

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)

        return out


class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.DogwalkOpt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return out


class Domain_Classifier(nn.Module):

    def __init__(self):
        super(Domain_Classifier, self).__init__()

        self.domain_classifier = nn.Sequential(

            nn.Linear(feature_flatten_dim, 128),
            nn.Linear(128, 2))

    def forward(self, input, constant=0.1):
        input = GradReverse.grad_reverse(input, constant)
        out = self.domain_classifier(input)

        return F.log_softmax(out, 1)


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


# CDAN

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)




hidden_size = 128


if __name__ == '__main__':

    import torch
    import torchvision

    dummy_input_fe = torch.zeros((64, 3, 50))

    fe = Extractor()
    cc = Class_Classifier()

    feat = fe(dummy_input_fe)
    print(feat.shape)

    out = cc(feat)
    print(out.shape)
