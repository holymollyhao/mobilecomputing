import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .RTBatchNorm import RTBatchNorm1d
from .DomainSpecificBN import DomainSpecificBatchNorm1d
import conf


'''
class Object(object):
    pass
conf.args = Object()
conf.args.da = "00000000000"
conf.args.dataset = "pacs"
conf.args.gpu_idx = 0
conf.args.opt = {}
conf.args.opt['num_class'] = conf.ExtraSensoryOpt['num_class']
'''

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 32 # 32:win5, 128:win10
input_channel_dim = 31

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
        self.fc = nn.Linear(feature_flatten_dim, conf.ExtraSensoryOpt['num_class'])

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)
        out = self.fc(out)

        return out


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()

        if conf.args.method in ['TT_SINGLE_STATS', 'TT_SINGLE_STATS_ADAPTIVE', 'TENT_STATS']:
            bn = RTBatchNorm1d
        else:
            bn = nn.BatchNorm1d
        self.feature = nn.Sequential(

            # from featmatching
            # nn.Conv1d(input_channel_dim, 16, kernel_size=3, stride=1),
            # nn.LeakyReLU(0.3),
            # bn(16),
            #
            # nn.Conv1d(16, 16, kernel_size=3, stride=1),
            # nn.LeakyReLU(0.3),
            # bn(16),
            #
            # nn.Conv1d(16, 32, kernel_size=5, stride=4),
            # nn.LeakyReLU(0.3),
            # bn(32),
            #
            # nn.Conv1d(32, 32, kernel_size=3, stride=1),
            # nn.LeakyReLU(0.3),
            # bn(32),
            #
            # nn.Conv1d(32, 64, kernel_size=5, stride=4),
            # nn.LeakyReLU(0.3),
            # bn(64),
            #
            # nn.Conv1d(64, 100, kernel_size=5, stride=1),
            # nn.LeakyReLU(0.3),
            # bn(100),
            # nn.AvgPool1d(4),

            #win10
            # nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            # nn.ReLU(True),
            # bn(32),
            #
            # nn.Conv1d(32, 32, kernel_size=3),
            # nn.ReLU(True),
            #
            # bn(32),
            # nn.Conv1d(32, 64, kernel_size=3),
            # nn.ReLU(True),
            # bn(64),
            #
            # nn.Conv1d(64, 64, kernel_size=3),
            # nn.ReLU(True),
            # bn(64),



            #win5
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True),
            bn(32),

            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(True),
            bn(32),


        )

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)

        return out

class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.ExtraSensoryOpt['num_class']))

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
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
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

class Extractor_DSBN(nn.Module):

    def __init__(self, num_domains):
        super(Extractor_DSBN, self).__init__()

        self.l1 = nn.Sequential(
            #win5
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True))
        self.bn1 = DomainSpecificBatchNorm1d(32, num_domains)

        self.l2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(True))
        self.bn2 = DomainSpecificBatchNorm1d(32, num_domains)

    def forward(self, input, domain):
        out = self.l1(input)
        out = self.bn1(out, domain)
        out = self.l2(out)
        out = self.bn2(out, domain)

        out = out.view(input.size(0), -1)
        return out


class Class_Classifier_DSBN(nn.Module):

    def __init__(self, num_domains):
        super(Class_Classifier_DSBN, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.ExtraSensoryOpt['num_class']))

    def forward(self, input, domain):
        out = self.class_classifier(input)

        return out

class RandomLayer(nn.Module):
    def __init__(self):
        super(RandomLayer, self).__init__()
        self.input_dim_list = [feature_flatten_dim, conf.ExtraSensoryOpt["num_class"]]
        self.input_num = len(input_dim_list)
        self.output_dim = 1024
        self.random_matrix = [torch.randn(self.input_dim_list[i], self.output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class AdversarialNetwork(nn.Module):
    def __init__(self, lr_mult=10, decay_mult=2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(feature_flatten_dim * conf.ExtraSensoryOpt["num_class"], hidden_size)
        # self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.decay_mult = decay_mult
        self.lr_mult = lr_mult

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        # x = self.ad_layer2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":self.lr_mult, 'decay_mult':self.decay_mult}]

if __name__ == '__main__':
    # fe = Extractor()
    # cc = Class_Classifier()
    # print(sum(p.numel() for p in fe.parameters() if p.requires_grad)+sum(p.numel() for p in cc.parameters() if p.requires_grad))
    #
    # feat = fe(torch.zeros((10,6,256)))
    # print(feat.shape)
    # out = cc(feat)
    # print(out.shape)
    # pass

    import torch
    import torchvision

    dummy_input_fe = torch.zeros((64, 31, 5))

    fe = Extractor()
    cc = Class_Classifier()

    feat = fe(dummy_input_fe)
    print(feat.shape)

    out = cc(feat)
    print(out.shape)
