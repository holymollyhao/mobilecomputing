import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
import conf
import numpy as np

'''
class Object(object):
    pass
conf.args= Object()
conf.args.da = "00000000000"
conf.args.dataset = "pacs"
conf.args.gpu_idx = 0
conf.args.opt={}
conf.args.opt['num_class']=conf.OpportunityOpt['num_class']
'''


device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
input_channel_dim = 6#conf.args.opt['input_dim']

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

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()


        self.feature = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=16, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(64),


            nn.Conv1d(64, 128, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(128),


            nn.Conv1d(128, 128, kernel_size=6),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=6),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            nn.BatchNorm1d(256),

            nn.MaxPool1d(2),


        )

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)
        return out

    def get_parameters(self):
        return [{"params": self.feature.parameters(), "lr_mult": 10, 'decay_mult': 2}]

net = Extractor()
out = net(torch.zeros((10, input_channel_dim, 256)))
feature_flatten_dim = out.size(1)#30720

class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()


        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.ICHAROpt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return out

    def forward_except_last(self, input):
        out = self.class_classifier[0](input)
        out = self.class_classifier[1](out)
        out = self.class_classifier[2](out)
        out = self.class_classifier[3](out)
        out = self.class_classifier[4](out)
        out = self.class_classifier[5](out)

        return out

    def get_parameters(self):
        return [{"params": self.class_classifier.parameters(), "lr_mult": 10, 'decay_mult': 2}]

# ichar_dataset
dfeature_flatten_dim = 2304


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

class AdversarialNetwork(nn.Module):
    def __init__(self, lr_mult=10, decay_mult=2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(feature_flatten_dim * conf.ICHAROpt["num_class"], hidden_size)
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
    # out = cc(fe(torch.zeros((10,6,512))))
    # print(out.shape)

    # net = Extractor_DA()
    # net(torch.zeros((10, 6, 512)))

    # gumbel_out = F.gumbel_softmax(torch.FloatTensor([-0.78062,-3.70236,-2.49411,3.56326,-1.50357,-1.86860,4.23477,0.66190,-0.55903,-2.56409,-1.30175]), hard=False)
    # print(gumbel_out)
    # gumbel_out = F.gumbel_softmax(torch.FloatTensor([-0.78062,-3.70236,-2.49411,3.56326,-1.50357,-1.86860,4.23477,0.66190,-0.55903,-2.56409,-1.30175]), tau=0.1, hard=False)
    # print(gumbel_out)
    # gumbel_out = F.gumbel_softmax(torch.FloatTensor([-0.78062,-3.70236,-2.49411,3.56326,-1.50357,-1.86860,4.23477,0.66190,-0.55903,-2.56409,-1.30175]), tau=0.01, hard=False)
    # print(gumbel_out)

    fe = Extractor()
    cc = Class_Classifier()
    print(sum(p.numel() for p in fe.parameters() if p.requires_grad)+sum(p.numel() for p in cc.parameters() if p.requires_grad))

    feat = fe(torch.zeros((10,6,256)))
    print(feat.shape)
    out = cc(feat)
    print(out.shape)
    pass