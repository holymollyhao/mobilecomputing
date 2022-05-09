import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import conf
'''
class Object(object):
    pass
conf.args= Object()
conf.args.da = "00000000000"
conf.args.dataset = "pacs"
conf.args.gpu_idx = 0
conf.args.opt={}
conf.args.opt['num_class']=conf.HHAROpt['num_class']
'''

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 256
input_channel_dim = 6


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
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.MaxPool1d(2),


        )


    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)

        return out


class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.HHAROpt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return out





class ALL(nn.Module):

    def __init__(self):
        super(ALL, self).__init__()

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
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=6, stride=2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, 256, kernel_size=6),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.MaxPool1d(2),
        )


        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, conf.args.opt['num_class']))

    def forward(self, input):
        out = self.feature(input)
        out = out.view(input.size(0), -1)
        out = self.class_classifier(out)

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

class AdversarialNetwork(nn.Module):
    def __init__(self, lr_mult=10, decay_mult=2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(feature_flatten_dim * conf.HHAROpt["num_class"], hidden_size)
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

    dummy_input_fe = torch.zeros((64,6,256))
    dummy_input_cc = torch.zeros((64, 256))

    fe = Extractor()
    cc = Class_Classifier()

    path = '/home/tsgong/git/DAPPER/log/hhar/Src/tgt_a.nexus4/211020_src_pretrain_model_s0/cp/cp_last.pth.tar'
    checkpoint = torch.load(path, map_location=f'cuda:{conf.args.gpu_idx}')

    fe.load_state_dict(checkpoint['feature_extractor'])
    cc.load_state_dict(checkpoint['class_classifier'])

    print(fe(dummy_input_fe).shape)
    print(cc(dummy_input_cc).shape)

    all = ALL()

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "fe_actual_input_1" ] + [ "fe_learned_%d" % i for i in range(56) ]
    output_names = [ "fe_output1" ]

    torch.onnx.export(fe, dummy_input_fe, "../hhar_fe.onnx", verbose=True, input_names=input_names, output_names=output_names, do_constant_folding=True)

    input_names = [ "cc_actual_input_1" ] + [ "cc_learned_%d" % i for i in range(2) ]
    output_names = [ "cc_output1" ]

    torch.onnx.export(cc, dummy_input_cc, "../hhar_cc.onnx", verbose=True, input_names=input_names, output_names = output_names, do_constant_folding = True)


    torch.onnx.export(all, dummy_input_fe, "../hhar_all.onnx", verbose=True, input_names=input_names, output_names = output_names, do_constant_folding = True)
