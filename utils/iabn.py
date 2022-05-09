import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import conf
import copy
import statistics

def convert_iabn(module, **kwargs):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = InstanceAwareBatchNorm2d if isinstance(module, nn.BatchNorm2d) else InstanceAwareBatchNorm1d
        module_output = IABN(
            num_channels=module.num_features,
            k=conf.args.iabn_k,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )

        '''
        if module.affine:
            with torch.no_grad():
                module_output._bn.weight.data.copy_(module.weight)
                module_output._bn.bias.data.copy_(module.bias)
        module_output._bn.running_mean.data.copy_(module.running_mean)
        module_output._bn.running_var.data.copy_(module.running_var)
        module_output._bn.num_batches_tracked.data.copy_(module.num_batches_tracked)
        '''
        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(child, **kwargs)
        )
    del module
    return module_output


class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k=k
        self.affine = affine
        self._bn = nn.BatchNorm2d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

        # self.alpha = nn.Parameter(torch.Tensor(num_channels, 1, 1))
        # self.alpha.data.fill_(1.0)

        self.log_dic={}
        self.log_dic['b'] = []
        self.log_dic['c'] = []
        self.log_dic['h'] = []
        self.log_dic['w'] = []
        self.log_dic['mu'] = []
        self.log_dic['sigma2'] = []
        self.log_dic['mu_b'] = []
        self.log_dic['sigma2_b'] = []
        self.log_dic['s_mu'] = []
        self.log_dic['s_sigma2'] = []
        self.log_dic['mu_adj'] = []
        self.log_dic['sigma2_adj'] = []
    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        if True in torch.isnan(x):
            print('NAN')
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) #IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)


        # if sigma2_b.mean() < conf.args.sigma2_b_thres:
        #     if conf.args.use_in:
        #         mu_adj = mu
        #         sigma2_adj = sigma2
        #     else:
        #         mu_adj = mu_b
        #         sigma2_adj = sigma2_b
        # else:
        # s^2 / s^2_B ~= N(1, np.sqrt(2 / (n - 1)))^2) for large n
        # '''
        if h*w <=conf.args.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w)) #sigma2_b, mu_b에서 h*w개의 sample을 뽑았을 때의 mu, 즉 IN의 mean distribution (based on seen samples)
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))#sigma2_b, mu_b에서 h*w개의 sample을 뽑았을 때의 sigma, 즉 IN의 sigma distribution (based on seen samples)

            # mu_delta = self._softshrink(mu - mu_b, self.k * s_mu).detach()
            # mu_adj = mu_b + (sigma2_b > conf.args.sigma2_b_thres) * mu_delta
            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)#.detach() # mu가 mu_b보다 얼마나 떨어져있는지를 self.k * s_mu로 추정, softshrink로 보정

            # sigma2_delta = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2).detach()
            # sigma2_adj = sigma2_b + (sigma2_b > conf.args.sigma2_b_thres) * sigma2_delta
            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)#.detach()  # sigma2가 sigma2_b보다 얼마나 떨어져있는지를 self.k * s_sigma2로 추정, softshrink로 보정

            # if conf.args.use_in:
            #     mu_adj[sigma2_b < conf.args.sigma2_b_thres] = mu[sigma2_b < conf.args.sigma2_b_thres]
            #     sigma2_adj[sigma2_b < conf.args.sigma2_b_thres] = sigma2[sigma2_b < conf.args.sigma2_b_thres]

            sigma2_adj = F.relu(sigma2_adj) #non negative
        # '''

        '''
        alpha = self.alpha.clamp(0,1).view(1, -1, 1, 1)
        mu_adj = alpha * mu_b + (1 - alpha) * mu
        sigma2_adj = alpha * sigma2_b + (1 - alpha) * sigma2
        sigma2_adj = F.relu(sigma2_adj)  # non negative
        '''

        # self.log_dic['b'].append(b)
        # self.log_dic['c'].append(c)
        # self.log_dic['h'].append(h)
        # self.log_dic['w'].append(w)
        # self.log_dic['mu'].append(float(mu.mean()))
        # self.log_dic['sigma2'].append(float(sigma2.mean()))
        # self.log_dic['mu_b'].append(float(mu_b.mean()))
        # self.log_dic['sigma2_b'].append(float(sigma2_b.mean()))
        # self.log_dic['s_mu'].append(float(s_mu.mean()))
        # self.log_dic['s_sigma2'].append(float(s_sigma2.mean()))
        # self.log_dic['mu_adj'].append(float(mu_adj.mean()))
        # self.log_dic['sigma2_adj'].append(float(sigma2_adj.mean()))

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n

    def print_stats(self):
        print(self.alpha.mean())
        # print(self._bn.weight.mean())
        '''
        for key in self.log_dic:
            if len(self.log_dic[key])>2:
                mean = statistics.mean(self.log_dic[key])
                stdev = statistics.stdev(self.log_dic[key])
                max_val = max(self.log_dic[key])
                min_val = min(self.log_dic[key])
                # print(f'{key}\t{mean}\t{stdev}\t{max_val}\t{min_val}')
                print(f'{mean}', end='\t')
            else:
                print(f'', end='\t')
        print()
        '''

class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = affine
        self._bn = nn.BatchNorm1d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)
        # self.alpha = nn.Parameter(torch.Tensor(num_channels, 1))
        # self.alpha.data.fill_(1.0)

        self.log_dic={}
        self.log_dic['b'] = []
        self.log_dic['c'] = []
        self.log_dic['l'] = []
        self.log_dic['mu'] = []
        self.log_dic['sigma2'] = []
        self.log_dic['mu_b'] = []
        self.log_dic['sigma2_b'] = []
        self.log_dic['s_mu'] = []
        self.log_dic['s_sigma2'] = []
        self.log_dic['mu_adj'] = []
        self.log_dic['sigma2_adj'] = []
    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, l = x.size()

        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        # s^2 / s^2_B ~= N(1, np.sqrt(2 / (n - 1)))^2) for large n

        # '''
        if l <=conf.args.skip_thres:
            #     or sigma2_b.mean() < conf.args.sigma2_b_thres:
            # if conf.args.use_in:
            #     mu_adj = mu
            #     sigma2_adj = sigma2
            # else:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l) ##
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            # mu_delta = self._softshrink(mu - mu_b, self.k * s_mu).detach()
            # mu_adj = mu_b + (sigma2_b > conf.args.sigma2_b_thres) * mu_delta
            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)#.detach()
            # sigma2_delta = self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2).detach()
            # sigma2_adj = sigma2_b + (sigma2_b > conf.args.sigma2_b_thres) * sigma2_delta
            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)#.detach()
            sigma2_adj = F.relu(sigma2_adj)

            # if conf.args.use_in:
            #     mu_adj[sigma2_b < conf.args.sigma2_b_thres] = mu[sigma2_b < conf.args.sigma2_b_thres]
            #     sigma2_adj[sigma2_b < conf.args.sigma2_b_thres] = sigma2[sigma2_b < conf.args.sigma2_b_thres]
        # '''

        '''
        if l <=1:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            alpha = self.alpha.clamp(0,1).view(1, -1, 1)
            mu_adj = alpha * mu_b + (1 - alpha) * mu
            sigma2_adj = alpha * sigma2_b + (1 - alpha) * sigma2
            sigma2_adj = F.relu(sigma2_adj)  # non negative
        '''
        # self.log_dic['b'].append(b)
        # self.log_dic['c'].append(c)
        # self.log_dic['l'].append(l)
        # self.log_dic['mu'].append(float(mu.mean()))
        # self.log_dic['sigma2'].append(float(sigma2.mean()))
        # self.log_dic['mu_b'].append(float(mu_b.mean()))
        # self.log_dic['sigma2_b'].append(float(sigma2_b.mean()))
        # if l!=1:
        #     self.log_dic['s_mu'].append(float(s_mu.mean()))
        #     self.log_dic['s_sigma2'].append(float(s_sigma2.mean()))
        # self.log_dic['mu_adj'].append(float(mu_adj.mean()))
        # self.log_dic['sigma2_adj'].append(float(sigma2_adj.mean()))


        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n

    def print_stats(self):
        print(self.alpha.mean())
        # print(self._bn.weight.mean())
        '''
        for key in self.log_dic:
            if len(self.log_dic[key])>2:
                mean = statistics.mean(self.log_dic[key])
                stdev = statistics.stdev(self.log_dic[key])
                max_val = max(self.log_dic[key])
                min_val = min(self.log_dic[key])
                # print(f'{key}\t{mean}\t{stdev}\t{max_val}\t{min_val}')
                print(f'{mean}', end='\t')
            else:
                print(f'', end='\t')
        print()
        '''