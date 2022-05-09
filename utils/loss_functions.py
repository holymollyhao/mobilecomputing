import torch
from torch import nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
# but there is a bug in the original code: it sums up the entropy over a batch. so I take mean instead of sum
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x/self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax+1e-6)
        b = entropy.mean()

        # b = F.softmax(x/self.temp_factor, dim=1) * F.log_softmax(x/self.temp_factor, dim=1)
        # b = -1.0 * b.mean()
        return b


# https://github.com/alinlab/cs-kd/blob/master/train.py
class KDLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        q = torch.softmax(target / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q) * (self.temp_factor ** 2) / input.size(0)
        return loss

class DiversityLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(DiversityLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        soft_prob = F.softmax(x/self.temp_factor, dim =1)
        pb_pred_tgt = soft_prob.sum(dim=0)  # [9]
        pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt  # [9] sums to 1.   # normalizatoin to a prob. dist.
        target_div_loss = -torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))
        return target_div_loss


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, x, target_prob):
        soft_prob = F.softmax(x, dim =1)
        pb_pred_input = soft_prob.sum(dim=0)  # [9]
        pb_pred_input = 1.0 / pb_pred_input.sum() * pb_pred_input  # [9] sums to 1.   # normalizatoin to a prob. dist.
        prob1 = pb_pred_input
        # prob2=  F.softmax(target_prob, dim=1)
        prob2= target_prob

        m = 0.5 * (prob1 + prob2)
        loss = 0.0
        loss += F.kl_div(torch.log(prob1), m, reduction="batchmean")
        loss += F.kl_div(torch.log(prob2), m, reduction="batchmean")

        return (0.5 * loss)
