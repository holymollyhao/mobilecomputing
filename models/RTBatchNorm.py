"""
https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
"""

import torch
import torch.nn as nn
import conf
def compare_bn(bn1, bn2):
    err = False
    if not torch.allclose(bn1.running_mean, bn2.running_mean):
        print('Diff in running_mean: {} vs {}'.format(
            bn1.running_mean, bn2.running_mean))
        err = True

    if not torch.allclose(bn1.running_var, bn2.running_var):
        print('Diff in running_var: {} vs {}'.format(
            bn1.running_var, bn2.running_var))
        err = True

    if bn1.affine and bn2.affine:
        if not torch.allclose(bn1.weight, bn2.weight):
            print('Diff in weight: {} vs {}'.format(
                bn1.weight, bn2.weight))
            err = True

        if not torch.allclose(bn1.bias, bn2.bias):
            print('Diff in bias: {} vs {}'.format(
                bn1.bias, bn2.bias))
            err = True

    if not err:
        print('All parameters are equal!')


class RTBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(RTBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0
        # online_adapt_momentum = conf.args.bn_momentum
        online_adapt_momentum = conf.args.bn_momentum  # 0.01

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:

            if input.dim() == 3:
                mean = input.mean([0, 2])
                # use biased var in train
                var = input.var([0, 2], unbiased=False)

            elif input.dim() == 2:
                mean = input.mean([0])
                # use biased var in train
                var = input.var([0], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            '''
            if input.dim() == 3:
                mean = input.mean([0, 2])
            elif input.dim() == 2:
                mean = input.mean([0])
            # print(mean)
            # print(input.shape,end='\t')
            f = open('/home/tsgong/git/WWW/dist.txt', 'a')
            f.write(f'{str(input.shape)}\t')
            [f.write(f'{float(x):.5f}\t') for x in mean]
            f.write(f'\n')
            f.close()

            mean = self.running_mean
            var = self.running_var
            '''
            # '''
            if input.size(0) == 1:  # test-time; one-by-one
                if input.dim() == 3:
                    mean = input.mean([0, 2])
                elif input.dim() == 2:
                    mean = input.mean([0])
                # use biased var in train
                prev_mean = self.running_mean
                prev_var = self.running_var

                with torch.no_grad():
                    self.running_mean = (1 - online_adapt_momentum) * prev_mean + online_adapt_momentum * mean
                    # update running_var with unbiased var
                    self.running_var = (1 - online_adapt_momentum) * (
                            prev_var + online_adapt_momentum * ((mean - prev_mean) ** 2))

                # one-by-one test, set mean and var as new running mean and var
                mean = self.running_mean
                var = self.running_var
            else:
                raise NotImplementedError  # we do not want to get batch test data
            # '''

        if input.dim() == 3:
            input = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
            if self.affine:
                input = input * self.weight[None, :, None] + self.bias[None, :, None]
        elif input.dim() == 2:
            input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
            if self.affine:
                input = input * self.weight[None, :] + self.bias[None, :]

        return input


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


if __name__ == '__main__':

    '''
    ##### TEST MyBatchNorm2d #####
    # Init BatchNorm layers
    my_bn = MyBatchNorm2d(3, affine=True)
    bn = nn.BatchNorm2d(3, affine=True)

    compare_bn(my_bn, bn)  # weight and bias should be different
    # Load weight and bias
    my_bn.load_state_dict(bn.state_dict())
    compare_bn(my_bn, bn)

    # Run train
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())

    # Run eval
    my_bn.eval()
    bn.eval()
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
    '''

    ##### TEST RTBatchNorm1d #####
    # Init BatchNorm layers
    my_bn = RTBatchNorm1d(3, affine=True)
    bn = nn.BatchNorm1d(3, affine=True)

    compare_bn(my_bn, bn)  # weight and bias should be different
    # Load weight and bias
    my_bn.load_state_dict(bn.state_dict())
    compare_bn(my_bn, bn)

    # Run train
    # for _ in range(10):
    #     scale = torch.randint(1, 10, (1,)).float()
    #     bias = torch.randint(-10, 10, (1,)).float()
    #     x = torch.randn(10, 3, 100) * scale + bias
    #     out1 = my_bn(x)
    #     out2 = bn(x)
    #     compare_bn(my_bn, bn)
    #
    #     torch.allclose(out1, out2)
    #     print('Max diff: ', (out1 - out2).abs().max())
    #
    # # Run eval
    # my_bn.eval()
    # bn.eval()
    # for _ in range(10):
    #     scale = torch.randint(1, 10, (1,)).float()
    #     bias = torch.randint(-10, 10, (1,)).float()
    #     x = torch.randn(10, 3, 100) * scale + bias
    #     out1 = my_bn(x)
    #     out2 = bn(x)
    #     compare_bn(my_bn, bn)
    #
    #     torch.allclose(out1, out2)
    #     print('Max diff: ', (out1 - out2).abs().max())

    # Run eval (single input)
    my_bn.train()
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(1, 3, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
