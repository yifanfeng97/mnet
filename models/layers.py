import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SparseMM(torch.autograd.Function):

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_output = None
        if self.needs_input_grad[0]:
            grad_output = torch.mm(self.sparse.t(), grad_output)
        return grad_output


class MeshConvolution(Module):
    """
    Mesh convolution layer
    """

    def __init__(self, in_ft_num, out_ft_num, bias=True):
        super(MeshConvolution, self).__init__()
        self.in_ft_num = in_ft_num
        self.out_ft_num = out_ft_num
        self.weight = Parameter(torch.Tensor(in_ft_num, out_ft_num))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft_num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ( '\
                + str(self.in_ft_num) + ' -> '\
                + str(self.out_ft_num) + ')'
