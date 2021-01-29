from typing import Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class Binarize(Function):
    r"""Binarizer.

    Takes a Tensor and a threshold value as input.
    Sets positions where value >= threshold to 1,
    and the rest 0.
    """
    @staticmethod
    def forward(
        ctx,
        input_tensor: Tensor,
        threshold: float,
    ) -> Tensor:
        ctx.set_materialize_grads(False)
        # HACK: Do I really need to make it a float to pretend to need gradients?
        return torch.ge(input_tensor, threshold).to(torch.float16)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None]:
        # Gradient passes straight through
        return grad_output, None


class MaskedLinear(nn.Linear):
    r"""Subclass of torch.nn.Linear.

    The weights are masked according to a binary mask,
    which is obtained by binarizing a real-valued trainable mask.

    The binarizer is simply a thresholding function,
    where the threshold is a hyperparameter.

    # TODO: Short mathematical representation of layer from paper
    # TODO: Maybe remove the description redundant to Linear

    See https://www.aclweb.org/anthology/2020.emnlp-main.174.pdf for more details.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        threshold: Threshold for binarizer. Default: 0.0

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}
        mask:   the real-valued mask of shape
                :math:`(\text{out\_features}, \text{in\_features})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool=True,
        threshold: float=0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.threshold = threshold
        self.mask = Parameter(Tensor(out_features, in_features))
        # TODO: Init mask to adjust initial sparsity?
        nn.init.uniform_(self.mask, -1, 1)

    @classmethod
    def build_from_linear(
        cls,
        linear: nn.Linear,
        threshold: float=0.0,
    ):
        l = cls(linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                threshold=threshold)
        l.weight = linear.weight
        if linear.bias is not None:
            l.bias = linear.bias
        l.to(linear.weight.device)
        return l

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Overrides Linear.forward().
        Masks the weights before the matrix multiplication.
        """
        mask_bin = Binarize.apply(self.mask, self.threshold)
        return F.linear(input_tensor, mask_bin * self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, threshold={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.threshold
        )
