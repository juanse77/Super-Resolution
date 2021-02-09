��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
SuperResolution
qNNtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)RqX   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)Rq(X   conv1q(h ctorch.nn.modules.conv
Conv2d
qXN   C:\Anaconda3\.conda\envs'\torch_env\lib\site-packages\torch\nn\modules\conv.pyqX�  class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)
qtqQ)�q}q(hhhh	)Rq (X   weightq!ctorch._utils
_rebuild_parameter
q"ctorch._utils
_rebuild_tensor_v2
q#((X   storageq$ctorch
FloatStorage
q%X   1891175108992q&X   cuda:0q'K�Ntq(QK (KKKKtq)(KK	KKtq*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   1890311620224q2X   cuda:0q3KNtq4QK K�q5K�q6�h	)Rq7tq8Rq9�h	)Rq:�q;Rq<uhh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBhh	)RqCX   trainingqD�X   in_channelsqEKX   out_channelsqFKX   kernel_sizeqGKK�qHX   strideqIKK�qJX   paddingqKKK�qLX   dilationqMKK�qNX
   transposedqO�X   output_paddingqPK K �qQX   groupsqRKX   padding_modeqSX   zerosqTubX   conv2qUh)�qV}qW(hhhh	)RqX(h!h"h#((h$h%X   1890311619264qYX   cuda:0qZM�Ntq[QK (KKKKtq\(K6K	KKtq]�h	)Rq^tq_Rq`�h	)Rqa�qbRqch1h"h#((h$h%X   1890311616960qdX   cuda:0qeKNtqfQK K�qgK�qh�h	)RqitqjRqk�h	)Rql�qmRqnuhh	)Rqohh	)Rqphh	)Rqqhh	)Rqrhh	)Rqshh	)Rqthh	)RquhD�hEKhFKhGKK�qvhIKK�qwhKKK�qxhMKK�qyhO�hPK K �qzhRKhShTubX   upsampleq{(h ctorch.nn.modules.pixelshuffle
PixelShuffle
q|XV   C:\Anaconda3\.conda\envs'\torch_env\lib\site-packages\torch\nn\modules\pixelshuffle.pyq}X)  class PixelShuffle(Module):
    r"""Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(N, L, H_{in}, W_{in})` where :math:`L=C \times \text{upscale\_factor}^2`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} \times \text{upscale\_factor}`
          and :math:`W_{out} = W_{in} \times \text{upscale\_factor}`

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    __constants__ = ['upscale_factor']

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
q~tqQ)�q�}q�(hhhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hD�X   upscale_factorq�KubuhD�ub.�]q (X   1890311616960qX   1890311619264qX   1890311620224qX   1891175108992qe.       똾���������Ft�
���!���J�[w���Y�wΫ���׾�����      R[����=�M����<�Ž�=)35���$�>��<G�D�8q�������+���+�a�'�`�y֟��h-�"=�8�="��=��*>+?<��6�-�;O
�=̂�=� �>��>���=T��=4�>�{@>�o#>�@P>+�>5OI��6۽���<]��3�3�&���H�pr�4����X�d�8���q�ژ)�ђ���q�;�nF�ܽ�L��m��d��=�/��i��8/p�; ����Ȋ�1ś�{e^��;\{g�v�м����$K7��߷��ϥ��߻��=K��;W��&��=�И=d�u>��&=Q�p=��<}#>�q_>�/@>k�>��>c,]>���=�ɋ=(� >�սϲ̻fv�jLJ��!:�h+��i���m�!�O��(&<�C����;���߽���[�M���N�V�6���J��=���R=9D�ߟ$�Im��MR=&�|<.q��K(�V����r��C�0�w�%���>�D1Ի�z��y�!��(<(�R<�v>�a=��=��j=Z�S=�P>���<+Bb>���>u:V>�â>w��>��>A�/>hE5>g_�=�ڽj+=�X�.��<��G�#�~�e]$�}�+�	�,<.��=ERx=9�%��5��Q��Y�8�Ž!��X��]KK��Ɗ=� ���=�}����>��0J�=|<�<���<�W��[�m��F���d1�A����T��᎙<�h>B�R;�`��I3>�sf>���;�I>��9>��*>���=�ou>�\8>q\�=k;>�>��>A�>�fq>��ڽ�U\����SϽ8�A�O��]N�㜸�Nh�6�D�C.|��O����K,����;����W�+���y��.���<�D��T�(�=��<�  �Ul���M� �<���;��6��_��ͽ�]�P�[����
>i�g>�(�=fj�=�>t>�a>;V3��>�=��T=�>P�q>�1k>�^o>�L�=dT=��{>�|�=Q�X���T=��C��/���e�r?9�?��6�o.������s�=�1w�@�9<��o=!�;��=���������b&���q<WR��(�
<� �"]b=M�޽0#{�lB���a�ٝ�'Eǽĭ�F��Ġ���k��t;>.>�f>o�g>#N�<�-n=�!{>�g=J3>��m>�P�=|� >s4B>
=�>͈>�?#>p��=��%<)�~�.�:r'�̹ �u��=c �??E��8� '��S[ʼ�/�="Q
=]&?��²=c+ν\L潷��=�֔�iHܼdx�t��ĺ����2޻�mq=����z�/mǻ<���
K���[��7������D��VH���0��1�D>b�`>(�&> �>s�.=�
g<�S>�(o>ۚV=��|>�>�=ė�>y�>Yc=�e�=8g�>g�R>S'(>�.<A�����F�L�Z<�x!=#Ø�W�����I���v�佩�=L2�=�c�=��=0&�=���J�=����?:v���t�i��;���������:̽n2r�D��=���<Y�E�F�<��s[�\���ǖ�cνD��P���}>��X>5w>��= 2|>V�m=%�=e��<p��=BcA>DE4>�5>��<b�;>ڽ�>ʙ�=�*�=�1�>�Ԁ��+#�����*���)=��P���;�˽\���b�={�����6�1g3��{���O=���=<ڝ=�`�=j~M���3=���O�0�`���<Y��<�Z�����*,�5/<z�<[c�=Z;��v���qF�� /��&���R?=��v>J�W>T:f=X&>EO>,`n=��^=��>�c�>'��=8>4R�>��:��O>�V_=6Ј=�Ӵ=7��{�d���G�����zj� �C�*����]F�˜;�O�>��=��*>��>׽=;5�=��� �����<rr,��>��3���">��-R;|��<����.?���]=ZH��S����N�%�F���?�{�="z�;<)�ADf>�"%=�Z�=TP9>�H=�y|>\C�=��>�nH>��>^u>�a>��@>C�=��c><u\>�s�={�<�d�<�N���G��!B��+A�GW���0�k?�A�)�A��<�j ��&׼�%|< �<f��=ܢ>��+���=��\���{��H�n���r�UԱ�K�W���A=�7�|�6�bke�	��v�A���QG��U�<a$����C�;8!>��`=�/>�=Y>��h>��?=�s >;6>�>N�M>���=���=��;=3g>�!J>�v�<~#'>a�=OBR��]	��~���ꝼ.�ֽ�ݯ��|#�{X5�=!I�4䒽\F)��:�=��">j�=9{==.L��t^��:>"C���~�<����kY�j�3�@T�K3<g�E�� )�ن��A<A�z5���k<q鍽��yJ�S�Q�U�?�*�=�*�>��q>i_">�o�=�>1>~
�>ʶ>=V �=�6�<��=w
g>�^>�L�;<��>���<i��=� >SN�:�<�=Y�	��Z�����+����L��O�$�?>+P`;���=�2>�qY=�3ԼX,�=\7>]��=       �~>?̜%���o��
?�5*��       ��/<Z�#����=N+L�V�û�S�q��=%-�=D�]�z�iM��O�:Pל����=(�̽d�Z�r2��n�ս諺��|>�����Kŧ��敾a����������;��o=}͆�EQ���#ڽZ�<?>� �J�D���vi"��*¼�<ɽo��|v�Z3:���4�\�F���+�O�=����S��sS=Q�"<|1�=Z�=~��<-��: >5��=Y"��	��:_֗=���+� ?��ŉ;�>�=s�>��= >�6��b^H�{R��N��=!s�=}�[�`>օm�r�#>+"�=O�\>��}>�*�=`�b>I�-=�#d>�n>)~C�j+\>�ש=�����J>�a>�g>���o����@��=\O�=ɨ+=?)>t=>�M����=Ž��[�������?ü�z����z�j�=� ==~�.��=pۄ��a]���w=� �#����D�P��=�!�=��;���>Q��A|�=^X����<�
���!����h=l�J?�3��=}ɷ<��w��D�q~:���w��9���=��5�����锽ul���󃾥4��cm7=��i��x"� ������=��=�=���<����̽a�4>�>E�>�[�>܇���|>�,>yl�=