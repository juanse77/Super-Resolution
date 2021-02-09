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
q%X   2798205300464q&X   cuda:0q'K�Ntq(QK (K	KKKtq)(KK	KKtq*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   2799342494832q2X   cuda:0q3K	Ntq4QK K	�q5K�q6�h	)Rq7tq8Rq9�h	)Rq:�q;Rq<uhh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBhh	)RqCX   trainingqD�X   in_channelsqEKX   out_channelsqFK	X   kernel_sizeqGKK�qHX   strideqIKK�qJX   paddingqKKK�qLX   dilationqMKK�qNX
   transposedqO�X   output_paddingqPK K �qQX   groupsqRKX   padding_modeqSX   zerosqTubX   conv2qUh)�qV}qW(hhhh	)RqX(h!h"h#((h$h%X   2799342494736qYX   cuda:0qZM�Ntq[QK (KK	KKtq\(KQK	KKtq]�h	)Rq^tq_Rq`�h	)Rqa�qbRqch1h"h#((h$h%X   2799342498672qdX   cuda:0qeKNtqfQK K�qgK�qh�h	)RqitqjRqk�h	)Rql�qmRqnuhh	)Rqohh	)Rqphh	)Rqqhh	)Rqrhh	)Rqshh	)Rqthh	)RquhD�hEK	hFKhGKK�qvhIKK�qwhKKK�qxhMKK�qyhO�hPK K �qzhRKhShTubX   conv3q{h)�q|}q}(hhhh	)Rq~(h!h"h#((h$h%X   2799342495120qX   cuda:0q�M�Ntq�QK (K	KKKtq�(K�K	KKtq��h	)Rq�tq�Rq��h	)Rq��q�Rq�h1h"h#((h$h%X   2799342498576q�X   cuda:0q�K	Ntq�QK K	�q�K�q��h	)Rq�tq�Rq��h	)Rq��q�Rq�uhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hD�hEKhFK	hGKK�q�hIKK�q�hKKK�q�hMKK�q�hO�hPK K �q�hRKhShTubX   conv4q�h)�q�}q�(hhhh	)Rq�(h!h"h#((h$h%X   2799342498384q�X   cuda:0q�K�Ntq�QK (KK	KKtq�(KQK	KKtq��h	)Rq�tq�Rq��h	)Rq��q�Rq�h1h"h#((h$h%X   2799342493872q�X   cuda:0q�KNtq�QK K�q�K�q��h	)Rq�tq�Rq��h	)Rq��q�Rq�uhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hD�hEK	hFKhGKK�q�hIKK�q�hKKK�q�hMKK�q�hO�hPK K �q�hRKhShTubX   upsampleq�(h ctorch.nn.modules.upsampling
Upsample
q�XT   C:\Anaconda3\.conda\envs'\torch_env\lib\site-packages\torch\nn\modules\upsampling.pyq�X�  class Upsample(Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``

    Shape:
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, `bicubic`, and `trilinear`) don't proportionally
        align the output and input pixels, and thus the output values can depend
        on the input size. This was the default behavior for these modes up to
        version 0.3.1. Since then, the default behavior is
        ``align_corners = False``. See below for concrete examples on how this
        affects the outputs.

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='nearest')
        >>> m(input)
        tensor([[[[ 1.,  1.,  2.,  2.],
                  [ 1.,  1.,  2.,  2.],
                  [ 3.,  3.,  4.,  4.],
                  [ 3.,  3.,  4.,  4.]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> m(input)
        tensor([[[[ 1.0000,  1.2500,  1.7500,  2.0000],
                  [ 1.5000,  1.7500,  2.2500,  2.5000],
                  [ 2.5000,  2.7500,  3.2500,  3.5000],
                  [ 3.0000,  3.2500,  3.7500,  4.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> m(input)
        tensor([[[[ 1.0000,  1.3333,  1.6667,  2.0000],
                  [ 1.6667,  2.0000,  2.3333,  2.6667],
                  [ 2.3333,  2.6667,  3.0000,  3.3333],
                  [ 3.0000,  3.3333,  3.6667,  4.0000]]]])

        >>> # Try scaling the same data in a larger tensor
        >>>
        >>> input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
        >>> input_3x3[:, :, :2, :2].copy_(input)
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]]]])
        >>> input_3x3
        tensor([[[[ 1.,  2.,  0.],
                  [ 3.,  4.,  0.],
                  [ 0.,  0.,  0.]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> # Notice that values in top left corner are the same with the small input (except at boundary)
        >>> m(input_3x3)
        tensor([[[[ 1.0000,  1.2500,  1.7500,  1.5000,  0.5000,  0.0000],
                  [ 1.5000,  1.7500,  2.2500,  1.8750,  0.6250,  0.0000],
                  [ 2.5000,  2.7500,  3.2500,  2.6250,  0.8750,  0.0000],
                  [ 2.2500,  2.4375,  2.8125,  2.2500,  0.7500,  0.0000],
                  [ 0.7500,  0.8125,  0.9375,  0.7500,  0.2500,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> # Notice that values in top left corner are now changed
        >>> m(input_3x3)
        tensor([[[[ 1.0000,  1.4000,  1.8000,  1.6000,  0.8000,  0.0000],
                  [ 1.8000,  2.2000,  2.6000,  2.2400,  1.1200,  0.0000],
                  [ 2.6000,  3.0000,  3.4000,  2.8800,  1.4400,  0.0000],
                  [ 2.4000,  2.7200,  3.0400,  2.5600,  1.2800,  0.0000],
                  [ 1.2000,  1.3600,  1.5200,  1.2800,  0.6400,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info
q�tq�Q)�q�}q�(hhhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hD�X   nameq�X   Upsampleq�X   sizeq�NX   scale_factorq�G@       X   modeq�X   nearestq�X   align_cornersq�NubuhD�ub.�]q (X   2798205300464qX   2799342493872qX   2799342494736qX   2799342494832qX   2799342495120qX   2799342498384qX   2799342498576qX   2799342498672qe.�       W�\=
�=Ygɽ���`\����U=)#(�=������O9�;Á��>o繽+�>�w��� =���=�EZ����?1>�R�=LM�=��׽VB\>D>7�,�K3�=or�=aΆ=_-�ˢ�=3�����f���:��TD=(�H>�O%>����$��=5�����=��8=Ah�=ǇD�-�)Cü�:>��9<V�"�f�E=��ڽv=9��x�c>��3>HN��QH>Eq�=:5}>�>X�a>���ji�=F3>���:�o=i�H>ɻ=J�-��y�<7 >vC��&&2�Ǟ�Q�9���=N�=�i��u9ԽPZo�Y��= ��<5YF>�|>�H>��y=�7>|�=��0�&�=��ٽ�	2>.-�=5�<>�N3>�L�=�3=qAG�ˁ>^�>�8��3���|->\PȽ��,>A������d?׽������B<��R>���<O��=�+�o�ӽC�N���1>�����s=��>]t!=«��hX>�V�<=ZM���>|@>�E���LU>���<g�=�>��
k>Y��=2n%�o��=�$�M&ٽ�f�=�68��н���=�Y���Z>�
�<�u!����=�=�	ѽ�q7�5�O��*3=�NQ�b�>��b�F|�����������'=g����_c�f�G�V�Y��2v���,����=a�e���r��ˉ����b��=l�=U[�<D��<�=��+�$�!��ؽ�Sg�/�D���Q��>�=�<�W彉��Q|�&�k]p=��=���^�Z��p���:a<D���N�E[�<�I5��B�=j����=F[�9���| 0�p����G>1�`���z�Ly��U�=��F�.�A���[=x?4��I�[�2>��[=�MX�]L���Y���˼�>�39>���5�7>�X<9m���>�)
>��%>Y:�=y�>��=�r=w #=�]Ͻ��»�h��a��<�V<Η��       ���������      F��=�i�<.��<�=�����V<��ƽZ	�=��<�E��<ǽ�.6�/�=��ֽϒ��b�,�^.ƽw� =����=gi0����;��Z�~��c���yY�={�<�A�:2��<ͷԼ<+k��`�=�'޽���=4铽�.��Z,�=��=����� ���s��:�~��퇽�Y��C׋=�$���Ӵ�zZ�����<A۽ϋ<f��@��{%ٻ�V�;;��lC3=�+<Pڜ=؎=ѕ�=�-�t�����ؽn�<���/�<�S��[�Ҽ���y�;���<L�����=qA�<"ї�CyR=r��w����<���=��:�W��<�ª=C�=��=p,�=�*p=Q?�L��=����q�=�B�=Ͻӵ���=X�=����~��=j1������Cڽ�\���߽��<��=��M=��=7��<D����ȧ�#2�h�@�-P�=����1�%<S/W=��c<9�=U�<ؖE=W��=�m">�K>���=b�>����9��ȼ����s��㓽7�:���=��= 氽|�==�u�=�}������P�=�/R�Z��==N}<�,�1Y�8�s=}i����<�K�<�|�;3����=yo�=�0�;��g���p=fO�=�<�|�}�&=64� �ɼH>�p�<��:= ����u=�;y�;1�=�C�<*'=L�漕=�7���Z�����<t���H=7�f���'=Δy���=���U��ɶý��=?�ق<am�=V�>�7=�l��!�=�y���=>��=ڪ����>�)~�E����<tt��g�W!�4�;T��zb���<�	>j(����=�C=Z���2�l;��7�>ve3>���=����`9>��>%%A�)�>�>g�ƻ��<��=�Q>.�>���=j[�=vVu�L�X;���ý�m�<�)�=*k[�\F
>�L<<9���q%d�n4H=�z�:;�K=�*\=q�y�d��=~ �<Ӫ=�D�JB�����"x»i��?�ϽzȞ�Oʓ��S=LuQ;�z\�d�H=L�R:웽~�<�R�h
���F�=<;_=[xU=2<�=�����ս.ڹ<>n=��=�>n��Y�V/����=9��=}u�=����1����: Z�=�������ѽYJ8<Z_<D`���dr��̽`�=����9=��W���<�9��ky=��3�T�lxR��Y��q$��Ks=$��;/�=�!i��6��#��=2����bL��'������P�=��=���E��O�r�� �=!�j<ۺN=��k=�e'��2=�7ཻ��=T�����a�z=�?}�b�=U��< ��p�"=�'i���=䚚<H��=8�6=�9%=��g=�L]��1��#�<���=�B���$:=1���WR�=�R&<�(���uQ��@=d���A�׽� <��;&ђ��Y�y<ؽ��r=��N;�ӽ���^�e��<f�=+��=]⾽X�v��b�=6n���?���<�[����f�kg=<D�ļH�ͽU�	���f���в�VI�K�����=�<�7����1<<x���g0=@�;�H1=~�����
�Nϸ��i<G}#����z���^<5�s=�P�<^=$K�~�<Gw=�g�=!����=�6���N#�g0�=��m=�>�����=0K;���=��b=,-὾5���O=^�����j����Q<<a��:�!����=	�A��x����,��u:�=Ωս���^^�<oH���ζ=����C�?�>'h<>��=�����A=�����
X<���0�=Q`��;��=�M��GI������&=�b��z-�@�f=�n=�)��I��=��u=P�G��8�A�S�����j���=�%��b�߽ݔ����=��=��"��=�w=rG���x=��=��Լ��_���=u|\�=�<�Vh=�|�=>@<���<Jkb��9��ǽ��Ƚ$�������ǽ�����=}���e~�����3�<��h�b,���h=m >j|>P`�=�=f�=��>�c>=N�q=�l�=�D�=%�	=	Ֆ=ܰ=���=H�4��N�=ɓ�=۔<>㫻���=by����#�D�ֽA��=���� ����=�0���'�=�K�=�_�<N���2'`=��<Ÿ��Ad��X'�� �=��=�a���Mc=���<��=��<�(B��Y=d��<,-q��M�=^��=�U�=4�L=lE	����=!O4����L�3��#�;�
��5�:�$=�߄;�:��XӢ���=q#�=�������� �=jh�=I0O��홼=g�tѡ�u#>&�=1.l��Ȩ=.90>v��=�"�<@�>���	N�=��(==l�=��2>e(�=���=���=�ϒ;��6=��<1=6��3���U��`��<�Π<��=���o�G�_1�<�l[=E�<=pgֽʎ�=��X��&v='6���8�<�|�=��=�������<�LJ���@=��'=#�=�:�=�M�=�V�=��
�=gM�<����U������<�>?Ƽ�;�=,�=L��b�$��;r�Z���
�Y`�9t��,�t=�F�=��<���F���<��B�R��龼5�
�����م����4�5�j9=�ҽ���<��<�=`o>a��=y��=���;t�=��=G�=�쒽2 	��4=��>�6<N���6C�=XF=���sa=��=l��<�ۼ�<�&ǽ1ߔ��u�=Ǆ�<�6��ˎs=��V�0t�=`f1���=󚭽�;="zy�D{=���K�<�����۳=sG�=s3��������L\޼���=���!6K�:=0��<�
�=~k�<;+�={Eҽf�<���<ԥ�1k�`��=��^�����v�=�p="������4=W=�=���=n�W�۽�ǿ<��9<�ײ=Z�U<���3��=B��i�S=H�	��S����ӽ2N�=}7Z=�c�=:����	���q=��H=Wo��%�]=C��3V=��������KJ<��ֽ�5=���+�=;ϻ=��DF{��!׽t�=OW=��;E�����=�+<�D6��]�=�c��\#��	2t=�/��N���C|=�;\=q�2�f���{|=��A=���n����<=��׊q��=c�ǽ;��P���u�<�
O��෽&&?�c3�����=m4U�U�X;�<����{�ڽ8��;\��;Y��<�|=�OF=�&�;�cʽ�>׽�/����<�D�=���������=s%�<TGϽڰ۽����H���b[�%,�=��U�������q�]�=nԽ{g�:^�H>�d ���H=�E���;�=;��m�>��T���P<���=�,�<Ǥ�=��w=�ټq���<�<ڼ�r��*��=gx$>݂�=��=�w6�t+��SĽe�=8{�=�gZ=�?(>�P�=��<2mg=G�H=ǽ=�!>�JK=[�$�f>�"�� ��g>= 
����=�0=��=�_Q���=�t�=���d�=�yl�p�;�;��$�=�v= x}=��=��u=�D=��=�0����=�z��xfƽ����he��lt��]��ͱ�=̝�� ������=�K=U��X�G�OE�=L��=p����=���bU�����Ó��;j��gB�y�<�����E=�d���ҽ��ڼJ��=��2�H=��V�DA=��n=����'ֽ��<��%�R݇�?�=.	���p�=�+�z8^=� <�픽���<)p�Ė޼�fԼ!",��¯=�Q �F��=ߓz=�஽4æ���>�a��=�dm��6R<��=��ڽ��=�j���Q��$'˽�\��v�=��=�"=��'=�S;2�պ�W=�|�="[�=W�S�ú������=,��<�U��J<(�v���=�V|���=�fd;P*ƽt��;���2�=m"���|	=��=�D=�y8�ID�=���1"�=�
>�����=�����<�D����5� v&=�]�=u)T=7�.=�>�x=Aﴽd��=9�=w�ڼ	�=z�:<�k[��4^=�h <Bjf=W����B�=��=#\����=.�ᅱ���=�ɽ*[=B줽��ݽSG��RZ`=+N�����ח)=ƻ�=��=38��iɼ��L=�[���eA=&L\<Q"����ý"={p=ը�<&��=GT��ql��T�=���=
L���V�=s|�����=o>w� 
S�]3�<�W�=B$�=�V�=p�=�|y� X��l�=!�;:߹���"=6�l=�׽�	+=��=E����m=�h��}=���(���nS���<������h=���=�S=4ϴ�&���[:�����=󡌽�|���ס���i���F���<%�<�Y/=��=�P=�.V�!��<�/=��X�k�I�a	7��]���jj�����lP=��=dbp=�iý�H���b�= ��X�g��iֽF:g�X�=�5[��|���?=;|��#d�z���=e[���W����=����<6Җ=lھ��un=5��=�H"=q+����=آ����<Ϭ<�����=t��<���ڹ�=�Y�w��=�ݣ<4(%=�)��g�=��o�R��=�F�C�麶뮽c�Z��&�=r�r��Ž���? �C�&�^������=��нg�=�B��3��=nM���#�Ϻ��$=��H��B<��ɽ5 =+��=��=FZ�;"��=-#�=����n~�=��d=%b��dU�= �_=M0���wĽiH�t����<S>�<���]���Iֽjຽ�
2����;��ϽXҖ��+��$�=l��=���;�J<޳��7ݢ=�vr��Z�=G��=�Ş=S1r<L�=r�<���=�%��$�[�:�=�~ͽZ<�Kn< ��;0iR�����З�=��<+w�=q��<�5R=l�����+�%_<M��΅����;�w��q�=O�v��M�ocټYW�=�T=W�;� =��R�� ��;��kӎ=����R�����<��t��<r������=A�Q�Ā�<��<a*=�]��Sk�2�=֟�=Fuɽ�8>!0����y���*�=�ͱ=��?�Cz�~`�=��<�K�=�e'=�z<�3=�`�=��d��k�����=�S;��=�/����,;ZK�9R�=��-=+�"�cUм*b�=rf�=|���K��@���=z��=���I�\<��=NW�<��V�4�=	/ֽ����S��=��=Z�j=8����2⫼\*��|��=Kq�=�ϩ=��Ǽ���J�=�B�=�1l����=��/g�<J��=�{>RB�=;>�_�������*�p�s=��=*����=C�r=#>���W���^<�$�=:�=S��=-U��1j�<��<&��=Rl�=��n=t�������0ы<͔���B�<�TB��緽O-=�8U���=����d��?=�oo��7<>f�=ү�<�w<B����=�«��EV��"0=���D�=
�=�'T��ܟ�<�/=��½��Y��ti=���0�s=��|���=�/=�T]=���<��=�����>�B=��=��4;�7>h�4<�!k=�-�=����!>�B�}��;��=2k��|��Z~��0V<�yV��=���=�]���Ow=¨%>p)��!}�=�k=y�.<���=H�<��=��5�'�d<���=5����=�ᴽ�6���>��=eg�vG�=�,M=R�=�ծ<-,E=��>T����@�Xm�=l�<��=�˨��>�7�;�j޻�<U=	       D�>�i(�&.���>��K>RT>�ޣ<ڂ�=v��=�      %���괽5���=v�<��A��3=�l =�g@�n�2=�F�=��=��<e��=���=�on=�[V=}�Y=r"�=�A�=͍(=x=��ֻ�F�=�h=�o�;�o�%�=%Z
<Љ�=a	¼�y��RN=���;���t\==u�=g�J��*K=�D=๒�>`�Z-�=�]��k���e.�����<��T���
=�W���;v���|�=`�=��G��.�=�^�<ԝ�=�z=���%��<>��:i=�2B��a�<1Bc=��<P8��WX�d��w���%����Q��z�bv��ϵ�x�r�3�<��<�m-=�x�<%;+��M=JZ����<�Z� �5�-���#�<��=�E"<�)l;^�>=@�V<��4�Gt��sK=l(^=s\j<���f�����o�k�Hp��B�<N�+�U�+==̔�t�r�B=k�}��:Y=��_=�࿻�>�<�/=N�=V[�J��Q�<E�P<G��h�,��Ę=?��벒���u=a�<��=w�Y�C���s�;=-d����e=*�J=^�`��=*��������&�=�dL��
>�s�x=�Ψ=@<�4x=�K�<�_[9�"~=�!�=/��=�C�=b)�=X�7�t�\�繀��]���<�<ݰ�ph*=�y��g�g<@Ԋ��0<~�y=8�)�����_+=BBo����E3a9����XZ��S�I=�8o��߻B���r��<���<JA����==<ۼ]m<�5������ڙ�b��u<�=`3��'�*=f1�#4=�L�	��&ؼ�{(=L�=]Ύ�J�M==�f$���z���=7���E��,���,�5Č�CT��=g7'��U��U}=8U=|p;A]ռj��<=�Q9�HB��qF~=���;��2; g�<;���=��-�r���;=}�b=��2=�sY�[3���	��>��}��</i�=uӁ�����2�=��.��fk<zhR=�6����;�I�<U5��R=B(=�Z����=zP��k͌��GU��d�$������`A��)턽EZ�=��	<��s�7S���k���`=�/�ƭ�<��U;�xP�Tr=ژ ���<�V<����/����Y��o�����Q={�h=�3@=����̃�=�[x��衽z�a��c����<'[�;3&=�[��"N���<\���b��;�Լ�t=9��<g�޼��|=gmR=26�K�7;̹d��{==��X=[���=N��O;��N�<��8�ǡ�t�������R=к!=��W=�d��q�p=�t=��;]xI=��r=^��<aM��
?=��.�A=:ә�x��:��g�G���8��������'=HJ=�?:=���{��F>����=�>"��=��>2�%>�8>�2S=Ax>o =̓�;���=���ု=�ng��?;rnr�
��������"�s�iL�:W�t=�m��y����<�d:x�<U���ޯ��P�=���K����g=�P�=HV�M��{��;�^�������Ȅ�)Sʽ`�<��:�7B=Zh���ph������&o���Ȼ.v.�O}c=ig��	C����}X)������h<;	%�<9�<�<�;7���M�"�F��=��0����<� ��l��0����<Жu���J�F��=('�=�F<�V�<���=9�>��=�z�=M��=�Z��S�`=v�_�q�c=T�<j>�9r�;P�c�3�)�&��=�D�<:Oļ9��=�)���[���Q��ؼ��+E�<�a���W��ư,���ȼz�G�4:N<�=��-��dѻ�q�=`<�=��+�%�s�������=w5��ƕݼ�Y�=Vc�<������=�:=��W=��1=�Î=��<�"o�*��<!��0U=�V[�>i��&'�l�%�~�.=9��=D�f<D�C;��=PU>�<\;�c=�1�=�@+=�ǟ�C��=B_Z=��=K�7��p=�o=� ��ݳ�q�)�F�k� ���ҹ���c/�Bړ��P�=�=�:/�
��"��������<K�P��y�=�����Qa=�L�Ȍ���T=�+r;��w<�kx�ݯ&=&(	�ޠ4<��v�v���p���;,=W��;��;�恜=w�=���u^Y=��������G��:�=tx�o���2,=�.��g-=�GU�(�=�uD��Ν=�j<��;��L=�A��o���Z���<��ٻ�H��ۼ�X�����<�ل=�Aм��Y=Ъ��}w��Ls�IE	=d�5;=��I�����p��Xd�<�l�<7&�����=J�D<��d=D1�ƥ���%=�s�����;��ZmJ�Ꝍ;( =�o��	7���+�BC�<��~<Nǁ<Q=�'��`6|=���<��:S�=��;��P-�m�,�������=@�6����=��L<��}���5Mn=l�>�ſ�=�K~������ܼ��Z=��=xR�<�ǝ=o$8=����$ܼ&�6=5�}��f:=/f�=��)�l��=�Z>��="K��0�U�F�L=-g=��C�^�=�xl�7�P=`��<��ռ����]�����˫�<��C;6��<�=�=�=Qߐ����;�Nz=�|�ֵH��P����<UK#���\��N4<��5���3=[+-��J��2�=<~�P<�$���5��TO�;��Ƚt��=���=*(�=b��=Sђ=�`>K�>��=���=��g�M>�b��:C�;=� .�wSO=�s\=�*[=9'���t���:=ݥt�4>�9<�f=�����=&���_��������0-=��x��md���=�!L��1&J�e�p�􍋼��)�X�D�7�/�a<�<=�:=B��ˌ�<B{� �@�S'���<�/;�s����nּ%��<G�5���P=�0�7�=�J�y��:^���]g��ٯ�AEJ���6��=Ì�;q0I���l=��w=����b�3����Z	>��=��>��1=���=9�>8s�=c�=[Ǧ=��D=G�$����ϥ<�X<���=�ҍ=�����e�.�)��#i�mv��O�V��~��X0��c�l��w=w(;�T)m=�2=�-�<���<�in�eh=�c�<FF��[������O=���:��3='�=��μ�)����h���Ѽ붨=g���p\�l��=%�����=mLV��R=���=�P���Ρ<�ƽ�.���맽0���TS�?>>��睼V�=�PF>�)>��6>���=���=Np�=s�G>��w=s{<�=O���ȃ�<7 q=ޚ�d���u`X=L���e��֙�<Yė=���= ��=00�=��=�<=Kc�:��=e�$����=U�F<&T(=|t���_;?�=C�I=��`=|�8<'{��W*}=��&����<� <Ǆ��Z=ɹ(��2���u����v�t�X=�=Q�M���5��7��B<�l���V[������;�=�9k=>^�<� ���%E=!�I=�=��.����lӾ=��=%��=�TQ�-`������p<:�������5<���<�>���f��9U��&<�Q���OW��4�<��E�Z!-=�"�/��<�(e�o���c���m��s
�Li=\c�b�]hr=��E=�s��cs=�M6�8Ȏ=F=ú'=k<{+�=�J�=q��=6�=.JH��u<�؁��4=i�k���۫<�0=��=��W�Q��=���C�<:=ˬ�=�^]�j���`��;�=pRz=c:�=�����:w/W��� =	Oh���n<�;G=����+�C�j)R<�K=��<8M��C���=�x
����<��<�A#=��<ό=����oO�=�գ��_ ���<b��9=���=Md�=�y=�˽���/�:�z^ټ� ?=�ӼK%��Q<>��ͼݴ��7H��-(���4=e�;�u����<��N=�F����m<�۵;�ϝ�΋\=Y���a^߽2�V��e��-�<˗j�+q�=��4=꼠Q�+�S�[�7�XT��䀽2K#<
}��lߴ�k����ϗ���~�p�d�����y��6=+����Ԅ��?=�q�= w��5=�t�=�q[���c=nQ.=��<`U˼vf�<���4��,�<<�=�=X��=�!�=*���p;=��=IL�<�˻�[�=S1->0}�=Iq�=P�.>[�>#�%>�,�=(WS=�#=[�=���=�yy=ݎ�<��=d�=o�;ⅹ=��Լ�N�=d��<��L��	M���=���[W@=K���#f[���0=G�%�`䫽����>p��m2�=蝽���<uI=ܢ�<�ߴ<{�R���=�}��]�=�\�xP=�i�=M�7=�w~=�=���=�9�=e�h��V�oo<V!�<�S3��1�9s��<�Y���=}�=n��=I��<�|&�	�һ�#u��M�<M�d����V*�k�<YV�=��{=4���D��L��=�k-�5�I<R���󑅽�|m�j�9�Pj;=�*F����O�=���<�n=�!=q<�=Ӑ�=+ڀ��(>H8���<<�N>a	�<��]<�y�=K5=]b{=�9�<��U��HH�Pz���==a�<�h=>�by�<�R�=R�=�RٻU�c=�(=��0=��E��8/=}�^�l��= 	���S�=���@k�=)ZN����ۣ=]�!=۪���O��>� �����{y=r�P�)5�=�4=�(=�<��̼�Gd��P=��u�Uw���h���v�Ae�<��?��_�<��U�q̼�\��H��[�4=c�=�Dl�[pi=���"/�=�w=���!=Ag�����<�ˁ=�/=hg=���<*�Q<������<�'B�O�'=���-��.�<;D��&y=�����9=!梽�=:���i��S2J�;�<d�
���w=�+�:��=�L4=M�漤�k=�=Nf�;��<>CJ=���=��R�z=�Ɂ=yz3:
n}�BM<�-�gV�=	Dy�#�+<X��C[ �Ć`�[|�<���=ݳ���ش�#W�=Q��=�ѼDg���=o[~�����F=>�<p|�=g������<Z�H�>�V����1�㞽t�ü��C=R�]� ���p�;ky=i[��[��v=f��<"�=;��%���;<���PW�=�=�����V�<�{�e=�� �#b�	Po��-�<�Qn��f���G=mW�:`���?+<��;�K_�ڷa�4�Q=N��=b~���<ּ�!���:	x=���=4�{���B=��1=Ԍ�=���<����7�=��嚕=��g=[�=�9�=,b<��=�I�n}
���N�o_�����<AB�=�Lں�.��Z�<=��<B<�BƼuU=����Y��/��RC+=8�^��Ġ;���<;З<���=4K=�x=�X����u���=��A==ۚ;��=��=X0=w;�=���=�ey�]�R=�ͦ;�
�<5�]=��=0K�=�k=��W;I��ͦ�=%�O���d:��i�lʐ�9V<�>%<���~��@ۆ�օ>=ʒI=~[�9����	�=��8�+
�=$ͼ�E�=d�Q<w�B=�"S�µ=�_C����;䐛=$���ܭ<��A=A)"�H���_E(<39�<�ڒ����;���<��<Kv�;�=c�< �c<�l�^jx=��U��D�<NSQ�w?����=���9G^=�vӻ��-����<ɴ�<1c�=]�b=u�J�?=�ٲ<C`L=��4%Y5<~�<�UH��v=[�=�e�<����LY�<���{'�<g�j=X�=.x=�_ؼz�j�p��<M��=��<��~<Bl��B-�=�(J<�΋�ZE���´XI$��d�
	�<{ν`���       �P��d��v
�<����=�F�54�;�Ƚ�9%�Y��=�'ǽ�Oڼ�<4���ԽɎ��@�����á�����8-��Խ�9���9���;:�����輲�꺇��y�<&0�$2�<�7����n<1ѽ�u�H�7;l��Jd���I��}��r�
��~Խ����z��_G�_������<��K�P�ԠW=[Ľ%Ӽg��fa><�#=&_>el�=,�2>_NN>P�>��B>Ea>2\ٽ���HE=Ǔ=�K�|�<�f���%<=)Uν�$��������R<��\^_=�J<����0I=p��.�=��d=�q=�n=v�=Z���7='��<1B���<u#���J4��Q��=���}rϼE�$=�Lt<�o7�e���f����%����6�ҽ�q�����O<�edӼ5�x�ue���Ԛ=��o=L���Ok<�k=f;U<؂_�f��(3���J%���d�:*����%��t.�z\ӽ��^=��¼+���@�ͽ���fZջ��꼮k<���=�=��>-/�=8��=���z|=
0>�<>��c=�q�=��=ح�<ڛ'��4����R�>�o��!Ͻ��Q=6��=�ڽw�=�I�=F��=g����=q1�<Yh�=��>�0+>�59{��=.f/<��4>`�<H&���<Q��=x��<Y��<گ�=�譽� ����=��=��6�b�S���۽NM��0��:��^G̽[Ə�i'v���2<'�����l;��s= �����/��b�=��=�$�=4_K������L��h��xG�Y�-��$���Y��;��>�&�<}������=�  =[�>1� ���=�C�=���=CI�<�jK�*��=��H<>��0�<L�f=��=��=!_�;g�=|�g�h�=~S>�����������:bv>�$��#>V:=�>Y�>羈:e},>	       =����Խ,��>�ἆ��>lj<fd�<��"��(ż       �ݕ�[�=�z>B�%=Ɨw=���=�b�j�����=�{R��">R����==��= �ŽQ�H>�&Q��*�>