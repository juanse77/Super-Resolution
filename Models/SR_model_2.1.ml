��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
SuperResolution
qNNtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)RqX   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)Rq(X   convq(h ctorch.nn.modules.conv
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
q%X   2506830152624q&X   cuda:0q'M�Ntq(QK (KKKKtq)(KKKKKtq*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   2507998040432q2X   cuda:0q3KNtq4QK K�q5K�q6�h	)Rq7tq8Rq9�h	)Rq:�q;Rq<uhh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBhh	)RqCX   trainingqD�X   in_channelsqEKX   out_channelsqFKX   kernel_sizeqGKK�qHX   strideqIKK�qJX   paddingqKKK�qLX   dilationqMKK�qNX
   transposedqO�X   output_paddingqPK K �qQX   groupsqRKX   padding_modeqSX   zerosqTubX   upsampleqU(h ctorch.nn.modules.pixelshuffle
PixelShuffle
qVXV   C:\Anaconda3\.conda\envs'\torch_env\lib\site-packages\torch\nn\modules\pixelshuffle.pyqWX)  class PixelShuffle(Module):
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
qXtqYQ)�qZ}q[(hhhh	)Rq\hh	)Rq]hh	)Rq^hh	)Rq_hh	)Rq`hh	)Rqahh	)Rqbhh	)RqchD�X   upscale_factorqdKubuhD�ub.�]q (X   2506830152624qX   2507998040432qe.�      *=>�6=U>��0>��>v�u;l}i>;�}=��=#`D=p�^>R�!>ΉM=bC�=���=AB2�S<>�1�=/�<��N>V�2>a�8>O�=�hx;��=������=���;�����<�����>o�V= �?;c4o=�_>zd�<BR�f�&>�3�<�׹=�2!=3g'=4G>�����}R�+|>�w�=d������=k��ĭ�=�V<1��<wˋ<��>�Y�e=� >r�=D,�<6J�=��>�c<��=����=|=�=���[�[�@7�i,�6��=Zn�=���=���=l�=Ǌ�<�~�=�qG=e]>1iR=��|>���=$a=��u=�n>5E>�̆=FH>ӽ�;���=-�N>.�=��G=�tR<�X>4$=>�T�==�<k�=�✼%#�;$F�T_�$1���v���>v�<�h�<	� >fw<���ʰ��L�=��1=H�= �=+��=���=�J=�
>�sV=L�μp	�=G;_=הI<k/>���;}��B%��FL>hJ.�#� >��I=�l=Y>�)">�>��?<�Ψ�@����R�=6;�Q��Zqw�✦�6�;z��=��w=��D>nu=��=�%<��=�ϳ=DH8>0�<�.	>�@>uC�O!`>GQ>ARO>�;~=�"=�RI>��=kJ>ws>=�>�N>x��=�D>Rm>�A)���;���>2k#=��=+o�=)�=x�>D�)��臽�W=�;��=�>��=9Z�<_U�;#Q_�k���>^?�=YN=�Vk=�Ph���<r]�=4i���>
��=�-߼�q�=�x	>��=�9���3�=���=�#>%�<�;m��ħ<F_�<�t����8��<$S�=�F=���=��<��
=���=��?>�8�=/4->�5>(�i=�SP>}Jh>p^>��=8�8>���<��=���=4U�=>��=��]=V��=By�=�3�=���= p�=6KD=��=U6>EI>|Z��L@��p=/=P=�]/=�=�;71޼;F���S�=��<;�->!�a�� =q[�=��=tR!�BX�=z�=�9�<���=�,�<kY�=�b��
�P�B�E�=R��=�R��5�=�O	>P�<�=Q��=E�=7	H<���=�L!>SG4�Ϯ=���=��<k�W��m�=������;���=s�=��v=�0�=��>ø�=Ƕ��ߟ�>���<��#=�X>U�=:�\=k�8;](>O�=s�R��Az=7��<�rC��dA=?��=��~R:<]�>������}=> 0���=ѯ>�Q>��J���;,v=b��=����'W��X�=L��=5�8>`�;>�QU<h�:�6X�=��Ȟ{�9h[<z�=��>!��=�)>�5>(�>�س=��=���=�B�c��<¶U�W��<®B>�� >���=�j>oz�=�Ο=�0>�9&>�Q(>Z��=/8>Gu��Ux>U%?=��>*$��c�=�5Y<U��=��=�4>(�=%��=f�J��넻!M='MD������S��K�8<��<�y�==m�=��B=?<�8>�� >�@<��H��I
=�-+=)k��h��<ܮ>��s��j��\����>\��=���=|�n��T<N�>�c>��=�C=�p�=V�=]��=i�H=��=؞�=�q:�і>>_%�=�=���=>qc=�8>���=P�v=���=�>��=+N>*�o����=�s->�!�;FX>F�=�F���=:c�=	e<�q4>aO�=_H�=�$=G�	>\�<
O�=�P���C=��`�H�=U�>�@f�=t����>$���8Q�=ۊ>=�4>Y�>�:��`��SeI=d�=[���|�ۣ.=�����=����;�=/�=��ؼ�	>�=� �=���=fT�:�'�=�&>���=gT=>2�={p�=�o�<���=��=e(>�w�<��*=-����5>�o=���=�
�Á�=j��<��>�Y���<ĭc=������<���<W�/�'Z5>�I�<~��<H>�
>5�&>ءZ=�z:>!�<�+F>ŇN>^+�=(r+>��=�g�<)�>�X$��Qp=m=>���=��m��y�=�N����=��=��>�`�=�=�l�=��<P�H<n�6�ƍ$=���=�C�=�R}=���<%�Ϲ����=�0����T=�>F��=l�>���=��=&�=j�V=1>��>�<���==� 5>�3M>u>>�Z�=�8Z>k�=U"�<�e���>p��=]7=�C����*�~���{`�{A�<��3<~�<k��=��b>��S=�h=��;,f= cE=v�>>k^\>�<+>0`%>�vO=��#>A�=j�.>��=kʕ=�=t؛<<ԯ=�cỨ�꼀8+��}�<?�=J��=�Wa��6=y:6����J�߼��=Uf7=t੽���\�u���9��<�у���=� ���[�<qrF=�u�==�>J�;���< '�;��=�"�=g��<�~j����=�=_#>&
�G�(<}��=-�=��>:�H<���=�=G�-<��=tn]=_>@��=�o>�e�=�f
>��M>��s=D��=oP�=2�H>�6>/�?>Y�=�C�=h�7>J�>1�>�32>�C�=��n>�zM>��>��=��>/�I=���=�a�=O�5��½1g�=�+�i������<.ڽ��Z<���=:^d�0�=�A����=£�`�)=���;8���t��=�1۽�1ϼeM���h;��<⦵���#�D��=>��=:7>E_A<l>�W��*>ba�=�#�=�"=�>�KZ=�T�<n>�<�M�WZJ=�,_�yD4=�����N$>zy�:�i'>��m���:>e�5=��z=�>'!}="�=��&>�X�>���=�C5>�=��=5�=
�B>5�=�B>�6i>�~f>f�O>�ȱ=�՝=��=׆S>:Ri>�oF><�=�Z�=�:}ؽ|�i<�+=wh]��O[�ww�؍ѽ�\k=�L�=�H�=K=��ƽ�v�=Ň�=�u[=_`��Y�=����=}i���^=���Li�<�]�=�p=\�~<���=� > ����`>W]�=?��9S�U���>��=U�%=+��=���=�\�=s��<�s=�$�3t>�?���?Z<�>��<<�{�=���=��>X��<_�a>�q>���=:�7>^�~=x��=۟>��U>���=,��==�'>�T>6�>%�>�':=�.Q>�=�=���={ȁ>��:>��;>���1n!��b�R �=��L��=a�l= %½��^<W5=u{���[�=0>�:�`"=<�=<! <H.�=�i<f�=�>;²�����e�=6λ���4���c��!�=�s�<}��=�>�][=�>����1��ɽ�<`] >X`">qn�<�ְ����<fɃ=&���,>*�(>��ļ�=L�j=�A�<���=AV�خ>�^>e,^>Mr >�:�=���=��=��/>��'>b��=��U=�+�>�98>Q�W>u�="��= �=W3>�m{>�!>0]8>1s>|e=�>��>       ��W+�I�
����ǂ�y_�xK����A������