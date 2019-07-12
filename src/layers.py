"""Modified layers from fastai"""

from fastai.torch_core import *
from fastai.layers import *
from nntoolbox.vision import ResNeXtBlock, ConvolutionalLayer


def custom_conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[nn.Module]=nn.BatchNorm2d,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type is not None
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(norm_type(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


from nntoolbox.hooks import InputHook
class CustomMergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, input_hook: InputHook, dense: bool=False, remove_store: bool=True):
        super().__init__()
        self.dense = dense
        self.input_hook = input_hook
        self.remove_store = remove_store

    def forward(self, x):
        op = torch.cat([x, self.input_hook.store], dim=1) if self.dense else (x + self.input_hook.store)
        if self.remove_store:
            self.input_hook.store = None
        return op


def custom_res_block(nf, dense: bool = False, norm_type: Optional[nn.Module] = nn.BatchNorm2d, bottle: bool = False,
              **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(custom_conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        custom_conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


class CustomPixelShuffle_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self, ni: int, nf: int = None, scale: int = 2, blur: bool = False, norm_type=nn.BatchNorm2d,
                 leaky: float = None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class CustomResidualBlockPreActivation(ResNeXtBlock):
    def __init__(self, in_channels, activation=nn.ReLU, normalization=nn.BatchNorm2d):
        super(CustomResidualBlockPreActivation, self).__init__(
            branches=nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReplicationPad2d(1),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=0,
                            activation=activation, normalization=normalization
                        ),
                        nn.ReplicationPad2d(1),
                        ConvolutionalLayer(
                            in_channels, in_channels, 3, padding=0,
                            activation=activation, normalization=normalization
                        )
                    )
                ]
            ),
            use_shake_shake=False
        )


class ReflectionPaddedConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
        super(ReflectionPaddedConv, self).__init__(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            activation()
        )
