from fastai.torch_core import *

__all__ = ['AdaptiveConcatPool2d',
           'Flatten', 'Lambda', 'PoolFlatten', 'View', 'ResizeBatch', 'bn_drop_lin', 'conv2d', 'conv2d_trans',
           'custom_conv_layer', 'NormType', 'relu', 'batchnorm_2d', 'CustomPixelShuffle_ICNR', 'icnr',
           'SelfAttention', 'SequentialEx', 'MergeLayer', 'custom_res_block', 'sigmoid_range',
           'SigmoidRange', 'PartialLayer']


class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func: LambdaFunc):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


class View(nn.Module):
    "Reshape `x` to `size`"

    def __init__(self, *size: int):
        super().__init__()
        self.size = size

    def forward(self, x): return x.view(self.size)


class ResizeBatch(nn.Module):
    "Reshape `x` to `size`, keeping batch dim the same size"

    def __init__(self, *size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        size = (x.size(0),) + self.size
        return x.view(size)


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


def PoolFlatten() -> nn.Sequential:
    "Apply `nn.AdaptiveAvgPool2d` to `x` and then flatten the result."
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten())


NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type == NormType.BatchZero else 1.)
    return bn


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: Optional[nn.Module] = None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


class PooledSelfAttention2d(nn.Module):
    "Pooled self attention layer for 2d."

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.theta = spectral_norm(conv2d(n_channels, n_channels // 8, 1))  # query
        self.phi = spectral_norm(conv2d(n_channels, n_channels // 8, 1))  # key
        self.g = spectral_norm(conv2d(n_channels, n_channels // 2, 1))  # value
        self.o = spectral_norm(conv2d(n_channels // 2, n_channels, 1))
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        # code borrowed from https://github.com/ajbrock/BigGAN-PyTorch/blob/7b65e82d058bfe035fc4e299f322a1f83993e04c/layers.py#L156
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.n_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.n_channels // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.n_channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias=False,
           init: LayerFunc = nn.init.kaiming_normal_) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None: padding = ks // 2
    return init_default(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)


def conv2d_trans(ni: int, nf: int, ks: int = 2, stride: int = 2, padding: int = 0, bias=False) -> nn.ConvTranspose2d:
    "Create `nn.ConvTranspose2d` layer."
    return nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)


def relu(inplace: bool = False, leaky: float = None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


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


class SequentialEx(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i): return self.layers[i]

    def append(self, l): return self.layers.append(l)

    def extend(self, l): return self.layers.extend(l)

    def insert(self, i, l): return self.layers.insert(i, l)


class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x): return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


def custom_res_block(nf, dense: bool = False, norm_type: Optional[nn.Module] = nn.BatchNorm2d, bottle: bool = False,
              **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(custom_conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        custom_conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    "Sigmoid module with range `(low,x_max)`"

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x): return sigmoid_range(x, self.low, self.high)


class PartialLayer(nn.Module):
    "Layer that applies `partial(func, **kwargs)`."

    def __init__(self, func, **kwargs):
        super().__init__()
        self.repr = f'{func}({kwargs})'
        self.func = partial(func, **kwargs)

    def forward(self, x): return self.func(x)

    def __repr__(self): return self.repr


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Debugger(nn.Module):
    "A module to debug inside a model."

    def forward(self, x: Tensor) -> Tensor:
        set_trace()
        return x


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


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