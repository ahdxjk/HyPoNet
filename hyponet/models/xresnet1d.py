import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_conv1d import create_head1d, Flatten

from enum import Enum
import re

# delegates
import inspect


def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def store_attr(self, nms):
    "Store params named in comma-separated `nms` from calling context into attrs in `self`"
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(", *", nms):
        setattr(self, n, mod[n])


NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <= 3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, "weight"):
        func(m.weight)
    with torch.no_grad():
        if getattr(m, "bias", None) is not None:
            m.bias.fill_(0.0)
    return m


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm(
        "BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs
    )


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."

    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        padding=None,
        bias=None,
        ndim=2,
        norm_type=NormType.Batch,
        bn_1st=True,
        act_cls=nn.ReLU,
        transpose=False,
        init=nn.init.kaiming_normal_,
        xtra=None,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(
            conv_func(
                ni,
                nf,
                kernel_size=ks,
                bias=bias,
                stride=stride,
                padding=padding,
                **kwargs,
            ),
            init,
        )
        if norm_type == NormType.Weight:
            conv = torch.nn.utils.weight_norm(conv)
        elif norm_type == NormType.Spectral:
            conv = torch.nn.utils.spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act_cls is not None:
            act_bn.append(act_cls())
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn:
            act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


def AdaptiveAvgPool(sz=1, ndim=2):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)


def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)


def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(
        ks, stride=stride, padding=padding, ceil_mode=ceil_mode
    )

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X
class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"

    @delegates(ConvLayer.__init__)
    def __init__(
        self,
        expansion,
        ni,
        nf,
        stride=1,
        kernel_size=3,
        groups=1,
        reduction=None,
        nh1=None,
        nh2=None,
        dw=False,
        g2=1,
        sa=False,
        sym=False,
        norm_type=NormType.Batch,
        act_cls=nn.ReLU,
        ndim=2,
        pool=AvgPool,
        pool_first=True,
        dropout = 0.1,
        **kwargs,
    ):
        super().__init__()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        norm2 = (
            NormType.BatchZero
            if norm_type == NormType.Batch
            else NormType.InstanceZero
            if norm_type == NormType.Instance
            else norm_type
        )
        if nh2 is None:
            nh2 = nf
        if nh1 is None:
            nh1 = nh2
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        layers = (
            [
                ConvLayer(
                    ni,
                    nh2,
                    kernel_size,
                    stride=stride,
                    groups=ni if dw else groups,
                    **k0,
                ),
                ConvLayer(nh2, nf, kernel_size, groups=g2, **k1),
            ]
            if expansion == 1
            else [
                ConvLayer(ni, nh1, 1, **k0),
                ConvLayer(
                    nh1,
                    nh2,
                    kernel_size,
                    stride=stride,
                    groups=nh1 if dw else groups,
                    **k0,
                ),
                ConvLayer(nh2, nf, 1, groups=g2, **k1),
            ]
        )
        self.convs = nn.Sequential(*layers)
        convpath = [self.convs]
        if reduction:
            convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa:
            convpath.append(SimpleSelfAttention(nf, ks=1, sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni != nf:
            idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride != 1:
            idpath.insert((1, 0)[pool_first], pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.dropout(self.act(self.convpath(x) + self.idpath(x)))


######################### adapted from vison.models.xresnet
def init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


class XResNet1d(nn.Sequential):
    @delegates(ResBlock)
    def __init__(
        self,
        block,
        expansion,
        layers,
        p=0.0,
        input_channels=3,
        num_classes=1,
        stem_szs=(32, 32, 64),
        kernel_size=5,
        kernel_size_stem=5,
        widen=1.0,
        sa=False,
        act_cls=nn.ReLU,
        lin_ftrs_head=None,
        ps_head=0.5,
        bn_final_head=False,
        bn_head=True,
        act_head="gelu",
        concat_pooling=True,
        **kwargs,
    ):
        store_attr(self, "block,expansion,act_cls")
        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(
                stem_szs[i],
                stem_szs[i + 1],
                ks=kernel_size_stem,
                stride=2 if i == 0 else 1,
                act_cls=act_cls,
                ndim=1,
            )
            for i in range(3)
        ]

        # block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [
            int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)
        ]
        block_szs = [64 // expansion] + block_szs
        blocks = [
            self._make_layer(
                ni=block_szs[i],
                nf=block_szs[i + 1],
                blocks=l,
                stride=1 if i == 0 else 2,
                kernel_size=kernel_size,
                sa=sa and i == len(layers) - 4,
                ndim=1,
                **kwargs,
            )
            for i, l in enumerate(layers)
        ]

        head = create_head1d(
            block_szs[-1] * expansion,
            nc=num_classes,
            lin_ftrs=lin_ftrs_head,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling,
        )

        super().__init__(
            *stem,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *blocks,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        return nn.Sequential(
            *[
                self.block(
                    self.expansion,
                    ni if i == 0 else nf,
                    nf,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                    sa=sa and i == (blocks - 1),
                    act_cls=self.act_cls,
                    **kwargs,
                )
                for i in range(blocks)
            ]
        )

    def get_layer_groups(self):
        return (self[3], self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self, x):
        self[-1][-1] = x

    def forward(self, combined_signals = None ,ml_features = None, *args, **kwargs):
        # wave_feature = self.mlp(ml_features)
        # combined_features = torch.stack([combined_features, wave_feature], dim=1)  # 鍦ㄧ淮搴?鍫嗗彔
        y = super().forward(combined_signals)

        return y

# xresnets

# input_size(B,L,C)
def _xresnet1d(expansion, layers, **kwargs):
    return XResNet1d(ResBlock, expansion, layers, **kwargs)


def xresnet1d18(**kwargs):
    return _xresnet1d(1, [2, 2, 2, 2], **kwargs)


def xresnet1d34(**kwargs):
    return _xresnet1d(1, [3, 4, 6, 3], **kwargs)


def xresnet1d50(**kwargs):
    return _xresnet1d(2, [3, 4, 6, 3], **kwargs)


def xresnet1d101(**kwargs):
    return _xresnet1d(4, [3, 4, 23, 3], **kwargs)


def xresnet1d152(**kwargs):
    return _xresnet1d(4, [3, 8, 36, 3], **kwargs)


def xresnet1d18_deep(**kwargs):
    return _xresnet1d(1, [2, 2, 2, 2, 1, 1], **kwargs)


def xresnet1d34_deep(**kwargs):
    return _xresnet1d(1, [3, 4, 6, 3, 1, 1], **kwargs)


def xresnet1d50_deep(**kwargs):
    return _xresnet1d(4, [3, 4, 6, 3, 1, 1], **kwargs)


def xresnet1d18_deeper(**kwargs):
    return _xresnet1d(1, [2, 2, 1, 1, 1, 1, 1, 1], **kwargs)


def xresnet1d34_deeper(**kwargs):
    return _xresnet1d(1, [3, 4, 6, 3, 1, 1, 1, 1], **kwargs)


def xresnet1d50_deeper(**kwargs):
    return _xresnet1d(4, [3, 4, 6, 3, 1, 1, 1, 1], **kwargs)


# import torch
# # #
# # 鍒涘缓涓€涓?xresnet1d18 妯″瀷瀹炰緥
# model = xresnet1d18(num_classes=5,input_channels=3,dropout=0.2)
# print(model)
# # 鍒涘缓涓€涓殢鏈虹殑杈撳叆 tensor锛岀淮搴︿负 (batch_size, input_channels, sequence_length)
# # 鍋囪 batch_size=8, input_channels=3, sequence_length=100
# x = torch.randn(8, 100, 3)
#
# # 杩涜鍓嶅悜浼犳挱
# output = model(x)
#
# # 鎵撳嵃杈撳嚭鐨勫舰鐘?
# print("Output shape:", output.shape)
