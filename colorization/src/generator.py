import torch
from torch import nn
import torchvision.models as models


from fastai.vision.learner import create_body
from fastai.layers import SequentialEx
from fastai.vision.models.unet import (
    UnetBlock,
    ConvLayer,
    PixelShuffle_ICNR,
    ResizeToOrig,
    MergeLayer,
    ResBlock,
    model_sizes,
    hook_outputs,
    dummy_eval,
    BatchNorm,
    apply_init,
    _get_sz_change_idxs,
)


class CustomUnet(SequentialEx):
    def __init__(
        self,
        n_in=1,
        n_out=2,
        img_size=256,
        blur=False,
        act_cls=nn.ReLU,
        norm_type=None,
        self_attention=True,
        self_attention_xtra=False,
    ):
        img_size = (img_size, img_size)

        # initialize the pretrained encoder
        backbone = models.resnet34()
        encoder = create_body(backbone, n_in=n_in, pretrained=True, cut=-2)

        # get the hooks indexs
        sizes = model_sizes(encoder, size=img_size)
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
        sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        dummy_eval(encoder, img_size)

        ni = sizes[-1][1]

        # middle block
        middle = nn.Sequential(
            ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type),
            ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type),
        )

        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle]

        # Decoder blocks
        d1 = UnetBlock(
            512,
            256,
            sfs[0],
            blur=blur,
            self_attention=self_attention_xtra,
            act_cls=act_cls,
            norm_type=norm_type,
        )
        layers.append(d1)

        d2 = UnetBlock(
            512,
            128,
            sfs[1],
            blur=blur,
            self_attention=self_attention,
            act_cls=act_cls,
            norm_type=norm_type,
        )
        layers.append(d2)

        d3 = UnetBlock(
            384,
            64,
            sfs[2],
            blur=blur,
            act_cls=act_cls,
            norm_type=norm_type,
        )
        layers.append(d3)

        d4 = UnetBlock(
            256,
            64,
            sfs[3],
            blur=blur,
            act_cls=act_cls,
            norm_type=norm_type,
            final_div=False,
        )
        layers.append(d4)

        # final upsample
        final_up = PixelShuffle_ICNR(
            96, 96, scale=2, act_cls=act_cls, norm_type=norm_type
        )
        layers.append(final_up)

        toOrig = ResizeToOrig()
        layers.append(toOrig)

        merge = MergeLayer(dense=True)
        layers.append(merge)

        res_block = ResBlock(1, 97, 97, act_cls=act_cls, norm_type=norm_type)
        layers.append(res_block)

        last = ConvLayer(97, n_out, ks=1, act_cls=None, norm_type=norm_type)
        layers.append(last)

        apply_init(
            nn.Sequential(layers[3], layers[-2]), nn.init.kaiming_normal_
        )  # initialize the middle block and res block
        super().__init__(*layers)
