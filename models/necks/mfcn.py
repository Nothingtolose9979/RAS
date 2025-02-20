import torch
import torch.nn as nn
import torch.nn.functional as F

# MFCN: multi-scale feature concat network
__all__ = ["MFCN"]


class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides, use_patchify, patchsize, patchify_type="avg"):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize

        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]
        self.use_patchify = use_patchify
        self.patchsize = patchsize
        self.patchify_type = patchify_type

    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        feature_list = []
        # patchify, resize, & concatenate
        for i in range(len(features)):
            if self.use_patchify:
                feature_patchify = self.patchify(features[i], self.patchsize)
            else:
                feature_patchify = features[i]
            upsample = self.upsample_list[i]
            feature_resize = upsample(feature_patchify)
            feature_list.append(feature_resize)

        feature_align = torch.cat(feature_list, dim=1)

        return {"feature_align": feature_align, "outplane": self.get_outplanes()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides

    def patchify(self, features, patchsize=3):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * c * w//stride * h//stride, c]
        """
        padding = int((patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=patchsize, stride=1, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, *features.shape[2:]
        ) #b x c x patchsize x patchsize x w x h
        if self.patchify_type == "avg":
            pool_features = unfolded_features.mean(2).mean(2)
        elif self.patchify_type == "max":
            pool_features = unfolded_features.max(2)[0].max(2)[0]
        elif self.patchify_type == "avg+max":
            avg_features = unfolded_features.mean(2).mean(2)
            max_features = unfolded_features.max(2)[0].max(2)[0]
            pool_features = (avg_features + max_features) / 2
        else:
            assert False

        return pool_features
