import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Union, Tuple, List # Added typing imports

# Import sequence_mask from the preencoder module within the same package
from preencoder import sequence_mask

class ChannelSELayerMasked(nn.Module):
    """
    Squeeze-and-Excitation that supports a padding mask.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input.
    reduction_ratio : int, default=2
        Channel‐reduction factor (same as the classic SE block).

    Notes
    -----
    * `padding_mask` is expected to be a **bool tensor** with shape
      (B, 1, H, W).  `True` = padded, `False` = valid.
    * If `padding_mask` is omitted (or None), the layer behaves exactly
      like a standard SE block.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        super().__init__()
        reduced = max(1, num_channels // reduction_ratio)

        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced,   num_channels, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,              # (B, C, H, W)
        padding_mask=None  # (B, 1, H, W) -- True = padded
    ):
        B, C, H, W = x.shape

        # ---------- SQUEEZE (masked global average) ------------------
        if padding_mask is None:
            # Vanilla SE: mean over spatial dims
            squeeze = x.view(B, C, -1).mean(dim=2)           # (B, C)
        else:
            # Exclude padded positions
            #   mask_valid : (B, 1, H, W)   True = valid
            mask_valid = ~padding_mask.bool()
            # prevent div-by-zero
            denom = mask_valid.sum(dim=(2, 3), keepdim=False).clamp(min=1)  # (B,1)
            # spatial sum over valid positions
            summed = (x * mask_valid).view(B, C, -1).sum(dim=2)             # (B,C)
            squeeze = summed / denom                                        # (B,C)

        # ---------- EXCITATION ---------------------------------------
        excite = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze))))  # (B,C)

        # ---------- SCALE --------------------------------------------
        y = x * excite.view(B, C, 1, 1)

        return y


class MelSpectrogramPatchDiscriminator2D(nn.Module):
    """
    2-D PatchGAN discriminator for (time × mel) spectrograms.

    Input  : (B, T, F)  – time major
    Output : logits      (B, 1, H, W)
             patch_mask  (B, 1, H, W)  -- True = *valid* patch

    Args
    ----
    mel_channels   : number of mel bins (F)
    hidden_channels: list[int] – output channels per conv block
    kernel_sizes   : list[int] – square kernels, len = len(hidden_channels)+1
    stride         : tuple(int, int) or list[tuple(int,int)]       – stride for down-sampling conv blocks
    """

    def __init__(
            self,
            mel_channels: int,
            hidden_channels: list = (64, 128, 256, 512),
            kernel_sizes: list = (7, 5, 5, 3, 3),
            stride: Union[int, Tuple[int, int], List[Tuple[int, int]]] = (2, 2),
            lengthwise_only=False,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(hidden_channels) + 1, (
            "kernel_sizes must be hidden_channels len + 1"
        )

        # --- Convolutional backbone ------------------------------------
        self.convs = nn.ModuleList()
        self.mel_channels = mel_channels
        in_ch = 1  # we keep a single input channel and treat (F,T) as H×W

        ret_features_map = [True] * (len(hidden_channels) + 1)
        ret_features_map[0] = False
        ret_features_map[1] = False
        ret_features_map[-1] = False
        self.ret_features_map = ret_features_map

        # Build a per-layer list of (h_stride, w_stride)
        if isinstance(stride, int):
            layer_strides = [(1, stride)] * len(kernel_sizes)
        elif isinstance(stride, tuple) and len(stride) == 2:
            layer_strides = [stride] * len(kernel_sizes)
        else:
            # list of tuples
            assert len(stride) == len(kernel_sizes), "stride list must match kernel_sizes"
            layer_strides = [tuple(s) for s in stride]

        # intermediate layers
        for out_ch, k, (sh, sw) in zip(hidden_channels, kernel_sizes[:-1], layer_strides[:-1]):
            if lengthwise_only:
                # only convolve / stride in time (width)
                kernel = (1, k)
                stride_ = (1, sw)
                padding = (0, (k - 1) // 2)
            else:
                if isinstance(k, tuple):
                    k1, k2 = k
                else:
                    k1 = k
                    k2 = k
                # square conv
                kernel = (k1, k2)
                stride_ = (sh, sw)
                padding = ((k1 - 1) // 2, (k2 - 1) // 2)

            self.convs.append(
                spectral_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel,
                        stride=stride_,
                        padding=padding,
                    )
                )
            )
            in_ch = out_ch

        # final logits conv (always square & stride=1)
        k = kernel_sizes[-1]
        if isinstance(k, tuple):
            k1, k2 = k
        else:
            k1 = k
            k2 = k

        pad = (0, (k - 1) // 2) if lengthwise_only else ((k1 - 1) // 2, (k2 - 1) // 2)
        kernel = (1, k) if lengthwise_only else (k1, k2)
        self.convs.append(
            spectral_norm(
                nn.Conv2d(in_ch, 1,
                          kernel_size=kernel,
                          stride=(1, 1),
                          padding=pad)
            )
        )

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.se_block = ChannelSELayerMasked(in_ch, 8)

        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_mask(self, T: int, Freq: int, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths : (B,) – number of valid time frames
        returns : (B, 1, F, T) – True = padded
        """
        # time mask (B, T)
        tmask = sequence_mask(T, lengths)  # True = padded
        # broadcast across frequency bins and channel dim
        return tmask.unsqueeze(1).unsqueeze(2).expand(-1, 1, Freq, -1)

    # ------------------------------------------------------------------
    def forward(
            self,
            x: torch.Tensor,  # (B, T, F)
            x_lengths: torch.Tensor,  # (B,)
            return_features: bool = False,
    ):
        B, T, _ = x.shape

        # build padded mask BEFORE any conv
        padded_mask = self._build_mask(T, self.mel_channels, x_lengths)  # (B,1,F,T)

        # bring to 2-D conv layout
        # (B, Lmel, subCmel) => (B, subCmel, Lmel)
        # We want Cmel to be the 1st dim after batch because our subbin Ds
        # stide along the 2nd dim only
        out = x.transpose(1, 2).unsqueeze(1)  # (B,1,F,T)

        features = []

        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                out = self.se_block(out, padded_mask)

            out = self.activation(conv(out))

            # ----- down-sample mask to match feature map --------------
            stride_h, stride_w = conv.stride
            if stride_h > 1 or stride_w > 1:
                padded_mask = F.max_pool2d(
                    padded_mask.float(),
                    kernel_size=(stride_h, stride_w),
                    stride=(stride_h, stride_w),
                    ceil_mode=True,
                ).bool()

            # zero-out fully padded patches
            out = out.masked_fill(padded_mask, 0.0)

            if return_features and self.ret_features_map[i]:
                features.append((out, padded_mask))

        # -------- flip mask so True = valid for loss -------------------
        patch_mask = ~padded_mask  # (B,1,H,W)

        if return_features:
            return out, patch_mask, features
        return out, patch_mask


class MultiBinDiscriminator(nn.Module):
    """
    Splits the mel axis into `n_bins` equal bands and runs an independent
    MelSpectrogramPatchDiscriminator on each band.
    """

    def __init__(
            self,
            mel_channels: int,
            n_bins: int = 4,
            hidden_channels: list = (64, 128, 256, 512),
            kernel_sizes: list = (7, 5, 5, 3, 3),
            n_no_strides: int = 2,
    ):
        super().__init__()
        assert mel_channels % n_bins == 0, "mel_channels must divide n_bins"

        sub_hidden_channels = []
        for h in hidden_channels:
            assert h % n_bins == 0, f"hidden size {h} must divide n_bins"
            sub_hidden_channels.append(h)

        self.n_bins = n_bins
        bin_size = mel_channels // n_bins
        strides_lst = []
        for i in range(len(kernel_sizes)):
            # first n_no_strides layers: no downsampling in time
            if i < n_no_strides:
                strides_lst.append((1, 1))
            else:
                # after that, stride=2 along time only
                strides_lst.append((1, 2))

        ksizes_lst = [(3, ks) for ks in kernel_sizes]

        self.discriminators = nn.ModuleList(
            [
                MelSpectrogramPatchDiscriminator2D(
                    mel_channels=bin_size,
                    hidden_channels=list(sub_hidden_channels),
                    kernel_sizes=ksizes_lst,
                    stride=strides_lst,  # stride
                )
                for _ in range(n_bins)
            ]
        )

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                return_features: bool = False):
        """
        x: (B, T, C_mel) – full spectrogram
        returns lists of per-bin outputs.
        """
        splits = torch.split(x, x.size(-1) // self.n_bins, dim=-1)

        outs, masks, feats = [], [], []
        for disc, sub_x in zip(self.discriminators, splits):
            if return_features:
                o, m, f = disc(sub_x, x_lengths, True)
                outs.append(o); masks.append(m); feats.append(f)
            else:
                o, m = disc(sub_x, x_lengths, False)
                outs.append(o); masks.append(m)

        if return_features:
            return outs, masks, feats
        return outs, masks

