import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .attentions import ResidualBlock1D, APTx
from .quantizer import FSQ


def sequence_mask(max_length, x_lengths):
    """
    Make a bool sequence mask
    :param max_length: Max length of sequences
    :param x_lengths: Tensor (batch,) indicating sequence lengths
    :return: Bool tensor size (batch, max_length) where True is padded and False is valid
    """
    mask = torch.arange(max_length).expand(len(x_lengths), max_length).to(x_lengths.device)
    mask = mask >= x_lengths.unsqueeze(1)
    return mask


class ConvBlock2D(nn.Module):
    """
    2-D convolutional block that supports:
      • weight-norm wrapping
      • regular or depth-wise-separable conv
      • boolean padding mask (B, 1, H, W)  – keeps padded pixels at 0

    Forward signature
    -----------------
    y = block(x, x_mask=None)

    If x_mask is provided (True = padded), the block applies
    `out = out.masked_fill(mask_expanded, 0)` right *before* the
    non-linearity.  This mirrors the masking strategy used in
    ResidualBlock2D.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int, int] = 3,
            stride: int | tuple[int, int] = 1,
            dilation: int | tuple[int, int] = 1,
            *,
            depthwise: bool = False,
            use_weight_norm: bool = True,
            act: str = "relu",
            dropout: float = 0.1,
            bias: bool = True,
    ):
        super().__init__()

        # ------ util ------ #
        def _make_conv(cin, cout, k, s, d, groups=1):
            padding = (
                d * (k // 2) if isinstance(k, int)
                else (d[0] * (k[0] // 2), d[1] * (k[1] // 2))
            )
            conv = nn.Conv2d(
                cin, cout, k, stride=s, padding=padding,
                dilation=d, groups=groups, bias=bias
            )
            return weight_norm(conv) if use_weight_norm else conv

        # ------ conv path ------ #
        if depthwise:
            self.dw = _make_conv(in_channels, in_channels, kernel_size, stride, dilation,
                                 groups=in_channels)  # depth-wise
            self.pw = _make_conv(in_channels, out_channels, 1, 1, 1)  # point-wise
        else:
            self.conv = _make_conv(in_channels, out_channels, kernel_size, stride, dilation)

        # ------ activation ------ #
        if act.lower() == "gelu":
            self.activation = nn.GELU()
        elif act.lower() == "aptx":
            self.activation = APTx()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.depthwise = depthwise  # store flag for forward()
        self.conv_out = nn.Conv2d(out_channels, 1, 1)

    # --------------------------------------------------------------------- #
    def _apply_mask(self, tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            tensor = tensor.masked_fill(mask.expand_as(tensor), 0.0)
        return tensor

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x       : (B, Cin, H, W)
        x_mask  : (B, 1, H, W) boolean, True = padding
        """
        # (B, H, W)
        x = x.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1)
        if self.depthwise:
            out = self.dw(x)
            out = self._apply_mask(out, x_mask)
            out = self.pw(out)
        else:
            out = self.conv(x)

        out = self._apply_mask(out, x_mask)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv_out(out)
        return out.squeeze(1)


class PreEncoder(nn.Module):
    def __init__(self, mel_channels, channels, kernel_sizes, fsq_levels=[8, 8, 5, 5, 5], dropout=0.1):
        """
        Spectrogram Pre-Encoder.
        ResNet-based autoencoder with configurable encoder and decoder blocks.

        Parameters:
          - mel_channels (int): number of channels in the input spectrogram.
          - channels (list of ints): list of channel dimensions for encoder blocks.
            * The first element is the projected input dimension.
            * The last element is the latent dimension.
          - kernel_sizes (list of ints): list of kernel sizes for each ResidualBlock1D.
            Length should be len(channels) - 1. The decoder will use these lists in reverse.
        """
        super(PreEncoder, self).__init__()
        # Project input from mel_channels to channels[0]
        self.proj = nn.Linear(mel_channels, channels[0])
        self.pre = ConvBlock2D(1, channels[0], kernel_size=5, depthwise=True, act="aptx")
        self.quantizer_dim = len(fsq_levels)
        # Encoder: build a sequence of ResidualBlock1D modules
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock1D(channels[i], channels[i + 1], kernel_size=kernel_sizes[i], dropout=dropout, act="taptx",
                            norm="weight")
            for i in range(len(channels) - 1)
        ])

        # Quantization stage: here we use the latent dimension as the last element of channels.
        latent_dim = channels[-1]

        self.q_in_proj = nn.Linear(latent_dim, self.quantizer_dim)
        self.quantizer = FSQ(levels=fsq_levels)
        self.q_out_proj = nn.Linear(self.quantizer_dim, latent_dim)
        self.codebook_size = 8010  # TODO: dyn calculate this
        self.bos_token_id = 8001
        self.eos_token_id = 8002

        # Decoder: use the reversed lists so that the decoder mirrors the encoder.
        rev_channels = list(reversed(channels))
        rev_kernel_sizes = list(reversed(kernel_sizes))
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock1D(rev_channels[i], rev_channels[i + 1], kernel_size=rev_kernel_sizes[i], dropout=dropout,
                            act="taptx", causal=True, norm="weight")
            for i in range(len(rev_channels) - 1)
        ])
        self.post = ConvBlock2D(1, channels[0], kernel_size=5, depthwise=True, act="aptx")

        # Output projection: map from the decoder’s final channel (channels[0]) back to mel_channels.
        self.out_proj = nn.Linear(channels[0], mel_channels)

    def forward(self, x, x_lengths):
        """
        Forward pass.

        Parameters:
          - x: Tensor of shape (batch, mel_len, mel_channels)
          - x_lengths: (batch,), int lengths of each thing
        Returns:
          - Reconstructed tensor of shape (batch, mel_len, mel_channels)
        """
        # Project input to channel dimension channels[0]
        x = self.proj(x)  # (batch, mel_len, channels[0])
        # Permute to (batch, channels[0], mel_len) for 1D convolutions.
        x = x.permute(0, 2, 1)

        x_mask = sequence_mask(x.size(2), x_lengths)
        x_mask = x_mask.unsqueeze(1)  # (B, 1, T)
        x = self.pre(x, x_mask)

        # Pass through the encoder blocks
        for block in self.encoder_blocks:
            x = block(x, x_mask=x_mask)

        # Permute back to (batch, mel_len, latent_dim)
        x = x.permute(0, 2, 1)
        x = self.q_in_proj(x)
        xhat, indices = self.quantizer(x)
        x = self.q_out_proj(xhat)
        # Permute for the decoder
        x = x.permute(0, 2, 1)

        # Pass through the decoder blocks
        for block in self.decoder_blocks:
            x = block(x, x_mask=x_mask)

        x = self.post(x, x_mask)
        # Permute back to (batch, mel_len, channels[0])
        x = x.permute(0, 2, 1)
        # Final projection back to mel_channels
        x = self.out_proj(x)

        return x

    def encode(self, x, x_mask=None):
        """
        Encodes the input spectrogram into discrete latent indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, mel_len, mel_channels).
          - x_mask: Tensor of shape (batch, mel_len), bool where padded positions are True.
                   (This mask will be passed to each ResidualBlock1D, which is assumed to apply
                   .masked_fill(x_mask, 0) before its activation calls.)
        Returns:
            indices (torch.Tensor): Discrete token indices from the vector quantizer.
        """
        # Project input to latent_dim
        x = self.proj(x)
        # Permute to (batch, latent_dim, mel_len) for convolutional operations
        x = x.permute(0, 2, 1)

        if x_mask is None:
            x_mask = torch.zeros((x.size(0), 1, x.size(2)), device=x.device).bool()

        x = self.pre(x, x_mask)

        # Pass through the encoder blocks
        for block in self.encoder_blocks:
            x = block(x, x_mask=x_mask)
        # Permute back to (batch, mel_len, latent_dim)
        x = x.permute(0, 2, 1)
        # Project to quantizer input dimension (e.g. 4)
        x = self.q_in_proj(x)
        # Quantize and obtain indices
        _, indices = self.quantizer(x)
        return indices.long()  # otherwise cross entropy loss bitches later

    def decode(self, indices, x_mask=None, return_hidden=False):
        """
        Decodes discrete latent indices into a reconstructed spectrogram.

        Args:
            indices (torch.Tensor): Discrete token indices from the vector quantizer.

        Returns:
            x (torch.Tensor): Reconstructed spectrogram of shape (batch, mel_len, mel_channels).
        """
        # Convert indices to quantized latent codes (shape: (batch, mel_len, 4))
        xhat = self.quantizer.indices_to_codes(indices)
        # Project quantized representation back to latent_dim
        x = self.q_out_proj(xhat)
        # Permute to (batch, latent_dim, mel_len) for convolutional operations

        x = x.permute(0, 2, 1)

        if x_mask is None:
            x_mask = torch.zeros((x.size(0), 1, x.size(2)), device=x.device).bool()

        # Pass through the decoder blocks
        for block in self.decoder_blocks:
            x = block(x, x_mask=x_mask)

        if return_hidden:
            last_hid = x.clone()

        x = self.post(x, x_mask)
        # Permute back to (batch, mel_len, latent_dim)
        x = x.permute(0, 2, 1)
        # Project back to the original mel_channels
        x = self.out_proj(x)
        if return_hidden:
            return x, last_hid

        return x
