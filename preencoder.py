import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from collections import OrderedDict

from attentions import ResidualBlock1D, APTx
from quantizer import FSQ
import os
from typing import Optional, Tuple
import typing  # only for the Tuple type in TorchScript signature
from torch.nn.utils import remove_weight_norm


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



# controlflowless, torchscript trace-friendly
def pad_to_pow2_4d(x, x_mask, depth):
    # x: (B, 1, T, F)
    # x_mask: (B, T)
    mult = 1 << depth
    B = x.size(0)
    C = x.size(1)  # expect 1
    T = x.size(2)
    F = x.size(3)

    pad_len = (mult - (T % mult)) % mult

    x_pad = torch.zeros((B, C, pad_len, F), dtype=x.dtype, device=x.device)
    x_out = torch.cat((x, x_pad), dim=2)  # (B,1,T',F)

    m_pad = torch.ones((B, pad_len), dtype=x_mask.dtype, device=x_mask.device)
    m_out = torch.cat((x_mask, m_pad), dim=1)  # (B, T')
    m_out = m_out.reshape(B, 1, m_out.size(1), 1)  # (B,1,T',1)

    return x_out, m_out


# ───────────────────────────── helpers ──────────────────────────────
def wn_conv(in_ch, out_ch, k=3, s=1, g=1):
    p = (k - 1) // 2  # same-padding
    return weight_norm(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=g))


def apply_mask(x, mask4d):
    """Zero-out padded positions (mask=True) with masked_fill."""
    if mask4d is not None:
        x = x.masked_fill(mask4d, 0.0)
    return x


def downsample_mask(mask4d):
    """(B,1,T,1) → pooled ½T in time using max (padding stays True)."""
    return F.max_pool2d(mask4d.float(), kernel_size=(2, 1), stride=(2, 1)).bool()


def upsample_mask(mask4d):
    """Nearest upsample ×2 in time."""
    return F.interpolate(mask4d.float(), scale_factor=(2, 1), mode='nearest').bool()


@torch.jit.script
def crop_to_match(skip, x_like):
    # center-crop skip's time dim to match x_like.size(-2)
    target_T = x_like.size(-2)
    dt = skip.size(-2) - target_T
    if dt > 0:
        start = dt // 2
        return skip[..., start:start + target_T, :]
    else:
        return skip


# ─────────────────────────── building blocks ─────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.1):
        super().__init__()
        self.conv1 = wn_conv(c_in, c_out, k=3)
        self.conv2 = wn_conv(c_out, c_out, k=3)
        self.act = APTx()
        self.do = nn.Dropout(dropout)
        self.match = (c_in == c_out)

    def forward(self, x, mask4d=None):
        x = apply_mask(x, mask4d)
        y = self.do(self.act(self.conv1(x)))
        y = self.do(self.act(self.conv2(y)))
        if self.match:
            y = y + x
        y = apply_mask(y, mask4d)
        return y


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(2, 1))  # time ↓2
        self.conv = ConvBlock(c_in, c_out)

    def forward(self, x, mask4d=None):
        x = self.pool(x)
        mask4 = downsample_mask(mask4d) if mask4d is not None else None
        return self.conv(x, mask4), mask4


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 1), mode='nearest')
        self.conv = ConvBlock(c_in + c_skip, c_out)

    def forward(self, x, skip, mask4d=None):
        x = self.up(x)
        mask4 = upsample_mask(mask4d) if mask4d is not None else None

        skip = crop_to_match(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x, mask4), mask4


# ───────────────────────────── UNet Refiner ──────────────────────────
class UNetRefiner(nn.Module):
    """
    Args:
        in_channels: input channels (e.g. 2, for spec + hidden)
        base_ch    : channels of first stage (default 128)
        depth      : number of down/up steps
    Input:
        x        : (B, C, T, F)
        x_mask   : (B, T)  bool, True = padded   (optional)
    Output:
        residual : (B, T, F)  (use y = x + residual)
    """

    def __init__(self, in_channels, base_ch=128, depth=3, dropout=0.1, input_out_channels=[144, 128]):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]  # e.g. 128,256,512,1024

        self.pre = ConvBlock(in_channels, chs[0], dropout)

        # Down & Up stacks
        self.downs = nn.ModuleList(
            DownBlock(chs[i], chs[i + 1]) for i in range(depth)
        )
        self.mid = ConvBlock(chs[-1], chs[-1], dropout)

        self.ups = nn.ModuleList(
            UpBlock(chs[depth - i], chs[depth - i - 1], chs[depth - i - 1])
            for i in range(depth)
        )

        self.post = wn_conv(chs[0], 1, k=3)
        self.reproj = nn.Linear(input_out_channels[0], input_out_channels[1], bias=False)

    # ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        # x is already (B,C,T,F)
        original_len = x.size(2)
        x, mask4 = pad_to_pow2_4d(x, x_mask, self.depth)

        skips = []
        x = self.pre(x, mask4)

        # Down path
        cur_mask = mask4
        for down in self.downs:
            skips.append(x)
            x, cur_mask = down(x, cur_mask)

        # Bottleneck
        x = self.mid(x, cur_mask)

        # Up path
        for up in self.ups:
            skip = skips.pop()
            x, cur_mask = up(x, skip, cur_mask)

        out = self.post(apply_mask(x, cur_mask))
        out = out.squeeze(1)  # (B,T,F)

        # Crop back to original length
        out = out[:, :original_len, :]

        if x_mask is not None:
            out = out.masked_fill(x_mask.unsqueeze(-1), 0.0)

        out = self.reproj(out)

        return out


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
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None, return_hidden=False) -> torch.Tensor:
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
        hidden = self.activation(out)
        out = self.dropout(hidden)
        out = self.conv_out(out)
        out = out.squeeze(1)

        if return_hidden:
            return out, hidden

        return out


class PreEncoder(nn.Module):
    def __init__(self, mel_channels, channels, kernel_sizes, fsq_levels=[8, 8, 5, 5, 5], dropout=0.1,
                 refiner_base_channels=128, refiner_depth=3, refiner_hidden_proj_divisor=8):
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
        self.codebook_size = 1
        for level in fsq_levels:
            self.codebook_size *= level
        self.bos_token_id = self.codebook_size + 1
        self.eos_token_id = self.codebook_size + 2

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
        self.refiner_hidden_channels = mel_channels // refiner_hidden_proj_divisor
        self.hidden_proj = nn.Linear(channels[0], self.refiner_hidden_channels)

        self.refiner = UNetRefiner(in_channels=1, base_ch=refiner_base_channels, depth=refiner_depth,
                                   input_out_channels=[mel_channels + self.refiner_hidden_channels, mel_channels])

        # self.refiner = torch.jit.script(self.refiner)

    def forward(self, x, x_lengths):
        """
        Forward pass, for training only. For inference see .encode and .decode

        Parameters:
          - x: Tensor of shape (batch, mel_len, mel_channels)
          - x_lengths: (batch,), int lengths of each thing
        Returns:
          - x: Reconstructed tensor of shape (batch, mel_len, mel_channels)
          - x_post: Reconstructed and refined tensor of shape (batch, mel_len, mel_channels)
        """
        # Project input to channel dimension channels[0]
        x = self.proj(x)  # (batch, mel_len, channels[0])
        # Permute to (batch, channels[0], mel_len) for 1D convolutions.
        x = x.permute(0, 2, 1)

        x_mask_orig = sequence_mask(x.size(2), x_lengths)
        x_mask = x_mask_orig.unsqueeze(1)  # (B, 1, T)
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
        decoder_out = x
        for block in self.decoder_blocks:
            decoder_out = block(decoder_out, x_mask=x_mask)

        x_recon_chans = self.post(decoder_out, x_mask)
        # Permute back to (batch, mel_len, channels[0])
        x_recon_chans = x_recon_chans.permute(0, 2, 1)
        # Final projection back to mel_channels
        x_recon = self.out_proj(x_recon_chans)

        # Refiner step:
        # 1. Project hidden state to mel_channels
        hidden_for_refiner = self.hidden_proj(decoder_out.permute(0, 2, 1))

        refiner_in = torch.cat([x_recon, hidden_for_refiner], dim=2)  # append (B, T, Cmel + Chid)
        refiner_in = refiner_in.unsqueeze(1)  # (B, 1, T, Cmel + Chid)
        # 3. Calculate residual
        residual = self.refiner(
            refiner_in.detach(),  # Detach: We want only the refiner to receive GAN gradients.
            x_mask_orig)
        x_post = x_recon + residual

        return x_recon, x_post

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
        decoder_out = x
        for block in self.decoder_blocks:
            decoder_out = block(decoder_out, x_mask=x_mask)

        if return_hidden:
            last_hid = decoder_out.clone()

        x_recon_chans = self.post(decoder_out, x_mask)
        # Permute back to (batch, mel_len, latent_dim)
        x_recon_chans = x_recon_chans.permute(0, 2, 1)
        # Project back to the original mel_channels
        x_recon = self.out_proj(x_recon_chans)

        # Refiner step:
        # 1. Project hidden state to mel_channels
        hidden_for_refiner = self.hidden_proj(decoder_out.permute(0, 2, 1))

        refiner_in = torch.cat([x_recon, hidden_for_refiner], dim=2)  # append (B, T, Cmel + Chid)
        refiner_in = refiner_in.unsqueeze(1)  # (B, 1, T, Cmel + Chid)

        # 3. Calculate residual
        residual = self.refiner(
            refiner_in.detach(),  # Detach: We want only the refiner to receive GAN gradients.
            x_mask.squeeze(1))  # x_mask is (B, 1, T), refiner expects (B, T)
        x_post = x_recon + residual

        if return_hidden:
            return x_post, last_hid

        return x_post


def strip_weight_norm(module):
    for m in module.modules():
        for name in ("weight_g", "weight_v"):
            if hasattr(m, name):
                try:
                    remove_weight_norm(m)
                except Exception:
                    pass


def get_pre_encoder(model_path: str, device: str or torch.device, channels=[384, 512, 768], kernel_sizes=[7, 5, 3],
                    mel_channels=88, fsq_levels=[8, 5, 5, 5], refiner_base_channels=128, refiner_depth=3,
                    refiner_hidden_proj_divisor=8, inference=False):
    """
    Loads a Pre-Encoder model from a checkpoint file.

    Assumes the checkpoint was saved with the training script's structure,
    containing 'model_state_dict' and 'args' (or a compatible dict).

    Args:
        model_path (str): Path to the .pth checkpoint file.
        device (str or torch.device): The device to load the model onto ('cpu', 'cuda', etc.).

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The loaded ResNetAutoencoder1D model instance,
                                 moved to the specified device and set to eval mode.
            - model_args (argparse.Namespace or dict): The configuration arguments
                                                       used to initialize the model,
                                                       loaded from the checkpoint.
    Raises:
        FileNotFoundError: If the model_path does not exist.
        KeyError: If essential keys ('args', 'model_state_dict') are missing
                  from the checkpoint.
        RuntimeError: If load_state_dict fails (e.g., architecture mismatch).
        ImportError: If the ResNetAutoencoder1D class cannot be imported/found.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)  # Load to CPU first

    # --- 2. Instantiate Model ---
    try:
        model = PreEncoder(mel_channels=mel_channels, channels=channels, kernel_sizes=kernel_sizes,
                           dropout=0.0, fsq_levels=fsq_levels, refiner_base_channels=refiner_base_channels,
                           refiner_depth=refiner_depth, refiner_hidden_proj_divisor=refiner_hidden_proj_divisor)
    except NameError:
        raise ImportError(
            "ResNetAutoencoder1D class definition not found. Ensure model.py is accessible or the class is defined.")
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model with loaded config: {e}")

    # --- 3. Load Weights ---
    if 'model_state_dict' in checkpoint:
        pretrained_weights = checkpoint['model_state_dict']
        print("Found weights under 'model_state_dict' key.")

        # Optional: Handle 'module.' prefix (if saved using DataParallel/DDP)
        clean_weights = OrderedDict()
        has_module_prefix = False
        for k, v in pretrained_weights.items():
            if k.startswith('module.'):
                has_module_prefix = True
                clean_weights[k[7:]] = v  # remove `module.`
            else:
                clean_weights[k] = v
        if has_module_prefix:
            print("Removed 'module.' prefix from weight keys.")
        pretrained_weights = clean_weights  # Use the cleaned dictionary

        # Load the weights using strict=True (assumes exact match)
        try:
            model.load_state_dict(pretrained_weights, strict=True)
            print("Successfully loaded model weights.")
        except RuntimeError as e:
            print(f"Error loading state_dict (likely architecture mismatch): {e}")
            raise e  # Re-raise the error

    else:
        raise KeyError(f"Checkpoint missing 'model_state_dict' key containing weights.")

    # --- 4. Final Steps ---
    model.to(device)  # Move model to the target device
    model.eval()  # Set model to evaluation mode

    if inference:
        strip_weight_norm(model)

    print(f"Model loaded onto {device} and set to evaluation mode.")

    return model