import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from math import sqrt
from typing import Optional

# From model/subatts.py
class APTx(nn.Module):
    """
    APTx: Alpha Plus Tanh Times, an activation function that behaves like Mish,
    but is 2x faster.

    https://arxiv.org/abs/2209.06119
    """

    def __init__(self, alpha=1, beta=1, gamma=0.5, trainable=False):
        """
        Initialize APTx initialization.
        :param alpha: Alpha
        :param beta: Beta
        :param gamma: Gamma
        :param trainable: Makes beta and gamma trainable, dynamically optimizing the upwards slope and scaling
        """
        super(APTx, self).__init__()
        self.alpha = alpha
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x

class TransposeLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps, affine)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply LayerNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x

# From model/attblocks.py
def masked_fill_(x: torch.Tensor, mask: torch.Tensor, fill_value: float):
    """
    In-place masked fill. Where mask == True, fill `x` with `fill_value`.

    Args:
        x (torch.Tensor): Tensor to fill. Shape can be (B, C, L) or other shapes
                          broadcastable with the mask.
        mask (torch.Tensor): Boolean mask. Common shapes are (B, 1, L) or (B, L).
                             (True means invalid/padded position).
        fill_value (float): Value to fill with.
    """
    # Determine the target mask shape for broadcasting based on x's dimensions
    if x.ndim == 3 and mask.ndim == 3:  # e.g., x is (B, C, L), mask is (B, 1, L)
        if mask.shape[1] == 1 and x.shape[1] != 1:
            mask_expanded = mask.expand(-1, x.shape[1], -1)
        else:
            mask_expanded = mask  # Assume mask is already (B,C,L) or correctly broadcastable
    elif x.ndim == 3 and mask.ndim == 2:  # e.g., x is (B, C, L), mask is (B, L)
        mask_expanded = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
    elif x.ndim == mask.ndim:  # e.g. x is (B,L) and mask is (B,L), or x is (B,C,1) and mask is (B,1,1)
        mask_expanded = mask  # No expansion needed if dimensions match or broadcasting handles it
    else:
        # Fallback or error for unhandled mask/data shape combinations if necessary
        # For now, assume broadcasting will work or mask is already correct
        mask_expanded = mask

    x = x.masked_fill(mask_expanded.bool(), fill_value)
    return x

def masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes max over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Max pooled tensor. Shape: (B, C, 1).
    """
    x_clone = x.clone()
    # Fill invalid positions with a very small number so they won't dominate max
    masked_fill_(x_clone, mask, float('-inf'))
    max_vals, _ = x_clone.max(dim=-1, keepdim=True)  # shape: (B, C, 1)
    return max_vals

def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes average over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Average pooled tensor. Shape: (B, C, 1).
    """
    x_clone = x.clone()

    # Invert the mask to get a "valid" mask where True = valid
    valid_mask = ~mask  # shape: (B, 1, L)

    # Expand valid_mask to (B, C, L) if necessary, for element-wise operations
    if valid_mask.shape[1] == 1 and x_clone.shape[1] != 1:
        expanded_valid_mask = valid_mask.expand(-1, x_clone.shape[1], -1)  # (B, C, L)
    else:
        expanded_valid_mask = valid_mask  # Handles cases where C=1 or mask is already expanded

    # Fill invalid positions with zero, so they don't contribute to sum
    # Use the inverse of expanded_valid_mask (i.e., the original mask concept for invalid positions)
    x_clone.masked_fill_(~expanded_valid_mask, 0.0)

    # Sum across seq_len
    sum_vals = x_clone.sum(dim=-1, keepdim=True)  # shape: (B, C, 1)

    # Count of valid positions per (B, C)
    # Sum the expanded_valid_mask along L to get (B,C,1) counts directly.
    counts = expanded_valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # shape: (B, C, 1)

    avg_vals = sum_vals / counts
    return avg_vals

def causal_masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes causal max over the last dimension (seq_len) while ignoring masked (padded) positions.
    For each position i, the max is taken over x[:, :, 0:i+1].
    Returns a tensor of shape (B, C, L).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Causal max pooled tensor. Shape: (B, C, L).
    """
    x_clone = x.clone()
    # Fill padded positions with -inf so they are ignored by cummax unless all previous are -inf.
    masked_fill_(x_clone, mask, float('-inf'))
    # cummax computes cumulative max along the specified dimension
    causal_max_vals, _ = torch.cummax(x_clone, dim=-1)  # shape: (B, C, L)
    # If a position was originally masked (and thus -inf), and all preceding elements were also masked (or smaller),
    # it will remain -inf. This is generally desired for attention mechanisms.
    return causal_max_vals

def causal_masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes causal average over the last dimension (seq_len) while ignoring masked (padded) positions.
    For each position i, the average is taken over x[:, :, 0:i+1].
    Returns a tensor of shape (B, C, L).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Causal average pooled tensor. Shape: (B, C, L).
    """
    x_clone = x.clone()

    # Invert mask: True means valid
    valid_mask = ~mask  # Shape: (B, 1, L)

    # Fill invalid positions with zero for summation
    masked_fill_(x_clone, mask, 0.0)

    # Cumulative sum of values
    sum_causal = torch.cumsum(x_clone, dim=-1)  # Shape: (B, C, L)

    # Cumulative sum of valid counts
    # valid_mask (B, 1, L) is broadcasted during division.
    counts_causal = torch.cumsum(valid_mask.float(), dim=-1)  # Shape: (B, 1, L)

    # Clamp counts to avoid division by zero. If counts_causal is 0, avg is 0.
    counts_causal_clamped = counts_causal.clamp(min=1.0)

    avg_causal = sum_causal / counts_causal_clamped  # Broadcasts counts_causal over C dim

    # Where counts_causal was originally 0 (all preceding steps masked), avg_causal will be 0.
    # We need to ensure that these positions are correctly handled, e.g. if they should be 0.
    # If all elements up to 't' are masked, sum_causal is 0, counts_causal is 0 (clamped to 1). Result is 0.
    # This is generally a safe default.
    avg_causal.masked_fill_(counts_causal == 0, 0.0)  # Explicitly set to 0 if no valid elements

    return avg_causal

class CAM1D(nn.Module):
    """
    Channel Attention Module for 1D sequences.
    - Takes (B, C, L) as input.
    - If not causal: Pools across the L dimension (with masked max & avg) to get (B, C, 1).
    - If causal: Pools causally across L to get (B, C, L).
    - Then uses an MLP (two linear layers) to compute channel attention.
    - Finally multiplies it with the input (while respecting the mask).
    """

    def __init__(self, channels: int, reduction_ratio: int, causal: bool = False):
        super(CAM1D, self).__init__()
        self.channels = channels
        self.r = reduction_ratio
        self.causal = causal

        self.mlp = nn.Sequential(  # Renamed from 'linear' to 'mlp' for clarity
            nn.Linear(self.channels, self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.r, self.channels, bias=True)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
        Returns:
            torch.Tensor: Output tensor after channel attention. Shape: (B, C, L).
        """
        B, C, L_dim = x.shape  # Use L_dim to avoid conflict with L in permute

        if self.causal:
            # Causal masked pooling: Output shape (B, C, L_dim)
            max_pool_out = causal_masked_max_pool1d(x, mask)
            avg_pool_out = causal_masked_avg_pool1d(x, mask)

            # Permute for MLP: (B, C, L_dim) -> (B, L_dim, C) to apply MLP on C features per time step
            max_pool_perm = max_pool_out.permute(0, 2, 1)  # (B, L_dim, C)
            avg_pool_perm = avg_pool_out.permute(0, 2, 1)  # (B, L_dim, C)

            # Feed through MLP
            mlp_max = self.mlp(max_pool_perm)  # (B, L_dim, C)
            mlp_avg = self.mlp(avg_pool_perm)  # (B, L_dim, C)

            # Sum
            attn_logits = mlp_max + mlp_avg  # (B, L_dim, C)

            # Channel attention map
            attn_map = torch.sigmoid(attn_logits)  # (B, L_dim, C)

            # Permute back: (B, L_dim, C) -> (B, C, L_dim)
            attn_map = attn_map.permute(0, 2, 1)  # (B, C, L_dim)
        else:
            # Global masked pooling: Output shape (B, C, 1)
            max_pool_out = masked_max_pool1d(x, mask)
            avg_pool_out = masked_avg_pool1d(x, mask)

            # Flatten for MLP: (B, C, 1) -> (B, C)
            max_pool_flat = max_pool_out.squeeze(-1)
            avg_pool_flat = avg_pool_out.squeeze(-1)

            # Feed through MLP
            mlp_max = self.mlp(max_pool_flat)  # (B, C)
            mlp_avg = self.mlp(avg_pool_flat)  # (B, C)

            # Sum
            attn_logits = mlp_max + mlp_avg  # (B, C)

            # Channel attention map, unsqueeze to (B, C, 1) for broadcasting
            attn_map = torch.sigmoid(attn_logits).unsqueeze(-1)

        # Multiply by the original input (broadcasts attn_map if non-causal)
        output = attn_map * x

        # Mask out padded positions in the final output
        masked_fill_(output, mask, 0.0)

        return output

class SAM1D(nn.Module):
    """
    Spatial Attention Module for 1D sequences.
    - Takes (B, C, L) as input.
    - Produces a spatial attention map of shape (B, 1, L).
    - If causal, uses causal convolution.
    - Then multiplies it (elementwise) by the original input (while respecting the mask).
    """

    def __init__(self, kernel_size: int = 7, bias: bool = False, causal: bool = False):
        super(SAM1D, self).__init__()
        self.kernel_size_val = kernel_size  # Renamed to avoid conflict
        self.bias = bias
        self.causal = causal

        if self.causal:
            # For causal convolution, we pad (kernel_size - 1) on the left.
            # The Conv1d layer itself will have padding=0.
            self.causal_padding_amount = self.kernel_size_val - 1
            conv_padding = 0
        else:
            # Standard symmetric padding
            assert self.kernel_size_val % 2 == 1, "Kernel size must be odd for symmetric padding for SAM1D"
            conv_padding = self.kernel_size_val // 2

        self.conv = nn.Conv1d(
            in_channels=2,  # Max and Avg pooled features along channel dim
            out_channels=1,  # Output is a single attention map
            kernel_size=self.kernel_size_val,
            stride=1,
            padding=conv_padding,
            dilation=1,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
                                 This mask refers to the original sequence positions.
        Returns:
            torch.Tensor: Output tensor after spatial attention. Shape: (B, C, L).
        """
        # Max & Avg pooling across the channel dimension. Output shape: (B, 1, L)
        # These operations are inherently "spatially local" or "causal" in the sense that
        # pooling at time 't' only uses data from x[:,:,t].
        max_out_ch = x.max(dim=1, keepdim=True)[0]
        avg_out_ch = x.mean(dim=1, keepdim=True)

        # Apply mask to pooled features before concatenation and convolution.
        # This ensures that convolution doesn't operate on values from padded regions
        # if those regions influenced the channel pooling (e.g. mean).
        # Filling with 0.0 is a common choice.
        masked_fill_(max_out_ch, mask, 0.0)
        masked_fill_(avg_out_ch, mask, 0.0)

        # Concatenate along the channel dimension => shape (B, 2, L)
        concat_features = torch.cat((max_out_ch, avg_out_ch), dim=1)

        # Apply causal padding if needed before convolution
        if self.causal:
            # F.pad format for 1D (last dim): (pad_left, pad_right)
            concat_features = F.pad(concat_features, (self.causal_padding_amount, 0))
            # After padding, shape is (B, 2, L + causal_padding_amount)
            # Conv1d with padding=0 will then produce output of length L.

        # Convolution over the sequence dimension
        attn_logits = self.conv(concat_features)  # Expected shape: (B, 1, L)

        # Fill attention logits at masked positions with a large negative number.
        # This ensures that the sigmoid output for these positions is close to 0.
        # The 'mask' is (B, 1, L) and corresponds to original sequence length.
        masked_fill_(attn_logits, mask, -1e+4)

        # Apply sigmoid to get attention scores
        spatial_attn_map = torch.sigmoid(attn_logits)  # shape: (B, 1, L)

        # Ensure attention at padded positions is strictly zero after sigmoid.
        # This is somewhat redundant if logits were set to -1e9, but ensures exact zeros.
        masked_fill_(spatial_attn_map, mask, 0.0)

        # Multiply the spatial attention map with the original input tensor x.
        # The map (B, 1, L) broadcasts across the channel dimension of x (B, C, L).
        output = spatial_attn_map * x

        # Final explicit masking of the output tensor.
        # This ensures that any padded positions in the original x remain zeroed out.
        masked_fill_(output, mask, 0.0)

        return output

class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module for 1D sequences.
    - Applies Channel Attention (CAM1D).
    - Then applies Spatial Attention (SAM1D).
    - Adds the result to the original input as a residual connection.
    - Includes a `causal` option for both attention mechanisms.
    """

    def __init__(self, channels: int, reduction_ratio: int = 8, causal: bool = False, sam_kernel_size: int = 7):
        super(CBAM1D, self).__init__()
        self.causal = causal

        self.channel_attention = CAM1D(
            channels=channels,
            reduction_ratio=reduction_ratio,
            causal=self.causal
        )

        # Bias is typically False for SAM's convolution layer in many CBAM implementations.
        self.spatial_attention = SAM1D(
            kernel_size=sam_kernel_size,
            bias=False,  # Consistent with common practice
            causal=self.causal
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
                                 This mask indicates padded regions in the input 'x'.
        Returns:
            torch.Tensor: Output tensor after CBAM. Shape: (B, C, L).
        """
        # Apply Channel Attention
        # The mask is passed to ensure CAM handles padded sequences correctly.
        x_after_cam = self.channel_attention(x, mask)

        # Apply Spatial Attention
        # The mask is passed to ensure SAM handles padded sequences correctly.
        x_after_sam = self.spatial_attention(x_after_cam, mask)

        # Residual connection: Add the attention-modified features to the original input
        output = x_after_sam + x

        # Ensure that padded positions in the final output are zeroed out.
        # Although sub-modules also apply masking, this acts as a final safeguard,
        # especially important for the residual connection if 'x' had non-zero values
        # in its padded regions.
        masked_fill_(output, mask, 0.0)

        return output

# From model/attentions.py
class CausalConv1da(nn.Conv1d):
    """
    1-D *causal* convolution with optional weight-norm already applied.

    Padding rule
    ------------
    For kernel size *k* and dilation *d* the layer pads
    **d × (k − 1)** zeros on the *left* so that every output sample
    depends only on current and past inputs.

    Parameters
    ----------
    in_channels      : int
    out_channels     : int
    kernel_size      : int
    dilation         : int, default=1
    bias             : bool, default=True
    use_weight_norm  : bool, default=True
    **kwargs         : any extra Conv1d arguments (stride, groups, etc.)
    """
    def __init__(
        self,
        in_channels:   int,
        out_channels:  int,
        kernel_size:   int,
        dilation:      int  = 1,
        bias:          bool = True,
        use_weight_norm: bool = True,
        **kwargs
    ):
        # amount of left-padding to keep the op causal
        self.causal_padding = dilation * (kernel_size - 1)

        # Conv1d handles **no** padding; we pad manually in forward()
        kwargs.pop("padding", None)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
            **kwargs
        )

        # register weight-norm on this module's own .weight
        if use_weight_norm:
            nn.utils.weight_norm(self, name="weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal_padding:
            x = F.pad(x, (self.causal_padding, 0))  # (left, right)
        return super().forward(x)

class ResidualBlock1D(nn.Module):
    """
    Conv1D+Squeeze-Excite+LayerNorm residual block for sequence modeling with optional masking.
    Includes up/downsampling via strided convolutions.

    Accepts an optional x_mask (batch, 1, len) bool Tensor where padded elements are True.
    If provided, the mask is applied with .masked_fill() before each activation.
    The block also correctly transforms and returns the mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3, act="relu", causal=False, norm="layer", stride=1):
        super(ResidualBlock1D, self).__init__()

        self.stride = stride
        self.causal = causal
        if self.stride != 1 and self.causal:
            raise ValueError("Causal convolutions do not support striding.")

        assert norm in ["weight", "layer", "instance"], f"Unknown normalization type {norm}, must be 'weight', 'layer', or 'instance'"

        # Determine padding for strided convolutions
        if self.stride > 1: # Downsampling
            # Classic formula to approximate "same" padding for strided convs
            padding = (kernel_size - self.stride) // 2
        elif self.stride < 0: # Upsampling
            padding = (kernel_size - abs(self.stride)) // 2
        else: # stride == 1
            padding = "same" if not self.causal else 0


        # Layer definitions
        if self.causal:
            self.conv1 = CausalConv1da(in_channels, out_channels, kernel_size, dilation=dilation, use_weight_norm = norm == "weight")
            self.conv2 = CausalConv1da(out_channels, out_channels, kernel_size, dilation=dilation,  use_weight_norm = norm == "weight")
            self.cbam = None
        else:
            # Upsampling
            if self.stride < 0:
                s = abs(self.stride)
                self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=s, padding=padding)
            # Downsampling or no-op
            else:
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, stride=self.stride)

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding="same")
            self.cbam = CBAM1D(out_channels, causal=False) # CBAM1D is not causal by default

        # Normalization layers
        if norm == "weight":
            if not self.causal:
                self.conv1 = weight_norm(self.conv1)
                self.conv2 = weight_norm(self.conv2)
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        elif norm == "layer":
            self.norm1 = TransposeLayerNorm(out_channels)
            self.norm2 = TransposeLayerNorm(out_channels)
        else: # instance
            self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)

        # Activation and dropout
        if act == "taptx":
            self.relu = APTx(trainable=True)
        elif act == "aptx":
            self.relu = APTx()
        elif act == "relu":
            self.relu = nn.ReLU()
        else:
            raise RuntimeError(f"Unknown activation: {act}")
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or self.stride != 1:
            if self.stride > 1: # Downsample
                self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride)
            elif self.stride < 0: # Upsample
                self.residual = nn.Sequential(
                    nn.Upsample(scale_factor=abs(self.stride), mode='nearest'),
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                )
            else: # Stride is 1, but channels are different
                self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()


    def forward(self, x, x_mask=None):
        """
        Forward pass through the Residual Block 1D
        :param x: Input size (B, Cin, T)
        :param x_mask: Boolean sequence mask size (B, 1, T) where True=padded
        :return: Tuple of (Output size (B, Cout, T_out), updated_mask (B, 1, T_out))
        """
        # --- Adjust mask for up/downsampling ---
        if x_mask is not None:
            if self.stride > 1:
                # Downsample mask by taking the max over the stride window
                x_mask = F.max_pool1d(x_mask.float(), self.stride, self.stride).bool()
            elif self.stride < 0:
                # Upsample mask
                x_mask = F.interpolate(x_mask.float(), scale_factor=abs(self.stride), mode='nearest').bool()

        # --- Main path ---
        residual = self.residual(x)
        out = self.conv1(x)

        # Correct padding for ConvTranspose1d to match residual length
        if isinstance(self.conv1, nn.ConvTranspose1d):
            if out.shape[-1] != residual.shape[-1]:
                diff = out.shape[-1] - residual.shape[-1]
                out = out[..., diff // 2 : out.shape[-1] - (diff - diff // 2)]

        out = self.norm1(out)
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.cbam is not None:
            out = self.cbam(out, x_mask) # Pass mask to CBAM

        # --- Add residual and finalize ---
        out += residual
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)
        out = self.dropout(out)
        return out, x_mask
