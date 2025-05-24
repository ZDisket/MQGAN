import torch
import torch.nn as nn
import torch.nn.functional as F

class LSGANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0, decay=0.99, use_lecam=True):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.decay = decay
        self.use_lecam = use_lecam

        # we'll compute per‑element MSE and reduce manually
        self.criterion = nn.MSELoss(reduction='none')

        # EMA buffers
        self.register_buffer("ema_real", torch.tensor(0.0))
        self.register_buffer("ema_fake", torch.tensor(0.0))
        self.ema_initialized = False

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        # pred/target: (..., N) same shape. mask: same shape, bool or float, or None.
        err = self.criterion(pred, target)
        if mask is not None:
            # ensure float mask
            m = mask.float()
            err = err * m
            valid = m.sum()
            if valid.item() > 0:
                return err.sum() / valid
            else:
                return torch.tensor(0., device=pred.device)
        else:
            # default mean over all elements
            return err.mean()

    def update_ema(self, real_out: torch.Tensor, fake_out: torch.Tensor,
                   real_mask: torch.Tensor=None, fake_mask: torch.Tensor=None):
        # compute means
        if real_mask is not None:
            m = real_mask.float()
            real_mean = (real_out * m).sum() / m.sum().clamp(min=1)
        else:
            real_mean = real_out.mean()

        if fake_mask is not None:
            m = fake_mask.float()
            fake_mean = (fake_out * m).sum() / m.sum().clamp(min=1)
        else:
            fake_mean = fake_out.mean()

        if not self.ema_initialized:
            self.ema_real.copy_(real_mean.detach())
            self.ema_fake.copy_(fake_mean.detach())
            self.ema_initialized = True
        else:
            self.ema_real.mul_(self.decay).add_((1 - self.decay) * real_mean.detach())
            self.ema_fake.mul_(self.decay).add_((1 - self.decay) * fake_mean.detach())

    def lecam_loss(self, real_out: torch.Tensor, fake_out: torch.Tensor,
                   real_mask: torch.Tensor=None, fake_mask: torch.Tensor=None):

        # Ensure EMA tensors are on the same device as the outputs
        device = real_out.device
        self.ema_real = self.ema_real.to(device)
        self.ema_fake = self.ema_fake.to(device)

        ema_r = self.ema_real.detach()
        ema_f = self.ema_fake.detach()
        # LeCam on real
        if real_mask is not None:
            diff_r = (real_out - ema_f).clamp(min=0) * real_mask.float()
            term_r = diff_r.pow(2).sum() / real_mask.float().sum().clamp(min=1)
        else:
            term_r = ((real_out - ema_f).clamp(min=0) ** 2).mean()
        # LeCam on fake
        if fake_mask is not None:
            diff_f = (ema_r - fake_out).clamp(min=0) * fake_mask.float()
            term_f = diff_f.pow(2).sum() / fake_mask.float().sum().clamp(min=1)
        else:
            term_f = ((ema_r - fake_out).clamp(min=0) ** 2).mean()
        return term_r + term_f

    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        real_mask: torch.Tensor=None,
        fake_mask: torch.Tensor=None
    ):
        """
        real_output, fake_output: (B, 1, L)
        real_mask, fake_mask: same shape bool mask, or None.
        """
        # prepare targets
        real_tgt = torch.full_like(real_output, self.real_label)
        fake_tgt = torch.full_like(fake_output, self.fake_label)

        # LSGAN losses
        real_loss = self._masked_mse(real_output, real_tgt, real_mask)
        fake_loss = self._masked_mse(fake_output, fake_tgt, fake_mask)
        loss = 0.5 * (real_loss + fake_loss)

        if self.use_lecam:
            self.update_ema(real_output, fake_output, real_mask, fake_mask)
            loss = loss + self.lecam_loss(real_output, fake_output, real_mask, fake_mask)

        return loss

    def generator_loss(
        self,
        fake_output: torch.Tensor,
        fake_mask: torch.Tensor=None
    ):
        real_tgt = torch.full_like(fake_output, self.real_label)
        return self._masked_mse(fake_output, real_tgt, fake_mask)
