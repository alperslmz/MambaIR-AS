import torch
from collections import OrderedDict

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .mambair_model import MambaIRModel


@MODEL_REGISTRY.register()
class MambaIRModel_L1Reg(MambaIRModel):
    """MambaIR_noCA model with L1 weight regularisation (M3).

    Extends the pixel loss with an L1 penalty on all trainable parameters::

        L_total = ||IHQ - ILQ||_1  +  lambda * ||w||_1

    The hyper-parameter ``lambda`` (``l1_lambda``) is read from
    ``opt['train']['l1_lambda']`` and defaults to 1e-5.

    Comparing validation PSNR against MambaIRModel (M2 / noCA baseline)
    shows whether L1 sparsity can compensate for the removed CA module.
    """

    def init_training_settings(self):
        super().init_training_settings()

        train_opt = self.opt['train']
        self.l1_lambda = float(train_opt.get('l1_lambda', 1e-5))

        logger = get_root_logger()
        logger.info(
            f'[MambaIRModel_L1Reg] L1 weight regularisation enabled  '
            f'(lambda = {self.l1_lambda:.2e})'
        )

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # ── pixel loss  (L_base = ||IHQ - ILQ||_1) ──────────────────────
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # ── perceptual loss (optional) ────────────────────────────────────
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # ── L1 weight penalty  (lambda * ||w||_1) ────────────────────────
        l1_reg = torch.tensor(0.0, device=self.output.device)
        for param in self.net_g.parameters():
            if param.requires_grad:
                l1_reg = l1_reg + param.abs().sum()
        l1_reg = self.l1_lambda * l1_reg
        l_total += l1_reg
        loss_dict['l1_reg'] = l1_reg

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
