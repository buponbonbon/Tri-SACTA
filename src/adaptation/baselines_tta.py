"""
Baseline Test-Time Adaptation Methods for Multimodal Sentiment Analysis.

Implements three TTA baselines for comparison with Tri-SACTA:

1. **TENT**: Entropy minimization that updates only LayerNorm/BatchNorm
   affine parameters [Wang et al., ICLR 2021].

2. **SAR**: Sharpness-Aware and Reliable TTA with entropy filtering and
   periodic parameter restoration [Niu et al., ICLR 2023].

3. **MultimodalTTABaseline**: A generic multimodal TTA method inspired by
   READ/AV-CTTA that adapts all modalities equally using entropy
   minimization + simple cross-modal consistency, without spectral
   shift detection.

All baselines expose the same ``adapt_step`` interface as :class:`TriSACTA`
and use AMP (mixed-precision) for efficiency.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

try:
    from src.models.fusion_backbone import MultimodalFusionModel
except ImportError:
    from ..models.fusion_backbone import MultimodalFusionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

MODALITY_NAMES = ["visual", "audio", "text", "prosody"]


def _collect_norm_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect affine parameters from LayerNorm and BatchNorm layers.

    Args:
        model: PyTorch module to scan.

    Returns:
        List of ``nn.Parameter`` objects (weight and bias) from
        normalization layers.
    """
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def _configure_norm_training(model: nn.Module) -> None:
    """Set only normalization layers to train mode; freeze everything else.

    After calling this, only LayerNorm/BN affine parameters will receive
    gradients, and all other parameters are frozen with ``requires_grad=False``.

    Args:
        model: The model to configure.
    """
    model.eval()
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Enable training + gradients for norm layers
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-sample entropy from logits.

    Args:
        logits: (B, C) unnormalized class scores.

    Returns:
        (B,) entropy per sample.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


# ===========================================================================
# TENT: Test-Time Entropy Minimization
# ===========================================================================


class TENT:
    """TENT: Fully Test-Time Adaptation by Entropy Minimization.

    Updates only the affine parameters of LayerNorm/BatchNorm layers
    to minimize the entropy of model predictions at test time.

    Reference:
        Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
        Minimization", ICLR 2021.

    Attributes:
        model: Multimodal fusion model.
        config: Experiment configuration dict.
        optimizer: AdamW on norm parameters only.
        scaler: AMP GradScaler.
        source_state: Saved source checkpoint for reset.
    """

    def __init__(
        self,
        model: MultimodalFusionModel,
        config: dict,
    ) -> None:
        """Initialize TENT.

        Args:
            model: Pre-trained multimodal fusion model.
            config: Experiment config dict (uses config['tta']['lr']).
        """
        self.model = model
        self.config = config
        self.lr = config["tta"]["lr"]

        # Save source checkpoint
        self.source_state = copy.deepcopy(model.state_dict())

        # Configure: only norm params trainable
        _configure_norm_training(self.model)
        norm_params = _collect_norm_params(self.model)

        if not norm_params:
            logger.warning("TENT: No normalization layer parameters found.")

        self.optimizer = torch.optim.AdamW(
            norm_params if norm_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0

        logger.info("TENT initialized — lr=%.1e, %d norm params.",
                     self.lr, sum(p.numel() for p in norm_params))

    def adapt_step(
        self,
        batch: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Perform one TENT adaptation step.

        Minimizes the entropy of the fused model predictions by updating
        only normalization layer affine parameters.

        Args:
            batch: Dict with modality name → (B, D_m) feature tensor.
            modality_mask: Dict with modality name → (B,) availability mask.

        Returns:
            Dict with 'logits', 'loss', 'adapted' (always True).
        """
        self._step_count += 1
        self.optimizer.zero_grad(set_to_none=True)

        with autocast():
            out = self.model(
                features=batch,
                modality_mask=modality_mask,
            )
            logits = out["logits"]
            loss = _entropy(logits).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "logits": logits.detach(),
            "loss": loss.item(),
            "loss_ent": loss.item(),
            "loss_align": 0.0,
            "loss_reg": 0.0,
            "adapted": True,
            "shift_scores": {},
            "reliability_weights": {},
            "anchor_set": [],
            "adaptable_set": [],
            "corrupted_set": [],
        }

    def reset(self) -> None:
        """Reset model to source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_state))
        _configure_norm_training(self.model)
        norm_params = _collect_norm_params(self.model)
        self.optimizer = torch.optim.AdamW(
            norm_params if norm_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0
        logger.info("TENT reset to source checkpoint.")


# ===========================================================================
# SAR: Sharpness-Aware and Reliable TTA
# ===========================================================================


class SAR:
    """SAR: Sharpness-Aware and Reliable entropy minimization.

    Extends TENT with two key mechanisms:
    1. **Entropy filtering**: Only samples with entropy below a threshold
       contribute to the loss, discarding unreliable high-entropy predictions.
    2. **Periodic parameter restoration**: Periodically restores a fraction
       of the parameters toward the source checkpoint to prevent drift.

    Reference:
        Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild
        World", ICLR 2023.

    Attributes:
        model: Multimodal fusion model.
        config: Experiment configuration.
        entropy_threshold: Entropy filtering threshold (ln(C) * 0.4).
        restore_ratio: Fraction of params restored each step.
        restore_interval: Steps between parameter restorations.
    """

    # SAR-specific defaults
    DEFAULT_ENTROPY_RATIO = 0.4   # e_margin = 0.4 * ln(C)
    DEFAULT_RESTORE_RATIO = 0.01  # Restore 1% of param distance
    DEFAULT_RESTORE_INTERVAL = 10  # Restore every 10 steps

    def __init__(
        self,
        model: MultimodalFusionModel,
        config: dict,
    ) -> None:
        """Initialize SAR.

        Args:
            model: Pre-trained multimodal fusion model.
            config: Experiment config dict. Uses:
                - config['tta']['lr']: Learning rate.
                - config['model']['num_classes']: For entropy threshold.
                - config['tta'].get('sar_entropy_ratio', 0.4).
                - config['tta'].get('sar_restore_ratio', 0.01).
                - config['tta'].get('sar_restore_interval', 10).
        """
        self.model = model
        self.config = config
        self.lr = config["tta"]["lr"]
        num_classes = config["model"]["num_classes"]

        tta_cfg = config["tta"]
        entropy_ratio = tta_cfg.get("sar_entropy_ratio", self.DEFAULT_ENTROPY_RATIO)
        self.entropy_threshold = entropy_ratio * torch.log(
            torch.tensor(float(num_classes))
        ).item()
        self.restore_ratio = tta_cfg.get("sar_restore_ratio", self.DEFAULT_RESTORE_RATIO)
        self.restore_interval = tta_cfg.get("sar_restore_interval", self.DEFAULT_RESTORE_INTERVAL)

        # Save source checkpoint
        self.source_state = copy.deepcopy(model.state_dict())

        # Configure: only norm params trainable
        _configure_norm_training(self.model)
        norm_params = _collect_norm_params(self.model)

        self.optimizer = torch.optim.AdamW(
            norm_params if norm_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0

        logger.info(
            "SAR initialized — lr=%.1e, entropy_thresh=%.3f, "
            "restore_ratio=%.3f, restore_interval=%d.",
            self.lr, self.entropy_threshold, self.restore_ratio,
            self.restore_interval,
        )

    def adapt_step(
        self,
        batch: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Perform one SAR adaptation step.

        Pipeline:
        1. Forward pass to get logits.
        2. Compute per-sample entropy and filter to reliable samples.
        3. If reliable samples exist, compute mean entropy loss and step.
        4. Periodically restore parameters toward the source checkpoint.

        Args:
            batch: Dict with modality name → (B, D_m) feature tensor.
            modality_mask: Dict with modality name → (B,) availability mask.

        Returns:
            Dict with 'logits', 'loss', 'adapted', 'num_reliable'.
        """
        self._step_count += 1
        self.optimizer.zero_grad(set_to_none=True)

        adapted = False
        loss_val = 0.0
        num_reliable = 0

        with autocast():
            out = self.model(
                features=batch,
                modality_mask=modality_mask,
            )
            logits = out["logits"]  # (B, C)
            ent = _entropy(logits)  # (B,)

            # Entropy filtering: keep only reliable (low-entropy) samples
            reliable_mask = (ent < self.entropy_threshold)  # (B,)
            num_reliable = reliable_mask.sum().item()

            if num_reliable > 0:
                # Filtered entropy loss
                loss = ent[reliable_mask].mean()
                adapted = True
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        if adapted:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_val = loss.item()

        # Periodic parameter restoration toward source
        if self._step_count % self.restore_interval == 0:
            self._restore_parameters()

        return {
            "logits": logits.detach(),
            "loss": loss_val,
            "loss_ent": loss_val,
            "loss_align": 0.0,
            "loss_reg": 0.0,
            "adapted": adapted,
            "num_reliable": int(num_reliable),
            "shift_scores": {},
            "reliability_weights": {},
            "anchor_set": [],
            "adaptable_set": [],
            "corrupted_set": [],
        }

    def _restore_parameters(self) -> None:
        """Restore a fraction of adaptable parameters toward source.

        Each adaptable parameter θ is updated as:
            θ ← θ + restore_ratio * (θ_0 - θ)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.source_state:
                    source_param = self.source_state[name].to(param.device)
                    param.add_(
                        (source_param - param) * self.restore_ratio
                    )

    def reset(self) -> None:
        """Reset model to source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_state))
        _configure_norm_training(self.model)
        norm_params = _collect_norm_params(self.model)
        self.optimizer = torch.optim.AdamW(
            norm_params if norm_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0
        logger.info("SAR reset to source checkpoint.")


# ===========================================================================
# MultimodalTTABaseline: READ / AV-CTTA style
# ===========================================================================


class MultimodalTTABaseline:
    """Generic multimodal TTA baseline (READ/AV-CTTA style).

    A simple multimodal test-time adaptation approach that:
    - Adapts **all modalities equally** (no spectral shift detection).
    - Uses entropy minimization on fused predictions.
    - Adds a simple cross-modal consistency loss encouraging all modality
      heads to agree (pairwise KL divergence).
    - Updates all adaptable model parameters (not just norm layers).

    This serves as a multimodal-aware baseline that lacks the core
    innovations of Tri-SACTA: spectral shift detection, the anchor/
    adaptable/corrupted classification, and selective adaptation.

    Attributes:
        model: Multimodal fusion model.
        config: Experiment configuration.
        lambda_consist: Weight for cross-modal consistency loss.
    """

    # Default consistency weight
    DEFAULT_LAMBDA_CONSIST = 0.1

    def __init__(
        self,
        model: MultimodalFusionModel,
        config: dict,
    ) -> None:
        """Initialize MultimodalTTABaseline.

        Args:
            model: Pre-trained multimodal fusion model.
            config: Experiment config dict. Uses:
                - config['tta']['lr']: Learning rate.
                - config['tta'].get('lambda_consist', 0.1): Consistency weight.
                - config['tta'].get('lambda_reg', 0.01): Regularization weight.
        """
        self.model = model
        self.config = config
        tta_cfg = config["tta"]
        self.lr = tta_cfg["lr"]
        self.lambda_consist = tta_cfg.get("lambda_consist", self.DEFAULT_LAMBDA_CONSIST)
        self.lambda_reg = tta_cfg.get("lambda_reg", 0.01)

        # Save source checkpoint
        self.source_state = copy.deepcopy(model.state_dict())

        # Set up optimizer on ALL adaptable parameters
        adaptable_params = model.get_adaptable_params()
        if not adaptable_params:
            logger.warning("MultimodalTTABaseline: No adaptable parameters.")

        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0

        logger.info(
            "MultimodalTTABaseline initialized — lr=%.1e, λ_consist=%.3f, "
            "λ_reg=%.4f.",
            self.lr, self.lambda_consist, self.lambda_reg,
        )

    def adapt_step(
        self,
        batch: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Perform one multimodal TTA adaptation step.

        Loss = L_ent(fused) + λ_consist * L_consist + λ_reg * L_reg

        where L_consist is the mean pairwise KL divergence between all
        modality predictions, and L_reg is L2 distance from the source.

        Args:
            batch: Dict with modality name → (B, D_m) feature tensor.
            modality_mask: Dict with modality name → (B,) availability mask.

        Returns:
            Dict with 'logits', 'loss', 'loss_ent', 'loss_consist',
            'loss_reg', 'adapted'.
        """
        self._step_count += 1
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast():
            out = self.model(
                features=batch,
                modality_mask=modality_mask,
            )
            logits = out["logits"]  # (B, C)
            per_modality_logits = out.get("per_modality_logits", {})

            # --- Entropy loss on fused predictions ---
            loss_ent = _entropy(logits).mean()

            # --- Cross-modal consistency loss ---
            loss_consist = self._consistency_loss(per_modality_logits)

            # --- Parameter regularization ---
            loss_reg = self._regularization_loss()

            # --- Total loss ---
            loss_total = (
                loss_ent
                + self.lambda_consist * loss_consist
                + self.lambda_reg * loss_reg
            )

        self.scaler.scale(loss_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "logits": logits.detach(),
            "loss": loss_total.item(),
            "loss_ent": loss_ent.item(),
            "loss_align": loss_consist.item(),  # Mapped to 'loss_align' for interface compat
            "loss_consist": loss_consist.item(),
            "loss_reg": loss_reg.item(),
            "adapted": True,
            "shift_scores": {},
            "reliability_weights": {},
            "anchor_set": [],
            "adaptable_set": list(MODALITY_NAMES),  # Adapts all equally
            "corrupted_set": [],
        }

    def _consistency_loss(
        self, per_modality_logits: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute pairwise cross-modal consistency via KL divergence.

        L_consist = (2 / M(M-1)) Σ_{i<j} [KL(p_i || p_j) + KL(p_j || p_i)] / 2

        This is a symmetric version using the average of both KL directions.

        Args:
            per_modality_logits: Dict modality name → (B, C) logits.

        Returns:
            Scalar consistency loss tensor.
        """
        modalities = [m for m in MODALITY_NAMES if m in per_modality_logits]
        if len(modalities) < 2:
            return torch.tensor(
                0.0, device=next(self.model.parameters()).device
            )

        log_probs = {
            m: F.log_softmax(per_modality_logits[m], dim=-1) for m in modalities
        }
        probs = {
            m: F.softmax(per_modality_logits[m], dim=-1) for m in modalities
        }

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        num_pairs = 0
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mi, mj = modalities[i], modalities[j]
                # Symmetric KL: average of both directions
                kl_ij = F.kl_div(
                    log_probs[mi], probs[mj],
                    reduction="batchmean", log_target=False,
                )
                kl_ji = F.kl_div(
                    log_probs[mj], probs[mi],
                    reduction="batchmean", log_target=False,
                )
                loss = loss + 0.5 * (kl_ij + kl_ji)
                num_pairs += 1

        return loss / max(num_pairs, 1)

    def _regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization: ||θ − θ_0||² over adaptable params.

        Returns:
            Scalar regularization loss tensor.
        """
        device = next(self.model.parameters()).device
        reg_loss = torch.tensor(0.0, device=device)

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.source_state:
                source_param = self.source_state[name].to(device)
                reg_loss = reg_loss + (param - source_param).pow(2).sum()

        return reg_loss

    def reset(self) -> None:
        """Reset model to source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_state))
        adaptable_params = self.model.get_adaptable_params()
        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self._step_count = 0
        logger.info("MultimodalTTABaseline reset to source checkpoint.")
