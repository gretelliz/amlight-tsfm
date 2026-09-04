"""Lightweight LoRA implementation for any nn.Module.

Low-Rank Adaptation (LoRA) injects trainable low-rank matrices A and B alongside
each frozen base weight W. The adapted output is W·x + (B·A·x) * (alpha/r), where
r is the rank and alpha is the scaling factor. Only A and B are trained.

Hu et al. 2022 — https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Attention projection names across common transformer architectures.
# T5: q, k, v, o  |  HF standard: q_proj, k_proj, v_proj, out_proj
# Some models use query/key/value or query_proj/key_proj/value_proj.
_ATTN_NAMES: frozenset[str] = frozenset({
    "q", "k", "v", "o",
    "q_proj", "k_proj", "v_proj", "out_proj",
    "query", "key", "value",
    "query_proj", "key_proj", "value_proj",
    "Wq", "Wk", "Wv", "Wo",
    # nn.MultiheadAttention / custom decoders
    "in_proj", "c_attn", "c_proj",
    # TimesFM PatchedTimeSeriesDecoder variants
    "linear_q", "linear_k", "linear_v",
    "qkv", "proj",
})

# Fallback: any nn.Linear whose direct parent class name contains one of these
# keywords is also eligible for LoRA injection. Catches architectures whose
# projection layers use non-standard attribute names (e.g. TimesFM).
_ATTN_CLASS_KEYWORDS: tuple[str, ...] = ("attention", "attn")


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a trainable low-rank adapter.

    Base weight is frozen; only lora_A and lora_B are updated during training.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear
        for p in linear.parameters():
            p.requires_grad = False
        d_in   = linear.in_features
        d_out  = linear.out_features
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, d_in, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=device))
        self.scale  = alpha / r

    @property
    def weight(self) -> torch.Tensor:
        # T5 FFN checks `self.wo.weight` for dtype casting; expose it transparently.
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast LoRA matrices to input dtype (e.g. BFloat16 for Chronos/T5).
        # Parameters stay float32 for optimizer precision; cast only at runtime.
        lora_out = (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale
        return self.linear(x) + lora_out


def inject_lora(
    module: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: frozenset[str] = _ATTN_NAMES,
) -> int:
    """Replace matching nn.Linear children with LoRALinear in-place.

    Traverses the module tree recursively. A Linear layer is replaced when its
    attribute name is in target_names OR when its direct parent module's class
    name contains an attention keyword (catches non-standard naming like TimesFM).

    Caller is responsible for freezing all base parameters before calling this
    function (for p in model.parameters(): p.requires_grad = False).

    Returns the number of LoRA layers injected.
    """
    injected = 0
    parent_cls = type(module).__name__.lower()
    parent_is_attn = any(kw in parent_cls for kw in _ATTN_CLASS_KEYWORDS)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (name in target_names or parent_is_attn):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            injected += 1
        else:
            injected += inject_lora(child, r=r, alpha=alpha, target_names=target_names)
    return injected


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA parameters (lora_A, lora_B) — small portable file."""
    return {k: v for k, v in module.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def load_lora_weights(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: str = "cpu",
) -> None:
    """Load LoRA parameters into a module that already has LoRA layers injected.

    Merges the LoRA state dict on top of the module's current state, then loads
    with strict=True. Call inject_lora() before calling this function.
    """
    current = dict(module.state_dict())
    for k, v in state_dict.items():
        current[k] = v.to(device)
    module.load_state_dict(current)


def get_inner_module(model_name: str, model) -> nn.Module:
    """Return the inner nn.Module suitable for LoRA injection per model type.

    Each wrapper exposes its trainable core differently:
      chronos  — T5ForConditionalGeneration via _get_t5_model()
      moirai   — MoiraiModule (transformer backbone) via _model.module
      timesfm  — PatchedTimeSeriesDecoder via _get_torch_module()
      patchtst — PatchTSTModule directly via _model
    """
    if model_name == "chronos":
        return model._get_t5_model()
    elif model_name == "moirai":
        return model._model.module
    elif model_name == "timesfm":
        return model._get_torch_module()
    elif model_name == "patchtst":
        return model._model
    else:
        raise ValueError(f"Unknown model for LoRA: {model_name}")
