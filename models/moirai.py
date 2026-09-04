"""Moirai 1.1 (Salesforce) wrapper.

Moirai is a unified training approach for universal time series forecasting.
Moirai 1.1-R-large (300M parameters) was pre-trained on 27 billion time points
spanning IT metrics, finance, and transport, making it a strong zero-shot baseline
for network operations data.

HuggingFace model: Salesforce/moirai-1.1-R-large
Install: pip install uni2ts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import TSFMBase


class MoiraiModel(TSFMBase):

    name = "moirai"
    # Moirai 1.1 requires HuggingFace authentication (gated model).
    # Run `huggingface-cli login` before loading this model.
    # Without auth: falls back to moirai-1.0 automatically.
    _hf_model_id = "Salesforce/moirai-1.1-R-large"

    def __init__(
        self,
        context_len: int,
        prediction_len: int,
        device: Optional[str] = None,
        model_size: str = "large",
        version: str = "1.1",
    ) -> None:
        super().__init__(context_len, prediction_len, device)
        self.model_size = model_size
        self._hf_model_id = f"Salesforce/moirai-{version}-R-{model_size}"
        self._model = None
        self._pipeline = None

    def load(self) -> None:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError:
            raise ImportError(
                "Install Moirai: pip install uni2ts\n"
                "See https://github.com/SalesforceAIResearch/uni2ts"
            )
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # _locate is private Hydra API; location varies across versions
        try:
            from hydra._internal.utils import _locate
        except ImportError:
            try:
                from hydra._internal.instantiate._internal.utils import _locate
            except ImportError:
                from omegaconf import OmegaConf as _OC
                from importlib import import_module as _imp
                def _locate(path: str):  # type: ignore[misc]
                    parts = path.rsplit(".", 1)
                    mod = _imp(parts[0])
                    return getattr(mod, parts[1])

        cfg_path = hf_hub_download(self._hf_model_id, "config.json")
        wts_path = hf_hub_download(self._hf_model_id, "model.safetensors")
        cfg = json.loads(open(cfg_path).read())

        # Resolve distribution output from Hydra-style config
        distr_cls = _locate(cfg["distr_output"]["_target_"])
        components = [
            _locate(c["_target_"])(**{k: v for k, v in c.items() if k != "_target_"})
            for c in cfg["distr_output"]["components"]
        ]
        distr_output = distr_cls(components=components)

        module = MoiraiModule(
            distr_output=distr_output,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            patch_sizes=cfg["patch_sizes"],
            max_seq_len=cfg["max_seq_len"],
            attn_dropout_p=cfg["attn_dropout_p"],
            dropout_p=cfg["dropout_p"],
        )
        module.load_state_dict(load_file(wts_path))

        self._model = MoiraiForecast(
            module=module,
            prediction_length=self.prediction_len,
            context_length=self.context_len,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device)
        self._loaded = True

    def _get_samples(self, context: np.ndarray) -> torch.Tensor:
        """Run model inference and return raw samples (1, n_samples, pred_len, 1).

        With patch_size='auto', MoiraiForecast.past_length = context_len + pred_len.
        Internally, _get_distr() uses past_target[..., -context_length:, :] for the
        actual forecast and _val_loss() uses past_target[..., :context_length, :] only
        for patch-size selection.  The context data must therefore go into the LAST
        context_len positions of the past_target buffer, not the first.
        """
        ctx = context.flatten()[-self.context_len :]
        past_len = self._model.past_length  # context_len + pred_len when patch_size='auto'
        ctx_len = len(ctx)
        full     = torch.zeros(1, past_len, 1, dtype=torch.float32, device=self.device)
        observed = torch.zeros(1, past_len, 1, dtype=torch.bool,    device=self.device)
        is_pad   = torch.zeros(1, past_len,    dtype=torch.bool,    device=self.device)
        # Fill the LAST ctx_len positions — these are what _get_distr uses.
        full[0, past_len - ctx_len:, 0] = torch.tensor(ctx, dtype=torch.float32)
        observed[0, past_len - ctx_len:, 0] = True
        with torch.no_grad():
            samples = self._model(
                past_target=full,
                past_observed_target=observed,
                past_is_pad=is_pad,
            )
        return samples  # (1, n_samples, pred_len, 1)

    def forecast_many(self, contexts: list[np.ndarray], batch_size: int = 64) -> list[np.ndarray]:
        """Batched inference — avoids per-sample Python overhead on A30."""
        assert self._loaded
        past_len = self._model.past_length
        results = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            B = len(batch)
            full     = torch.zeros(B, past_len, 1, dtype=torch.float32, device=self.device)
            observed = torch.zeros(B, past_len, 1, dtype=torch.bool,    device=self.device)
            is_pad   = torch.zeros(B, past_len,    dtype=torch.bool,    device=self.device)
            for j, ctx in enumerate(batch):
                c = np.asarray(ctx).flatten()[-self.context_len :]
                # Fill the LAST len(c) positions — same layout as _get_samples.
                full[j, past_len - len(c):, 0] = torch.tensor(c, dtype=torch.float32)
                observed[j, past_len - len(c):, 0] = True
            with torch.no_grad():
                samples = self._model(
                    past_target=full,
                    past_observed_target=observed,
                    past_is_pad=is_pad,
                )
            # samples: (B, n_samples, pred_len, 1) → median over samples → (B, pred_len)
            medians = samples.median(dim=1).values.squeeze(-1).cpu().numpy()
            results.extend(list(medians))
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results

    def forecast(self, context: np.ndarray) -> np.ndarray:
        assert self._loaded, "call load() first"
        samples = self._get_samples(context)
        # samples: (1, n_samples, pred_len, 1) → median over samples axis
        return samples.median(dim=1).values.squeeze().cpu().numpy()

    def forecast_upper(self, context: np.ndarray, q: float = 0.9) -> np.ndarray:
        """Return q-th quantile of the predictive distribution (per timestep)."""
        assert self._loaded, "call load() first"
        samples = self._get_samples(context)
        # quantile over n_samples dim → (1, pred_len, 1) → squeeze → (pred_len,)
        return samples.quantile(q, dim=1).squeeze().cpu().numpy()

    def forecast_samples_many(
        self, contexts: list[np.ndarray], batch_size: int = 64
    ) -> list[np.ndarray]:
        """Batched sample inference. Returns list of (n_samples, pred_len) arrays."""
        assert self._loaded
        past_len = self._model.past_length
        results = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            B = len(batch)
            full     = torch.zeros(B, past_len, 1, dtype=torch.float32, device=self.device)
            observed = torch.zeros(B, past_len, 1, dtype=torch.bool,    device=self.device)
            is_pad   = torch.zeros(B, past_len,    dtype=torch.bool,    device=self.device)
            for j, ctx in enumerate(batch):
                c = np.asarray(ctx).flatten()[-self.context_len :]
                full[j, past_len - len(c):, 0] = torch.tensor(c, dtype=torch.float32)
                observed[j, past_len - len(c):, 0] = True
            with torch.no_grad():
                samples = self._model(
                    past_target=full,
                    past_observed_target=observed,
                    past_is_pad=is_pad,
                )
            # samples: (B, n_samples, pred_len, 1) → squeeze last dim → (B, n_samples, pred_len)
            arr = samples.squeeze(-1).cpu().numpy()
            results.extend([arr[k] for k in range(B)])
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 5,
        weights_dir: str | Path = "models/weights",
    ) -> dict[str, list[float]]:
        assert self._loaded, "call load() first"
        Path(weights_dir).mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(self._model.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        no_improve = 0

        for epoch in range(epochs):
            self._model.train()
            train_loss = _run_epoch(self._model, train_loader, optimizer, criterion, self.device)
            self._model.eval()
            val_loss = _run_epoch(self._model, val_loader, None, criterion, self.device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"  [{self.name}] epoch {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(self._model.state_dict(), Path(weights_dir) / f"{self.name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [{self.name}] early stopping at epoch {epoch+1}")
                    break

        self._model.load_state_dict(
            torch.load(Path(weights_dir) / f"{self.name}_best.pt", map_location=self.device)
        )
        return history

    def save_weights(self, path: str | Path) -> None:
        assert self._loaded
        torch.save(self._model.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        assert self._loaded
        self._model.load_state_dict(torch.load(path, map_location=self.device))


def _run_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """Train using MoiraiModule directly to preserve gradient flow.

    MoiraiForecast.forward() wraps inference under torch.no_grad(), so samples
    have no grad_fn. Instead we call model.module (MoiraiModule) which returns
    a Distribution whose .mean is computed from encoder parameters and retains
    a valid grad_fn for backpropagation.

    MoiraiModule.forward() expects inputs in "packed patch" format:
      target: (B, n_patches, max_patch_size)  — NOT raw timesteps
      patch_size: (B, n_patches)              — per-patch, not per-sequence
    This mirrors the internal _convert() logic of MoiraiForecast.
    """
    from einops import rearrange
    import torch.nn.functional as _F

    inner = model.module
    max_ps = max(int(p) for p in inner.patch_sizes)
    # Use the largest patch size for efficiency (fewer tokens → smaller attention)
    ps = max_ps

    total, count = 0.0, 0
    for context, target in loader:
        B, ctx_len, n_links = context.shape
        pred_len = target.shape[1]

        # Sample one random link per window to keep batch size manageable.
        link_idx = torch.randint(0, n_links, (B,))
        ctx = context[torch.arange(B), :, link_idx].to(device)   # (B, ctx_len)
        tgt = target[torch.arange(B), :, link_idx].to(device)    # (B, pred_len)

        ctx_3d = ctx.unsqueeze(-1)   # (B, ctx_len, 1)
        tgt_3d = tgt.unsqueeze(-1)   # (B, pred_len, 1)

        # Observed masks: all context observed, all future observed (training)
        obs_ctx = torch.ones(B, ctx_len, 1, dtype=torch.bool, device=device)
        obs_tgt = torch.ones(B, pred_len, 1, dtype=torch.bool, device=device)

        # Left-pad context / right-pad future so lengths are divisible by ps.
        # Matches MoiraiForecast._patched_seq_pad behavior.
        ctx_pad = (-ctx_len) % ps
        tgt_pad = (-pred_len) % ps

        ctx_padded = _F.pad(ctx_3d,  (0, 0, ctx_pad, 0))       # (B, ctx_len+ctx_pad, 1)
        tgt_padded = _F.pad(tgt_3d,  (0, 0, 0, tgt_pad))       # (B, pred_len+tgt_pad, 1)
        obs_ctx_p  = _F.pad(obs_ctx.float(), (0, 0, ctx_pad, 0)).bool()
        obs_tgt_p  = _F.pad(obs_tgt.float(), (0, 0, 0, tgt_pad)).bool()

        n_ctx_tok = (ctx_len + ctx_pad) // ps
        n_tgt_tok = (pred_len + tgt_pad) // ps
        n_total   = n_ctx_tok + n_tgt_tok

        # Reshape into patch tokens: (B, n_tokens, ps)
        # einops pattern "b (s p) d -> b (d s) p" with d=1 gives "b s p" → (B, n_tok, ps)
        ctx_patches = rearrange(ctx_padded, "b (s p) d -> b (d s) p", p=ps)
        tgt_patches = rearrange(tgt_padded, "b (s p) d -> b (d s) p", p=ps)
        obs_ctx_tok = rearrange(obs_ctx_p,  "b (s p) d -> b (d s) p", p=ps)
        obs_tgt_tok = rearrange(obs_tgt_p,  "b (s p) d -> b (d s) p", p=ps)

        # Pad each token to max_patch_size (models with multiple patch sizes share weights)
        ctx_patches = _F.pad(ctx_patches, (0, max_ps - ps))   # (B, n_ctx_tok, max_ps)
        tgt_patches = _F.pad(tgt_patches, (0, max_ps - ps))   # (B, n_tgt_tok, max_ps)
        obs_ctx_tok = _F.pad(obs_ctx_tok, (0, max_ps - ps))
        obs_tgt_tok = _F.pad(obs_tgt_tok, (0, max_ps - ps))

        target_pk = torch.cat([ctx_patches, tgt_patches], dim=1)        # (B, n_total, max_ps)
        obs_pk    = torch.cat([obs_ctx_tok, obs_tgt_tok], dim=1).bool() # (B, n_total, max_ps)

        # sample_id=1 marks valid tokens; sample_id=0 is reserved for padding.
        # With no sequence packing all tokens belong to the single series (id=1).
        sample_id = torch.ones(B, n_total,  dtype=torch.long,  device=device)
        variate_id = torch.zeros(B, n_total, dtype=torch.long,  device=device)
        time_id    = torch.arange(n_total, device=device).unsqueeze(0).expand(B, -1).contiguous()
        pred_mask  = torch.cat([
            torch.zeros(B, n_ctx_tok, dtype=torch.bool, device=device),
            torch.ones( B, n_tgt_tok, dtype=torch.bool, device=device),
        ], dim=1)
        ps_tensor  = torch.full((B, n_total), ps, dtype=torch.long, device=device)

        distr = inner(
            target=target_pk,
            observed_mask=obs_pk,
            sample_id=sample_id,
            variate_id=variate_id,
            prediction_mask=pred_mask,
            patch_size=ps_tensor,
            time_id=time_id,
        )

        # distr.mean: (B, n_total, max_ps) — extract future tokens, then flatten
        # Only the first ps values per patch are real predictions (rest are max_ps padding)
        pred_mean = distr.mean[:, n_ctx_tok:, :ps]              # (B, n_tgt_tok, ps)
        pred_flat = pred_mean.reshape(B, -1)[:, :pred_len]      # (B, pred_len)

        loss = criterion(pred_flat, tgt)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)
