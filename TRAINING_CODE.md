# ARTabPFN Training Code

Complete training codebase for autoregressive nanoTabPFN.

---

## configs/train_tabular_online.yaml

```yaml
device: cpu

data:
  batch_size: 512
  num_batches_per_epoch: 2000
  num_workers: 0
  d_list: [1]
  nc_list: [16, 32, 64, 128]
  num_buffer: 32
  num_target: 512
  normalize_y: true
  dtype: float32
  seed: 123

model:
  d_model: 64
  n_heads: 4
  n_layers: 12
  d_ff: 128
  num_features: 10
  buffer_size: 32
  num_components: 20

optimizer:
  lr: 5.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0

scheduler:
  use_scheduler: true
  warmup_steps: 2000
  total_steps: 20000

training:
  max_steps: 20000
  grad_clip: 1.0
  compile_model: true
  use_amp: true
  amp_dtype: bfloat16
  val_interval: 250
  precompile_masks: true
  precompile_shapes:
    - [16, 32, 512]
    - [32, 32, 512]
    - [64, 32, 512]
    - [128, 32, 512]

checkpoint:
  save_dir: checkpoints/artabpfn
  save_interval: 1000

logging:
  use_wandb: true
  project: artabpfn
  run_name: null
  tags: []
  log_interval: 50
```

---

## train.py

```python
"""Training loop for ARTabPFN with online tabular data generation."""

import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .model import ARTabPFN, create_dense_mask, create_row_mask
from .data import OnlineTabularDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class DataConfig:
    batch_size: int = 512
    num_batches_per_epoch: int = 2000
    num_workers: int = 0
    d_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    nc_list: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512, 1024])
    num_buffer: int = 32
    num_target: int = 512
    normalize_y: bool = True
    dtype: str = "float32"
    seed: int = 123


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 128
    num_features: int = 10
    buffer_size: int = 32
    num_components: int = 20


@dataclass
class TrainingConfig:
    max_steps: int = 20000
    grad_clip: float = 1.0
    compile_model: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    val_interval: int = 250
    precompile_masks: bool = True
    precompile_shapes: Optional[List[List[int]]] = None  # [[Nc, Nb, Nt], ...]


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    use_scheduler: bool = False
    warmup_steps: int = 2000
    total_steps: Optional[int] = None


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_interval: int = 1000


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project: str = "artabpfn"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_interval: int = 50


@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            optimizer=OptimizerConfig(**{
                k: tuple(v) if k == "betas" else v
                for k, v in d.get("optimizer", {}).items()
            }),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def precompile_masks(
    shapes: List[Tuple[int, int, int]], device: str
) -> Dict[Tuple[int, int, int], Tuple]:
    """Precompile masks for given (Nc, Nb, Nt) shapes."""
    mask_cache = {}
    print(f"Precompiling {len(shapes)} mask shapes...")

    for nc, nb, nt in shapes:
        key = (nc, nb, nt)
        num_rows = nc + nb + nt

        mask_features = create_dense_mask(seq_len=1, device=device)
        mask_rows = create_row_mask(
            num_rows=num_rows,
            context_len=nc,
            buffer_len=nb,
            device=device,
        )
        mask_cache[key] = (mask_features, mask_rows)
        print(f"  Precompiled: Nc={nc}, Nb={nb}, Nt={nt}")

    return mask_cache


def train(model: ARTabPFN, dataset: OnlineTabularDataset, config: Config) -> ARTabPFN:
    """Train ARTabPFN with online data generation."""
    device = config.device
    model = model.to(device)

    # Initialize wandb if enabled
    use_wandb = config.logging.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.logging.project,
            name=config.logging.run_name,
            config=asdict(config),
            tags=config.logging.tags,
        )

    if config.training.compile_model and device == "cuda":
        model = torch.compile(model, mode="default")

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )

    scheduler = None
    if config.scheduler.use_scheduler:
        total_steps = config.scheduler.total_steps or config.training.max_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            total_steps=total_steps,
        )

    model.train()

    loader = DataLoader(dataset, batch_size=None, shuffle=False)
    data_iter = iter(loader)

    # Precompile masks if configured
    if config.training.precompile_masks and config.training.precompile_shapes:
        shapes = [tuple(s) for s in config.training.precompile_shapes]
        mask_cache = precompile_masks(shapes, device)
    else:
        mask_cache: Dict[tuple, tuple] = {}

    if config.checkpoint.save_dir:
        os.makedirs(config.checkpoint.save_dir, exist_ok=True)

    total_loss = 0.0
    t0 = time.perf_counter()

    for step in range(config.training.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(device)

        Nc, Nb, Nt = batch.xc.size(1), batch.xb.size(1), batch.xt.size(1)
        cache_key = (Nc, Nb, Nt)

        if cache_key not in mask_cache:
            mask_features = create_dense_mask(seq_len=1, device=device)
            mask_rows = create_row_mask(
                num_rows=Nc + Nb + Nt,
                context_len=Nc,
                buffer_len=Nb,
                device=device,
            )
            mask_cache[cache_key] = (mask_features, mask_rows)
        else:
            mask_features, mask_rows = mask_cache[cache_key]

        with torch.autocast(
            device, dtype=torch.bfloat16, enabled=config.training.use_amp and device == "cuda"
        ):
            loss = model(
                x_context=batch.xc,
                y_context=batch.yc.squeeze(-1),
                x_buffer=batch.xb,
                y_buffer=batch.yb.squeeze(-1),
                x_target=batch.xt,
                mask_features=mask_features,
                mask_rows=mask_rows,
                y_target=batch.yt.squeeze(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if (step + 1) % config.logging.log_interval == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / config.logging.log_interval
            steps_per_sec = config.logging.log_interval / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step+1:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {steps_per_sec:.1f} it/s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/steps_per_sec": steps_per_sec,
                    },
                    step=step + 1,
                )

            total_loss = 0.0
            t0 = time.perf_counter()

        if (
            config.checkpoint.save_dir
            and config.checkpoint.save_interval > 0
            and (step + 1) % config.checkpoint.save_interval == 0
        ):
            ckpt_path = Path(config.checkpoint.save_dir) / f"step_{step+1}.pt"
            torch.save({"step": step + 1, "model": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Finalize wandb
    if use_wandb:
        wandb.summary["total_steps"] = config.training.max_steps
        wandb.summary["final_loss"] = avg_loss
        wandb.finish()

    return model


def load_config(config_path: str) -> Config:
    """Load YAML config file."""
    with open(config_path) as f:
        return Config.from_dict(yaml.safe_load(f) or {})


def main(config: Optional[Config] = None):
    if config is None:
        config = Config()

    print(f"Using device: {config.device}")

    dataset = OnlineTabularDataset(
        batch_size=config.data.batch_size,
        num_batches=config.data.num_batches_per_epoch,
        d_list=config.data.d_list,
        nc_list=config.data.nc_list,
        num_buffer=config.data.num_buffer,
        num_target=config.data.num_target,
        normalize_y=config.data.normalize_y,
        dtype=getattr(torch, config.data.dtype),
        device="cpu",
        seed=config.data.seed,
    )

    model = ARTabPFN(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        num_features=config.model.num_features,
        buffer_size=config.model.buffer_size,
        num_components=config.model.num_components,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train(model, dataset, config)
    print("Training complete!")

    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else Config()

    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    main(config)
```

---

## model/model.py

```python
"""
Autoregressive nanoTabPFN model.

Architecture combining:
- Context/Buffer/Target structure for autoregressive inference
- TabPFN's two-stage attention (feature + row)
- MixtureGaussian head for regression

Training uses flex_attention, inference uses Triton kernels with KV caching.
"""

import math
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from .attention import MultiheadAttention


class Embedder(nn.Module):
    """
    Embeds tabular data into D-dimensional space.

    Embeds (x, y) pairs, then adds marker embedding to distinguish
    context, buffer, and target sections.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.x_embed = nn.Linear(1, d_model)
        self.y_embed = nn.Linear(1, d_model)

        # Marker embeddings: 0=target, 1=context, 2=buffer
        self.marker_embed = nn.Embedding(3, d_model)
        self._marker_lookup = {"target": 0, "context": 1, "buffer": 2}

    def _get_marker(
        self, batch_size: int, marker_type: str, device: torch.device
    ) -> Tensor:
        idx = torch.full(
            (batch_size, 1),
            self._marker_lookup[marker_type],
            dtype=torch.long,
            device=device,
        )
        return self.marker_embed(idx)

    def embed_context(self, x: Tensor, y: Tensor) -> Tensor:
        """Embed context (training) data. x: [B, N, C], y: [B, N] or [B, N, 1]"""
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        x_emb = self.x_embed(x.unsqueeze(-1))  # [B, N, C, D]
        y_emb = self.y_embed(y)  # [B, N, 1, D]
        emb = x_emb.mean(dim=2) + y_emb.squeeze(2)  # [B, N, D]
        marker = self._get_marker(x.size(0), "context", x.device)
        return emb + marker

    def embed_buffer(self, x: Tensor, y: Tensor) -> Tensor:
        """Embed buffer (AR) data. AR token needs added afterwards."""
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        x_emb = self.x_embed(x.unsqueeze(-1))
        y_emb = self.y_embed(y)
        emb = x_emb.mean(dim=2) + y_emb.squeeze(2)
        marker = self._get_marker(x.size(0), "buffer", x.device)
        return emb + marker

    def embed_target(self, x: Tensor) -> Tensor:
        """Embed target (test) data. x: [B, T, C], no y values."""
        x_emb = self.x_embed(x.unsqueeze(-1))  # [B, T, C, D]
        emb = x_emb.mean(dim=2)  # [B, T, D]
        marker = self._get_marker(x.size(0), "target", x.device)
        return emb + marker


class TwoStageTransformerLayer(nn.Module):
    """
    Two-stage attention layer.

    Stage 1: Feature attention - dense self-attention across columns [B*R, C, D]
    Stage 2: Row attention - Context/Buffer/Target pattern [B*C, R, D]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_features = MultiheadAttention(d_model, n_heads)
        self.attn_rows = MultiheadAttention(d_model, n_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, mask_features: BlockMask, mask_rows: BlockMask
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: [B, R, C, D] input tensor
            mask_features: Dense mask for feature attention
            mask_rows: Context/Buffer/Target mask for row attention

        Returns:
            output: [B, R, C, D]
            kv_cache: (K, V) from row attention for inference caching
        """
        B, R, C, D = x.shape

        # Feature attention [B*R, C, D]
        x_feat = x.reshape(B * R, C, D)
        attn_out, _ = self.attn_features(x_feat, x_feat, x_feat, mask_features)
        x = self.norm1((attn_out + x_feat).reshape(B, R, C, D))

        # Row attention [B*C, R, D]
        x_row = x.permute(0, 2, 1, 3).reshape(B * C, R, D)
        attn_out, kv = self.attn_rows(x_row, x_row, x_row, mask_rows)
        x = self.norm2((attn_out + x_row).reshape(B, C, R, D).permute(0, 2, 1, 3))

        # FFN
        return self.norm3(x + self.ff(x)), kv


class TwoStageTransformer(nn.Module):
    """Stack of two-stage transformer layers."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [TwoStageTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, mask_features: BlockMask, mask_rows: BlockMask
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        kv_cache = []
        for layer in self.layers:
            x, kv = layer(x, mask_features, mask_rows)
            kv_cache.append(kv)
        return self.norm(x), kv_cache


class MixtureGaussianHead(nn.Module):
    """
    Mixture of Gaussians head for regression.

    Outputs K mixture components, each with mean, std, and weight.
    Assumes num_components >= 2.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dim_y: int = 1,
        num_components: int = 5,
        std_min: float = 1e-3,
    ):
        super().__init__()
        self.dim_y = dim_y
        self.num_components = num_components
        self.std_min = std_min

        # Head outputs: mean, std, weight for each component
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, num_components * dim_y * 3),
        )

        # Mixture initialization
        self.mean_bias = nn.Parameter(torch.linspace(-1.0, 1.0, num_components))
        delta = 1.0 / (num_components - 1)
        self.std_bias = nn.Parameter(
            torch.ones(num_components) * self._inv_softplus(delta)
        )
        self.weight_bias = nn.Parameter(torch.zeros(num_components))

    @staticmethod
    def _inv_softplus(y: float) -> float:
        return math.log(math.exp(y) - 1)

    def _parameterize(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert network output to mixture parameters (mean, std, weight, log_weight)."""
        B, T, _ = z.shape
        K, D = self.num_components, self.dim_y

        raw = self.head(z).view(B, T, K, D, 3)
        raw_mean, raw_std, raw_weight = raw.unbind(dim=-1)

        mean = raw_mean + self.mean_bias[None, None, :, None]
        std = (
            F.softplus(raw_std + self.std_bias[None, None, :, None]).clamp(max=2.0)
            + self.std_min
        )

        log_weight = F.log_softmax(
            raw_weight + self.weight_bias[None, None, :, None], dim=2
        )
        weight = log_weight.exp()

        return mean, std, weight, log_weight

    def forward(
        self,
        z: Tensor,
        y: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor]:
        """
        Args:
            z: [B, T, D] embeddings
            y: [B, T, dim_y] targets (optional, for training)
            loss_mask: [B, T] mask (optional)

        Returns:
            loss: scalar loss (if y provided)
            mean, std, weight: mixture parameters
        """
        mean, std, weight, log_weight = self._parameterize(z)

        loss = None
        if y is not None:
            ll = self._log_likelihood(y, mean, std, log_weight)
            if loss_mask is not None:
                ll = ll.mean(-1) * loss_mask
                denom = loss_mask.sum().clamp(min=1)
            else:
                denom = ll.numel()
            loss = -ll.sum() / denom

        return loss, mean, std, weight

    def _log_likelihood(
        self, y: Tensor, mean: Tensor, std: Tensor, log_weight: Tensor
    ) -> Tensor:
        y = y.unsqueeze(2)
        log_prob = -0.5 * (
            math.log(2 * math.pi) + 2 * std.log() + ((y - mean) / std) ** 2
        )
        log_prob = log_prob + log_weight
        return torch.logsumexp(log_prob, dim=2)

    def sample(self, z: Tensor, num_samples: int = 1) -> Tensor:
        """Sample from the mixture distribution."""
        mean, std, weight, _ = self._parameterize(z)
        B, T, K, D = mean.shape

        weight_flat = weight.permute(0, 1, 3, 2).reshape(B * T * D, K)
        indices = torch.multinomial(weight_flat, num_samples, replacement=True)

        mean_flat = mean.permute(0, 1, 3, 2).reshape(B * T * D, K)
        std_flat = std.permute(0, 1, 3, 2).reshape(B * T * D, K)

        sel_mean = torch.gather(mean_flat, 1, indices).view(B, T, D, num_samples)
        sel_std = torch.gather(std_flat, 1, indices).view(B, T, D, num_samples)

        samples = sel_mean + sel_std * torch.randn_like(sel_mean)
        return samples.permute(0, 1, 3, 2)  # [B, T, num_samples, D]

    def log_likelihood(self, z: Tensor, y: Tensor) -> Tensor:
        """Compute log-likelihood for evaluation."""
        mean, std, _, log_weight = self._parameterize(z)
        ll = self._log_likelihood(y, mean, std, log_weight)
        return ll.sum(dim=-1)


class ARTabPFN(nn.Module):
    """
    Autoregressive TabPFN.

    - Context/Buffer/Target structure for AR inference
    - TabPFN's two-stage (feature + row) attention
    - MixtureGaussian head for regression
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 256,
        num_features: int = 10,
        buffer_size: int = 8,
        num_components: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.buffer_size = buffer_size

        self.embedder = Embedder(d_model)
        self.backbone = TwoStageTransformer(d_model, n_heads, n_layers, d_ff)
        self.head = MixtureGaussianHead(
            d_model, d_ff, dim_y=1, num_components=num_components
        )

        # AR position tokens
        self.ar_tokens = nn.Parameter(torch.randn(buffer_size, d_model) * 0.02)

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_buffer: Tensor,
        y_buffer: Tensor,
        x_target: Tensor,
        mask_features: BlockMask,
        mask_rows: BlockMask,
        y_target: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass for training.

        Args:
            x_context: [B, Nc, C] context features
            y_context: [B, Nc] context labels
            x_buffer: [B, Nb, C] buffer features
            y_buffer: [B, Nb] buffer labels
            x_target: [B, Nt, C] target features
            y_target: [B, Nt] target labels (optional)
            mask_features: Pre-computed feature attention mask
            mask_rows: Pre-computed row attention mask

        Returns:
            loss: Training loss (if y_target provided)
            mean: Predicted means [B, Nt, K, 1]
        """

        # Embed context/buffer/targets
        ctx_emb = self.embedder.embed_context(x_context, y_context)  # [B, Nc, D]
        buf_emb = (
            self.embedder.embed_buffer(x_buffer, y_buffer)
            + self.ar_tokens[: x_buffer.size(1)]
        )
        tgt_emb = self.embedder.embed_target(x_target)  # [B, Nt, D]

        Nc, Nb, Nt = ctx_emb.size(1), buf_emb.size(1), tgt_emb.size(1)

        embeddings = torch.cat([ctx_emb, buf_emb, tgt_emb], dim=1)
        embeddings = embeddings.unsqueeze(2)  # [B, R, 1, D]

        # Forward through transformer
        z, _ = self.backbone(embeddings, mask_features, mask_rows)  # [B, R, 1, D]

        # Extract target embeddings
        z_target = z[:, Nc + Nb :, 0, :]  # [B, Nt, D]

        # Predict
        if y_target is not None and y_target.dim() == 2:
            y_target = y_target.unsqueeze(-1)
        loss, mean, std, weight = self.head(z_target, y_target)

        return loss
```

---

## model/attention.py

```python
"""
flex_attention-based attention for autoregressive-nanoTabPFN.

Mask patterns adapted for TabPFN's two-stage attention:
- Feature attention: dense self-attention across columns
- Row attention: Context/Buffer/Target pattern for efficient autoregressive inference

Sequence structure for row attention:
    [Context (train rows)] [Buffer (AR tokens)] [Target (test rows)]
"""

from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    or_masks,
    BlockMask,
)

# Compile flex_attention only on CUDA (CPU doesn't support compiled flex_attention)
if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


class MultiheadAttention(nn.Module):
    """Multi-head attention using flex_attention. Returns KV for caching."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, Sq, D = q.shape
        _, Skv, _ = k.shape

        # Project and reshape
        qh = self.q_proj(q).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(k).view(B, Skv, self.n_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(v).view(B, Skv, self.n_heads, self.head_dim).transpose(1, 2)

        # Forward
        out = flex_attention(qh, kh, vh, block_mask=block_mask)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, Sq, D)
        return self.out_proj(out), (kh, vh)


# Mask cache to avoid recompilation
_mask_cache = {}


def create_dense_mask(seq_len: int, device: str = "cuda") -> BlockMask:
    """Dense self-attention mask for feature dimension (all attend to all)."""
    key = ("dense", seq_len, device)
    if key not in _mask_cache:

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0  # Always true

        _mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
        )
    return _mask_cache[key]


def create_row_mask(
    num_rows: int,
    context_len: int,
    buffer_len: int,
    attending_chunks: int | None = None,
    device: str = "cuda",
) -> BlockMask:
    """
    Row attention mask with Context/Buffer/Target sections.

    Args:
        num_rows: Total number of rows (context + buffer + target)
        context_len: Number of context (training) rows
        buffer_len: Number of buffer (AR) tokens
        attending_chunks: Number of target chunks that attend to buffer.
            If None, defaults to half the target chunks (target_len // (2 * buffer_len)).
            Target length must be 2 * N * buffer_len for some integer N.
        device: Device for mask tensors

    Returns:
        BlockMask implementing the ACE attention pattern
    """
    target_len = num_rows - context_len - buffer_len

    if attending_chunks is None:
        attending_chunks = target_len // (2 * buffer_len)

    key = ("row", num_rows, context_len, buffer_len, attending_chunks, device)
    if key not in _mask_cache:
        target_start = context_len + buffer_len

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def prefix_mask(b, h, q_idx, kv_idx):
            """All tokens can attend to context."""
            return kv_idx < context_len

        def localized_causal_ctx_buf(b, h, q_idx, kv_idx):
            """Causal attention within context+buffer region."""
            ctx_buf_end = context_len + buffer_len
            q_in_region = q_idx < ctx_buf_end
            kv_in_region = kv_idx < ctx_buf_end
            return q_in_region & kv_in_region & causal_mask(b, h, q_idx, kv_idx)

        def chunked_target_buffer(b, h, q_idx, kv_idx):
            """First N chunks of targets attend causally to buffer."""
            buffer_start = context_len

            q_in_target = q_idx >= target_start
            kv_in_buffer = (kv_idx >= buffer_start) & (kv_idx < target_start)

            base_condition = q_in_target & kv_in_buffer

            target_offset = q_idx - target_start
            buffer_offset = kv_idx - buffer_start

            # First attending_chunks * buffer_len positions attend
            in_attending_region = target_offset < (attending_chunks * buffer_len)

            # Causal within chunk
            chunk_position = target_offset % buffer_len
            causal = buffer_offset <= chunk_position

            return base_condition & in_attending_region & causal

        # Combine all masks (no diagonal self-attention for targets, matching nanoTabPFN)
        final_mask_mod = or_masks(
            prefix_mask,
            localized_causal_ctx_buf,
            chunked_target_buffer,
        )
        final_mask_mod.__name__ = f"row_mask_{num_rows}_{context_len}_{buffer_len}"

        _mask_cache[key] = create_block_mask(
            final_mask_mod,
            B=None,
            H=None,
            Q_LEN=num_rows,
            KV_LEN=num_rows,
            device=device,
        )
    return _mask_cache[key]


def create_context_self_attention_mask(
    context_len: int, device: str = "cuda"
) -> BlockMask:
    """Dense self-attention mask for context encoding (used in inference)."""
    key = ("context_self", context_len, device)
    if key not in _mask_cache:

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= 0  # Always true (dense)

        _mask_cache[key] = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=context_len,
            KV_LEN=context_len,
            device=device,
        )
    return _mask_cache[key]


def clear_mask_cache():
    """Clear cached masks."""
    _mask_cache.clear()
```

---

## data/data.py

```python
"""Data structures for autoregressive-nanoTabPFN."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DataAttr:
    """Dataclass with context/buffer/target components."""

    xc: Optional[torch.Tensor] = None  # [B, Nc, D] context features
    yc: Optional[torch.Tensor] = None  # [B, Nc, 1] context targets
    xb: Optional[torch.Tensor] = None  # [B, Nb, D] buffer features
    yb: Optional[torch.Tensor] = None  # [B, Nb, 1] buffer targets
    xt: Optional[torch.Tensor] = None  # [B, Nt, D] target features
    yt: Optional[torch.Tensor] = None  # [B, Nt, 1] target targets

    def to(self, device):
        """Move all tensors to the specified device."""
        return DataAttr(
            xc=self.xc.to(device) if self.xc is not None else None,
            yc=self.yc.to(device) if self.yc is not None else None,
            xb=self.xb.to(device) if self.xb is not None else None,
            yb=self.yb.to(device) if self.yb is not None else None,
            xt=self.xt.to(device) if self.xt is not None else None,
            yt=self.yt.to(device) if self.yt is not None else None,
        )
```

---

## data/online_dataset.py

```python
"""Online dataset for generating tabular batches on-the-fly."""

from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .data import DataAttr
from .tabular_sampler import TabularSampler


class OnlineTabularDataset(IterableDataset):
    """
    Dataset that generates tabular batches online via TabularSampler.

    Yields pre-batched DataAttr objects for training.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_batches: int,
        d_list: List[int],
        nc_list: List[int],
        num_buffer: int,
        num_target: int,
        normalize_y: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.d_list = list(d_list)
        self.nc_list = list(nc_list)
        self.nb = num_buffer
        self.nt = num_target
        self.normalize_y = normalize_y
        self.dtype = dtype
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.sampler = TabularSampler(
            dim_x=self.d_list,
            dim_y=1,
            is_causal=True,
            num_causes=None,
            num_layers=4,
            hidden_dim=64,
            noise_std=0.01,
            sampling="mixed",
            normalize_y=self.normalize_y,
            device=self.device,
            dtype=self.dtype,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[DataAttr]:
        for _ in range(self.num_batches):
            nc = int(np.random.choice(self.nc_list))
            batch = self.sampler.generate_batch(
                batch_size=self.batch_size,
                num_context=nc,
                num_buffer=self.nb,
                num_target=self.nt,
            )
            yield batch
```

---

## data/tabular_sampler.py

```python
"""Tabular data sampler using MLP-SCM prior for synthetic regression tasks."""

from typing import Optional, Tuple, List

import numpy as np
import torch

from .data import DataAttr
from .mlp_scm import MLPSCM


class TabularSampler:
    """
    Generate tabular regression data using MLP-SCM prior.

    Creates synthetic regression tasks where X and y are derived from
    a randomly initialized MLP's intermediate representations.
    """

    def __init__(
        self,
        dim_x: int | List[int] = 10,
        dim_y: int = 1,
        # MLP-SCM parameters
        is_causal: bool = True,
        num_causes: Optional[int] = None,
        num_layers: int = 4,
        hidden_dim: int = 64,
        noise_std: float = 0.01,
        sampling: str = "mixed",
        # Normalization
        normalize_y: bool = True,
        # Device/dtype
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim_x_list = [dim_x] if isinstance(dim_x, int) else list(dim_x)
        self.dim_y = dim_y
        self.device = device
        self.dtype = dtype
        self.normalize_y = normalize_y

        # MLP-SCM parameters
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.sampling = sampling

    def _generate_function(
        self, num_samples: int, dim_x: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single regression function with fixed dimensionality."""

        # Randomize MLP-SCM hyperparameters slightly for diversity
        if self.num_causes is None:
            base_causes = max(1, dim_x // 2)
        else:
            base_causes = int(np.clip(self.num_causes, 1, dim_x))

        lo = max(1, base_causes - 2)
        hi = min(dim_x, base_causes + 3)
        actual_num_causes = int(np.random.randint(lo, hi + 1))

        actual_num_layers = np.random.randint(
            max(2, self.num_layers - 1), self.num_layers + 2
        )
        actual_hidden_dim = np.random.randint(
            max(16, self.hidden_dim - 16), self.hidden_dim + 32
        )

        model = MLPSCM(
            seq_len=num_samples,
            num_features=dim_x,
            num_outputs=self.dim_y,
            is_causal=self.is_causal,
            num_causes=actual_num_causes,
            y_is_effect=True,
            in_clique=False,
            sort_features=True,
            num_layers=actual_num_layers,
            hidden_dim=actual_hidden_dim,
            mlp_activations=torch.nn.Tanh,
            init_std=np.random.uniform(0.8, 2.0),
            block_wise_dropout=True,
            mlp_dropout_prob=np.random.uniform(0.05, 0.2),
            scale_init_std_by_dropout=True,
            sampling=self.sampling,
            pre_sample_cause_stats=True,
            noise_std=self.noise_std,
            pre_sample_noise_std=True,
            device=self.device,
        )

        with torch.no_grad():
            X, y = model()

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        return X.to(self.dtype), y.to(self.dtype)

    def generate_batch(
        self,
        batch_size: int,
        num_context: Optional[int | List[int]] = None,
        num_buffer: int = 8,
        num_target: int = 128,
        context_range: Optional[Tuple[int, int]] = None,
    ) -> DataAttr:
        """
        Generate a batch of tabular regression tasks.

        All tasks in batch have same feature dimension and context size
        (no padding needed).
        """
        # Choose dimensions ONCE for entire batch
        dim_x = int(np.random.choice(self.dim_x_list))

        # Choose context size
        if num_context is None:
            if context_range is None:
                context_range = (32, 256)
            nc = np.random.randint(context_range[0], context_range[1] + 1)
        elif isinstance(num_context, int):
            nc = num_context
        else:
            nc = int(np.random.choice(num_context))

        nb = num_buffer
        nt = num_target
        total_samples = nc + nb + nt

        xc_list, yc_list = [], []
        xb_list, yb_list = [], []
        xt_list, yt_list = [], []

        for _ in range(batch_size):
            X, y = self._generate_function(total_samples, dim_x)

            # Shuffle
            perm = torch.randperm(total_samples)
            X = X[perm]
            y = y[perm]

            # Split
            xc, yc = X[:nc], y[:nc]
            xb, yb = X[nc : nc + nb], y[nc : nc + nb]
            xt, yt = X[nc + nb :], y[nc + nb :]

            # Normalize y using context statistics
            if self.normalize_y:
                y_mean = yc.mean()
                y_std = yc.std().clamp(min=1e-6)
                yc = (yc - y_mean) / y_std
                yb = (yb - y_mean) / y_std
                yt = (yt - y_mean) / y_std

            xc_list.append(xc)
            yc_list.append(yc)
            xb_list.append(xb)
            yb_list.append(yb)
            xt_list.append(xt)
            yt_list.append(yt)

        # Stack into batches
        return DataAttr(
            xc=torch.stack(xc_list),
            yc=torch.stack(yc_list),
            xb=(
                torch.stack(xb_list)
                if nb > 0
                else torch.zeros(
                    batch_size, 0, dim_x, device=self.device, dtype=self.dtype
                )
            ),
            yb=(
                torch.stack(yb_list)
                if nb > 0
                else torch.zeros(
                    batch_size, 0, self.dim_y, device=self.device, dtype=self.dtype
                )
            ),
            xt=torch.stack(xt_list),
            yt=torch.stack(yt_list),
        )
```

---

## data/mlp_scm.py

```python
"""
MLP-based Structural Causal Model (SCM) for synthetic tabular data generation.

Based on TabICL: https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to inputs."""

    def __init__(self, std: float | torch.Tensor = 0.01):
        super().__init__()
        if isinstance(std, torch.Tensor):
            self.register_buffer("std", std)
        else:
            self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class XSampler:
    """Samples initial cause variables for the SCM."""

    def __init__(
        self,
        seq_len: int,
        num_causes: int,
        pre_stats: bool = False,
        sampling: str = "normal",
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.num_causes = num_causes
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()
        else:
            self.means = None
            self.stds = None

    def _pre_stats(self):
        """Pre-sample mean and std for normal distributions."""
        means = np.random.normal(0, 1, self.num_causes)
        stds = np.abs(np.random.normal(0, 1, self.num_causes) * means)
        self.means = (
            torch.tensor(means, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )
        self.stds = (
            torch.tensor(stds, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .repeat(self.seq_len, 1)
        )

    def sample(self) -> torch.Tensor:
        """Sample cause variables. Returns (seq_len, num_causes)."""
        if self.sampling == "normal":
            return self._sample_normal_all()
        elif self.sampling == "uniform":
            return torch.rand(self.seq_len, self.num_causes, device=self.device)
        elif self.sampling == "mixed":
            return self._sample_mixed()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

    def _sample_normal_all(self) -> torch.Tensor:
        """Sample all features from normal distribution."""
        if self.means is not None:
            return torch.normal(self.means, self.stds.abs()).float()
        else:
            return torch.normal(
                0.0, 1.0, (self.seq_len, self.num_causes), device=self.device
            ).float()

    def _sample_normal(self, n: int) -> torch.Tensor:
        """Sample single feature from normal distribution."""
        if self.means is not None:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def _sample_multinomial(self) -> torch.Tensor:
        """Sample from weighted multinomial distribution."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def _sample_zipf(self) -> torch.Tensor:
        """Sample from Zipf distribution (centered, not scaled)."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def _sample_mixed(self) -> torch.Tensor:
        """Sample using probability-based mixture of distributions."""
        X = []
        zipf_p = random.random() * 0.66
        multi_p = random.random() * 0.66
        normal_p = random.random() * 0.66

        for n in range(self.num_causes):
            if random.random() > normal_p:
                x = self._sample_normal(n)
            elif random.random() > multi_p:
                x = self._sample_multinomial()
            elif random.random() > zipf_p:
                x = self._sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)

        return torch.stack(X, -1)


class MLPSCM(nn.Module):
    """
    Generates synthetic tabular datasets using an MLP-based Structural Causal Model.

    Creates regression data where features X and targets y are derived from
    intermediate representations of a randomly initialized MLP applied to
    sampled "cause" variables.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int | None = None,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 4,
        hidden_dim: int = 64,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "mixed",
        pre_sample_cause_stats: bool = True,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.num_layers = max(num_layers, 2)
        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        # Set num_causes
        if num_causes is None:
            self.num_causes = max(1, num_features // 2)
        else:
            self.num_causes = num_causes

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(
                self.hidden_dim, self.num_outputs + 2 * self.num_features
            )
        else:
            self.num_causes = self.num_features

        # Input sampler
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build MLP layers
        layers = [nn.Linear(self.num_causes, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self._make_layer_block())
        if not self.is_causal:
            layers.append(self._make_layer_block(is_output=True))

        self.layers = nn.Sequential(*layers).to(device)
        self._init_parameters()

    def _make_layer_block(self, is_output: bool = False) -> nn.Sequential:
        """Create activation -> linear -> noise block."""
        out_dim = self.num_outputs if is_output else self.hidden_dim

        if self.pre_sample_noise_std:
            noise_std = (
                torch.abs(torch.randn(1, out_dim, device=self.device) * self.noise_std)
                + 1e-6
            )
        else:
            noise_std = self.noise_std

        return nn.Sequential(
            self.mlp_activations(),
            nn.Linear(self.hidden_dim, out_dim),
            GaussianNoise(noise_std),
        )

    def _init_parameters(self):
        """Initialize MLP parameters."""
        for i, (name, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self._init_block_dropout(param)
            else:
                self._init_normal(param, i)

    def _init_block_dropout(self, param: torch.Tensor):
        """Block-wise sparse initialization."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        if block_size[0] == 0 or block_size[1] == 0:
            nn.init.normal_(param, std=self.init_std)
            return

        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        std = self.init_std / (
            math.sqrt(keep_prob) if self.scale_init_std_by_dropout else 1
        )

        for b in range(n_blocks):
            slices = tuple(slice(d * b, d * (b + 1)) for d in block_size)
            nn.init.normal_(param[slices], std=std)

    def _init_normal(self, param: torch.Tensor, idx: int):
        """Standard normal initialization with dropout."""
        if param.dim() != 2:
            return

        dropout = self.mlp_dropout_prob if idx > 0 else 0
        dropout = min(dropout, 0.99)
        std = self.init_std / (
            math.sqrt(1 - dropout) if self.scale_init_std_by_dropout else 1
        )

        nn.init.normal_(param, std=std)
        if dropout > 0:
            mask = torch.bernoulli(torch.full_like(param, 1 - dropout))
            param.data *= mask

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic (X, y) data."""
        causes = self.xsampler.sample()  # [seq_len, num_causes]

        # Forward through MLP, collecting intermediate outputs
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))

        # Skip first two (causes and first linear-only layer)
        outputs = outputs[2:]

        X, y = self._extract_xy(causes, outputs)

        # Handle NaNs
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.zeros_like(y)

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        return X, y

    def _extract_xy(
        self, causes: torch.Tensor, outputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features X and targets y from MLP outputs."""
        if not self.is_causal:
            return causes, outputs[-1]

        # Causal mode: sample X and y from intermediate representations
        outputs_flat = torch.cat(outputs, dim=-1)
        total_dim = outputs_flat.shape[-1]

        if self.in_clique:
            # Block sampling with random permutation within block
            max_start = total_dim - self.num_outputs - self.num_features
            start = random.randint(0, max(0, max_start))
            perm = start + torch.randperm(
                self.num_outputs + self.num_features, device=self.device
            )
        else:
            # Random sampling
            perm = torch.randperm(total_dim - 1, device=self.device)

        idx_X = perm[self.num_outputs : self.num_outputs + self.num_features]

        if self.y_is_effect:
            idx_y = list(range(-self.num_outputs, 0))
        else:
            idx_y = perm[: self.num_outputs]

        if self.sort_features:
            idx_X, _ = torch.sort(idx_X)

        X = outputs_flat[:, idx_X]
        y = outputs_flat[:, idx_y]

        return X, y
```
