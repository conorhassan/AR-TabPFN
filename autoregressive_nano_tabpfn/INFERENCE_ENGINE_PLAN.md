# Plan: ARTabPFNPredictor

Created: 2025-12-14
Status: PENDING APPROVAL

## Summary

Implement a naive InferenceEngine for ARTabPFN following the `InferenceEngine2` pattern from ace-v1_5. This provides autoregressive sampling and log-likelihood evaluation with KV caching. Triton optimization deferred to later phase.

---

## Background: What Problem Are We Solving?

**Training**: We see all data at once (context + buffer + targets). One forward pass.

**Inference**: We predict targets one at a time. Each prediction becomes context for the next.

**The challenge**: How do we efficiently reuse computation from previous predictions?

**Solution**: KV caching. Store the Key/Value vectors from previous tokens so we don't recompute them.

---

## Key Insight #1: Two-Stage Attention (and why we only cache row attention)

ARTabPFN has two-stage attention per layer:
```
Stage 1: Feature attention [B*R, C, D] - across columns
Stage 2: Row attention [B*C, R, D] - across rows
```

**After embedding, C=1** (features are averaged via `x_emb.mean(dim=2)`).

### What happens in feature attention when C=1?

```
Input: [B*R, 1, D]  (sequence length = 1)

Q, K, V = linear_proj(input)     # These projections STILL MATTER
score = softmax(Q @ K^T / √d)    # 1×1 matrix → always [[1.0]]
attn_out = score @ V             # = 1.0 × V = V

output = input + W_out(V)        # Residual connection
       = input + W_out(W_v(input))  # Two linear transforms
```

**The attention pattern is trivial** (each position attends only to itself with weight 1.0).

**But the linear projections (W_v, W_out) still transform the representation!**

### Why we don't cache feature attention:

| Aspect | Feature Attention | Row Attention |
|--------|------------------|---------------|
| Sequence length | C=1 | R (grows with context+buffer) |
| Cross-token dependency | None (each row independent) | Yes (targets attend to context) |
| Need to cache KV? | **No** | **Yes** |

- Feature attention: Each row only attends to itself → nothing to reuse
- Row attention: Targets attend to context → cache context KV to avoid recomputation

### Summary:
- **Run feature attention** for correctness (linear transforms matter)
- **Cache row attention KV** for efficiency (cross-token dependencies)

---

## Key Insight #2: The "Commit" Trick

**Problem**: In training, targets attend to context + buffer, but NOT to other targets.

**At inference**: When we decode target t2, we've already decoded t1. How do we prevent t2 from attending to t1's KV?

**Solution**: Don't commit target KVs to the cache!

```
transformer_decode(embedding, commit):
    - Write KV to cache (temporary)
    - Run attention
    - Only increment seq_len by `commit` positions
    - Non-committed KVs get overwritten next call
```

---

## Key Insight #3: Batch [buffer, target] Together

Instead of two separate forward passes:
```
❌ Slow: decode(buffer) → decode(target)
```

We batch them:
```
✅ Fast: decode([buffer, target]) with commit=1
```

- Buffer KV gets committed (future targets can attend to it)
- Target KV is temporary (gets overwritten next iteration)

---

## Concrete Example: Predicting 3 Targets

**Setup**: Context has 10 rows (Nc=10), we predict 3 targets.

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 0: Prefill context                                     │
│                                                             │
│ Cache: [ctx0, ctx1, ctx2, ..., ctx9, _, _, _, _]            │
│         └──────── Nc=10 ────────┘                           │
│ seq_len = 10                                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Decode first target                                 │
│                                                             │
│ Input: [t1]  (N=1, commit=0)                                │
│                                                             │
│ Cache: [ctx0, ..., ctx9, t1, _, _, _]                       │
│                          ↑                                  │
│                     temporary                               │
│ seq_len = 10 (unchanged, t1 not committed)                  │
│                                                             │
│ → Sample y1 from t1's output                                │
│ → Create buffer embedding: b1 = embed(x1, y1) + ar_token[0] │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Decode [b1, t2] together                            │
│                                                             │
│ Input: [b1, t2]  (N=2, commit=1)                            │
│                                                             │
│ Cache: [ctx0, ..., ctx9, b1, t2, _, _]                      │
│                          ↑   ↑                              │
│                     committed temporary                     │
│ seq_len = 11 (b1 committed)                                 │
│                                                             │
│ Attention patterns:                                         │
│   b1 attends to: [context]                                  │
│   t2 attends to: [context, b1]                              │
│                                                             │
│ → Sample y2 from t2's output                                │
│ → Create b2 = embed(x2, y2) + ar_token[1]                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Decode [b2, t3] together                            │
│                                                             │
│ Input: [b2, t3]  (N=2, commit=1)                            │
│                                                             │
│ Cache: [ctx0, ..., ctx9, b1, b2, t3, _]                     │
│                              ↑   ↑                          │
│                         committed temporary                 │
│ seq_len = 12 (b2 committed)                                 │
│                                                             │
│ Attention patterns:                                         │
│   b2 attends to: [context, b1]                              │
│   t3 attends to: [context, b1, b2]                          │
│                                                             │
│ → Sample y3 from t3's output                                │
└─────────────────────────────────────────────────────────────┘

RESULT: Targets only ever attended to context + buffers, never to other targets ✓
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE FLOW                           │
└─────────────────────────────────────────────────────────────┘

    x_context, y_context              x_target (all targets)
           │                                 │
           ▼                                 │
    ┌─────────────┐                          │
    │ embed_context│                         │
    └──────┬──────┘                          │
           │                                 │
           ▼                                 │
    ┌─────────────┐                          │
    │ prefill_kv  │ ← Cache context KV       │
    └──────┬──────┘                          │
           │                                 │
           ▼                                 │
    ╔═════════════════════════════════════╗  │
    ║     FOR t = 1, 2, ..., num_targets  ║◄─┘
    ╠═════════════════════════════════════╣
    ║                                     ║
    ║  x_t = x_target[:, t, :]            ║
    ║            │                        ║
    ║            ▼                        ║
    ║  ┌──────────────────┐               ║
    ║  │ embed_target(x_t)│               ║
    ║  └────────┬─────────┘               ║
    ║           │                         ║
    ║           ▼                         ║
    ║  ┌──────────────────────────────┐   ║
    ║  │ if t > 1:                    │   ║
    ║  │   embed_buffer(x_{t-1}, y_{t-1})│ ║
    ║  │   concat [b_{t-1}, t_t]      │   ║
    ║  └────────┬─────────────────────┘   ║
    ║           │                         ║
    ║           ▼                         ║
    ║  ┌──────────────────────────────┐   ║
    ║  │ transformer_decode(emb,      │   ║
    ║  │   commit = 0 if t==1 else 1) │   ║
    ║  └────────┬─────────────────────┘   ║
    ║           │                         ║
    ║           ▼                         ║
    ║  ┌─────────────┐                    ║
    ║  │ head.sample │ → y_t              ║
    ║  └─────────────┘                    ║
    ║                                     ║
    ╚═════════════════════════════════════╝
                    │
                    ▼
            y_predictions [B, num_targets]
```

## Scope

**In scope**:
- `InferenceEngine` class in `model/inference.py`
- `sample_sequence()` - batch autoregressive sampling
- `evaluate_joint_loglikelihood()` - log-likelihood evaluation
- KV cache management (init, prefill, decode)
- Integration with existing `MixtureGaussianHead`

**Out of scope** (for now):
- Triton kernel optimization (Phase 2)
- torch.compile optimization (later)
- Multi-GPU support

## Phases

### Phase 1: Core Structure

**Goal**: Basic class with cache management.

**File**: `model/inference.py`

```python
class InferenceEngine:
    """
    Efficient inference for ARTabPFN with KV caching.

    Usage:
        engine = InferenceEngine.from_trained_model(model)
        predictions = engine.sample_sequence(x_ctx, y_ctx, x_target)
    """

    def __init__(self, embedder, backbone, head, ar_tokens):
        self.embedder = embedder
        self.backbone = backbone
        self.head = head
        self.ar_tokens = ar_tokens

        # Will be set during inference
        self.seq_len = 0  # Current committed sequence length

    @classmethod
    def from_trained_model(cls, model: ARTabPFN) -> "InferenceEngine":
        return cls(model.embedder, model.backbone, model.head, model.ar_tokens)

    def init_kv_cache(self, B: int, max_seq: int, device, dtype):
        """Allocate empty KV cache for each layer."""
        for layer in self.backbone.layers:
            H = layer.attn_rows.n_heads
            Dh = layer.attn_rows.head_dim
            layer.k_cache = torch.zeros(B, H, max_seq, Dh, device=device, dtype=dtype)
            layer.v_cache = torch.zeros_like(layer.k_cache)
        self.seq_len = 0

    def clear_cache(self):
        """Reset cache state."""
        self.seq_len = 0
```

**Verification**: `engine = InferenceEngine.from_trained_model(model)` works.

---

### Phase 2: Context Prefill

**Goal**: Encode context and fill KV cache.

```python
def prefill_context(self, x_context: Tensor, y_context: Tensor):
    """
    Encode context and populate KV cache.

    After this call:
        - Cache positions [0, Nc) contain context KV
        - seq_len = Nc
    """
    # Embed context
    ctx_emb = self.embedder.embed_context(x_context, y_context)  # [B, Nc, D]

    # Run through transformer, extract and cache KV from row attention
    x = ctx_emb.unsqueeze(2)  # [B, Nc, 1, D] for two-stage attention

    for layer in self.backbone.layers:
        # Feature attention (trivial with C=1, but keep for correctness)
        B, R, C, D = x.shape
        x_feat = x.reshape(B * R, C, D)
        x_feat, _ = layer.attn_features(x_feat, x_feat, x_feat, dense_mask)
        x = layer.norm1(x_feat.reshape(B, R, C, D) + x)

        # Row attention - this is where we cache KV
        x_row = x.squeeze(2)  # [B, Nc, D]

        # Compute K, V and cache them
        k = layer.attn_rows.k_proj(x_row)  # [B, Nc, D]
        v = layer.attn_rows.v_proj(x_row)
        # Reshape to [B, H, Nc, Dh] and store
        layer.k_cache[:, :, :Nc, :] = reshape_for_cache(k)
        layer.v_cache[:, :, :Nc, :] = reshape_for_cache(v)

        # Complete the attention and FFN
        ...

    self.seq_len = Nc
```

**Verification**: Cache contains correct values matching forward pass.

---

### Phase 3: Transformer Decode (The Core)

**Goal**: Incremental decode with commit control.

```python
def transformer_decode(self, embedding: Tensor, commit: int) -> Tensor:
    """
    Process new embeddings using cached KV.

    Args:
        embedding: [B, N, D] - new token embeddings
        commit: How many positions to commit to cache
                (N-1 for [buffer, target], 0 for [target] only)

    Returns:
        [B, N, D] - transformed embeddings
    """
    B, N, D = embedding.shape
    x = embedding.unsqueeze(2)  # [B, N, 1, D]

    for layer in self.backbone.layers:
        # Feature attention (trivial)
        ...

        # Row attention with KV cache
        x_row = x.squeeze(2)  # [B, N, D]

        # Compute Q, K, V for new tokens
        q = layer.attn_rows.q_proj(x_row)
        k_new = layer.attn_rows.k_proj(x_row)
        v_new = layer.attn_rows.v_proj(x_row)

        # Write to cache (temporary until committed)
        cache_start = self.seq_len
        layer.k_cache[:, :, cache_start:cache_start+N, :] = reshape_for_cache(k_new)
        layer.v_cache[:, :, cache_start:cache_start+N, :] = reshape_for_cache(v_new)

        # Attention over [cached + new]
        total_len = self.seq_len + N
        k_full = layer.k_cache[:, :, :total_len, :]
        v_full = layer.v_cache[:, :, :total_len, :]

        # Causal attention (new tokens can attend to all previous + themselves)
        attn_out = causal_attention(q, k_full, v_full)

        # Residual + FFN
        ...

    # Commit only the first `commit` positions
    self.seq_len += commit

    return self.backbone.norm(x.squeeze(2))
```

**Key insight**: The `commit` parameter controls what stays in cache:
- `commit=0`: Target KV is temporary (overwritten next call)
- `commit=1`: Buffer KV is kept, target KV is temporary

---

### Phase 4: Autoregressive Decode

**Goal**: Single prediction step.

```python
def autoregressive_decode(
    self,
    x_target: Tensor,           # [B, 1, num_features]
    prev_x: Tensor = None,      # [B, 1, num_features] from previous step
    prev_y: Tensor = None,      # [B, 1] prediction from previous step
    ar_idx: int = 0,            # AR token index
) -> Tensor:
    """
    Decode one target, optionally with previous prediction as buffer.

    Returns:
        y_pred: [B, 1] sampled prediction
    """
    # Embed current target
    target_emb = self.embedder.embed_target(x_target)  # [B, 1, D]

    if prev_x is not None and prev_y is not None:
        # Create buffer embedding from previous prediction
        buffer_emb = self.embedder.embed_buffer(prev_x, prev_y)  # [B, 1, D]
        buffer_emb = buffer_emb + self.ar_tokens[ar_idx]

        # Batch [buffer, target] together
        embedding = torch.cat([buffer_emb, target_emb], dim=1)  # [B, 2, D]
        commit = 1  # Commit buffer, not target
    else:
        # First target, no buffer
        embedding = target_emb  # [B, 1, D]
        commit = 0  # Don't commit target

    # Decode
    z = self.transformer_decode(embedding, commit=commit)

    # Sample from last position (the target)
    y_pred = self.head.sample(z[:, -1:, :])  # [B, 1, num_samples, 1]

    return y_pred.squeeze(-1).squeeze(-1)  # [B, 1]
```

---

### Phase 5: High-Level API

**Goal**: User-friendly interface.

```python
def sample_sequence(
    self,
    x_context: Tensor,  # [B, Nc, num_features]
    y_context: Tensor,  # [B, Nc]
    x_target: Tensor,   # [B, Nt, num_features]
) -> Tensor:
    """
    Predict all targets autoregressively.

    Returns:
        y_pred: [B, Nt] predictions
    """
    B, Nc, _ = x_context.shape
    Nt = x_target.shape[1]
    device, dtype = x_context.device, x_context.dtype

    # Setup
    self.init_kv_cache(B, Nc + Nt, device, dtype)
    self.prefill_context(x_context, y_context)

    predictions = []
    prev_x, prev_y = None, None

    for t in range(Nt):
        x_t = x_target[:, t:t+1, :]
        ar_idx = t % self.ar_tokens.shape[0]

        y_t = self.autoregressive_decode(x_t, prev_x, prev_y, ar_idx)
        predictions.append(y_t)

        # Current becomes previous for next iteration
        prev_x, prev_y = x_t, y_t

    return torch.cat(predictions, dim=1)  # [B, Nt]


def evaluate_log_likelihood(
    self,
    x_context: Tensor,
    y_context: Tensor,
    x_target: Tensor,
    y_target: Tensor,  # True values
) -> Tuple[Tensor, Tensor]:
    """
    Sample predictions AND compute log-likelihood of true targets.

    Returns:
        y_pred: [B, Nt] sampled predictions
        log_ll: [B, Nt] log-likelihood of y_target under model
    """
    # Same as sample_sequence, but also call head.log_likelihood(z, y_target)
    ...
```

---

### Phase 6: Testing

```python
def test_prefill_matches_forward():
    """Context KV should match full forward pass."""

def test_single_target_decode():
    """Single target prediction works."""

def test_two_target_decode():
    """Second target correctly attends to first buffer."""

def test_full_sequence():
    """Full AR generation produces valid outputs."""

def test_log_likelihood():
    """LL computation matches head.log_likelihood."""
```

---

## Open Questions

1. **Batch size for AR decoding**: Should we support decoding multiple targets per step (like ACE's K parameter), or start with K=1 for simplicity?

2. **Cache reuse**: Should `sample_sequence` automatically reuse cache if called multiple times with same context? Or require explicit `clear_cache()`?

3. **Feature attention**: Since C=1, should we skip feature attention entirely in inference for efficiency? Or keep it for correctness/simplicity?

4. **Buffer handling**: The current model uses buffer for training. For inference, do we:
   - Use buffer (convert predictions to buffer embeddings)?
   - Or treat predictions as extended context?

   ACE uses buffer approach. Should we match that?

---

## File Structure

```
autoregressive_nano_tabpfn/
├── model/
│   ├── __init__.py        # Add InferenceEngine export
│   ├── inference.py       # NEW: InferenceEngine class
│   ├── model.py           # Existing (may need minor changes)
│   └── attention.py       # Existing
└── tests/
    └── test_inference_engine.py  # NEW: Tests
```

---

**Please review. Edit directly if needed, then confirm to proceed.**
