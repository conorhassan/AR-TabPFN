# Final Analysis: Triton Kernel Integration for Joint Sample Generation

## Executive Summary

This analysis synthesizes findings from **15 parallel Opus sub-agents** across two analysis cycles examining the integration of the `cross_attention_shared_ctx_kernel` into nanoTabPFN's joint sample generation.

**Key Insight**: The context K/V is genuinely shared across batch elements - it's the same training data. The current inference code's `[B, H, seq_len, D]` cache shape exists only because `flex_attention` requires matching batch dimensions. The Triton kernel's design (`[H, Nctx, D]` for K/V) is **exactly the right abstraction** for this problem.

**The Real Question**: How do we restructure the inference architecture to exploit the Triton kernel's shared-context optimization while handling the buffer K/V that diverges during AR generation?

---

## 1. The Optimization Opportunity

### Why Shared Context Matters

In nanoTabPFN, the context (training data) is **identical** for all batch elements:
- Same `x_context`, `y_context` is passed to all samples
- Context K/V projections produce identical results across batch
- The only reason to have batch dimension in K/V cache is API compatibility with `flex_attention`

### Memory Bandwidth Impact

For typical inference with B=512 samples, Nc=100 context points:

| Approach | Context K/V Memory Traffic |
|----------|---------------------------|
| Current (per-batch K/V) | `B * H * Nc * D * 2 = 512 * 4 * 100 * 32 * 2 = 13MB` per layer |
| Triton (shared K/V) | `H * Nc * D * 2 = 4 * 100 * 32 * 2 = 25.6KB` per layer |
| **Reduction** | **~500x** for context portion |

This is the core value proposition of the Triton kernel.

---

## 2. Integration Design

### The Hybrid Attention Pattern

The solution is to **separate context attention from buffer attention**, then combine them properly:

```
Attention Output = combine(
    TritonContextAttention(Q, K_ctx_shared, V_ctx_shared),
    BufferAttention(Q, K_buf_batched, V_buf_batched)
)
```

Where:
- `K_ctx_shared`: `[H, Nctx, D]` - computed once, shared
- `K_buf_batched`: `[B, H, Nbuf, D]` - per-sample, diverges during AR

### Cache Architecture Change

**Current** (`inference.py`):
```python
layer.k_cache = torch.zeros(B, H, max_seq_len, Dh)  # Monolithic, batched
layer.v_cache = torch.zeros(B, H, max_seq_len, Dh)
```

**Proposed**:
```python
# Shared context (computed once during prefill)
layer.k_ctx = None  # Will be [H, Nctx, Dh]
layer.v_ctx = None  # Will be [H, Nctx, Dh]

# Per-batch buffer (grows during AR generation)
layer.k_buf = torch.zeros(B, H, max_buf_len, Dh)
layer.v_buf = torch.zeros(B, H, max_buf_len, Dh)
```

### Integration Points

#### 1. `prefill_context` - Store Shared Context

```python
def prefill_context(self, x_context, y_context):
    ctx_emb = self.embedder.embed_context(x_context, y_context)  # [B, Nc, D]

    for layer in self.backbone.layers:
        # Compute K/V projections
        k = layer.attn_rows.k_proj(x_row)  # [B, Nc, D]
        v = layer.attn_rows.v_proj(x_row)  # [B, Nc, D]

        # All batch elements have IDENTICAL context K/V
        # Store only once: [H, Nc, Dh]
        k_reshape = k[0].view(Nc, H, Dh).transpose(0, 1)  # [H, Nc, Dh]
        layer.k_ctx = k_reshape
        layer.v_ctx = v[0].view(Nc, H, Dh).transpose(0, 1)

        # Continue with self-attention using full batched K/V for now
        # (Context self-attention doesn't benefit from sharing)
```

#### 2. `_layer_decode_with_cache` - Hybrid Attention

```python
def _layer_decode_with_cache_hybrid(self, layer, x, cache_start):
    # ... Q/K/V projections as before ...

    # Update buffer cache (per-batch)
    buf_pos = cache_start - self.context_len
    layer.k_buf[:, :, buf_pos:buf_pos+N] = k_new
    layer.v_buf[:, :, buf_pos:buf_pos+N] = v_new

    # Option A: Use Triton for context, flex_attention for buffer
    ctx_out, (m_ctx, l_ctx) = triton_context_attention_with_stats(
        q, layer.k_ctx, layer.v_ctx
    )

    buf_out, (m_buf, l_buf) = flex_buffer_attention_with_stats(
        q, layer.k_buf[:, :, :buf_pos+N], layer.v_buf[:, :, :buf_pos+N],
        causal_mask
    )

    # Combine with online softmax
    attn_out = online_softmax_combine(ctx_out, m_ctx, l_ctx, buf_out, m_buf, l_buf)

    # ... rest as before ...
```

#### 3. Online Softmax Combination

The key to combining two attention sources correctly:

```python
def online_softmax_combine(out_a, m_a, l_a, out_b, m_b, l_b):
    """
    Combine two attention outputs computed with different max values.

    out_a = sum(exp(s_a - m_a) * v_a)  (unnormalized)
    l_a = sum(exp(s_a - m_a))

    Returns: softmax([s_a, s_b]) @ [v_a, v_b]
    """
    m_combined = torch.maximum(m_a, m_b)

    # Rescale factors
    alpha_a = torch.exp(m_a - m_combined)
    alpha_b = torch.exp(m_b - m_combined)

    # Combined denominator
    l_combined = l_a * alpha_a + l_b * alpha_b

    # Combined numerator (already weighted by attention)
    out_combined = out_a * alpha_a + out_b * alpha_b

    return out_combined / l_combined
```

### Triton Kernel Modification

The current kernel returns normalized output. For proper combination, we need **unnormalized accumulator + softmax statistics**:

```python
# Modified kernel outputs:
# - acc: sum(exp(s - m) * v)  [unnormalized]
# - m: max(s)
# - l: sum(exp(s - m))
```

This is a minor modification to the existing kernel (skip final division, return `m` and `l`).

---

## 3. Use Case Viability

| Use Case | Context Shared? | Buffer? | Triton Viable? | Benefit |
|----------|-----------------|---------|----------------|---------|
| **Batched point prediction** | YES | None | **YES** | High |
| **`evaluate_joint_density`** | YES | Batched, known | **YES** (hybrid) | Medium |
| **`sample_sequence` step 0** | YES | None yet | **YES** | Low |
| **`sample_sequence` step 1+** | YES | Diverged | **YES** (hybrid) | Medium |

All use cases are viable with the hybrid approach. The buffer attention portion cannot be optimized, but the context attention portion (often 50-80% of total K/V for moderate Nc) benefits from sharing.

---

## 4. Implementation Plan

### Phase 1: Kernel Modification (2-4 hours)

Modify `triton_cross_attention` to return softmax statistics:

```python
def triton_cross_attention_with_stats(q, k_ctx, v_ctx):
    """Returns (unnormalized_output, max_scores, sum_exp_scores)"""
    # ... kernel modification to not normalize and return m, l
```

### Phase 2: Cache Restructuring (4-8 hours)

1. Modify `init_kv_cache` to create separate context/buffer caches
2. Modify `prefill_context` to store shared context K/V
3. Add `context_len` tracking separate from `seq_len`

### Phase 3: Hybrid Decode (8-16 hours)

1. Implement `_layer_decode_with_cache_hybrid`
2. Add `online_softmax_combine` utility
3. Handle edge cases (empty buffer, first step, etc.)

### Phase 4: Testing (4-8 hours)

1. Numerical equivalence tests vs current implementation
2. Performance benchmarks
3. Edge case coverage

**Total Estimate**: 18-36 hours of engineering effort

---

## 5. Alternative: Pure PyTorch Split-Attention

If Triton kernel modification is complex, the same optimization can be achieved in pure PyTorch:

```python
def split_attention_with_shared_context(q, k_ctx, v_ctx, k_buf, v_buf, buf_mask=None):
    """
    Context K/V: [H, Nctx, D] - shared
    Buffer K/V: [B, H, Nbuf, D] - per-batch
    """
    B, H, Lq, D = q.shape
    scale = D ** -0.5

    # Context attention (efficient einsum avoids expand)
    s_ctx = torch.einsum('bhld,hnd->bhln', q, k_ctx) * scale
    m_ctx = s_ctx.max(dim=-1, keepdim=True).values.clamp(min=-1e4)
    exp_ctx = torch.exp(s_ctx - m_ctx)
    l_ctx = exp_ctx.sum(dim=-1, keepdim=True)
    acc_ctx = torch.einsum('bhln,hnd->bhld', exp_ctx, v_ctx)

    # Buffer attention
    s_buf = torch.matmul(q, k_buf.transpose(-2, -1)) * scale
    if buf_mask is not None:
        s_buf = s_buf.masked_fill(~buf_mask, float('-inf'))
    m_buf = s_buf.max(dim=-1, keepdim=True).values.clamp(min=-1e4)
    exp_buf = torch.exp(s_buf - m_buf)
    l_buf = exp_buf.sum(dim=-1, keepdim=True)
    acc_buf = torch.matmul(exp_buf, v_buf)

    # Combine
    m_max = torch.maximum(m_ctx, m_buf)
    alpha_ctx = torch.exp(m_ctx - m_max)
    alpha_buf = torch.exp(m_buf - m_max)
    l_total = (l_ctx * alpha_ctx + l_buf * alpha_buf).clamp(min=1e-9)

    return (acc_ctx * alpha_ctx + acc_buf * alpha_buf) / l_total
```

This provides the same **memory sharing benefit** (context K/V not expanded to batch) without Triton, though with less aggressive optimization of the context attention itself.

---

## 6. Recommendations

### Immediate (This Week)

1. **Fix documentation bug** in `model.py` line 9 - it incorrectly states inference uses Triton kernels
2. **Add kernel tests** to verify correctness against PyTorch reference

### Short-Term (1-2 Weeks)

3. **Implement pure PyTorch split-attention** as proof of concept
4. **Benchmark** against current implementation to quantify gains

### Medium-Term (If Benchmarks Show > 20% Improvement)

5. **Modify Triton kernel** to return softmax statistics
6. **Implement hybrid decode** with Triton context + flex_attention buffer
7. **Comprehensive testing** for numerical equivalence

### Decision Points

- If pure PyTorch split-attention shows < 10% improvement: defer Triton integration
- If split-attention shows 10-30% improvement: implement Triton for additional gains
- If split-attention shows > 30% improvement: prioritize full integration

---

## 7. Summary

The Triton kernel's design is **correct for this problem**. The context K/V is genuinely shared, and the kernel exploits this. The challenge is not "can it work" but "how do we integrate it cleanly with the buffer attention that cannot be shared".

The hybrid approach (Triton for context, flex_attention for buffer, online softmax combination) is the right architecture. Implementation requires:
1. Cache restructuring to separate context from buffer
2. Kernel modification to expose softmax statistics
3. Careful combination of the two attention sources

The pure PyTorch alternative provides the same memory sharing benefit with less aggressive optimization, and can serve as both a stepping stone and a fallback.

---

*Analysis completed using parallel deep reasoning with 15 Opus sub-agents across 2 refinement cycles.*
