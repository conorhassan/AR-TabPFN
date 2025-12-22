# Plan: Triton Backend for Autoregressive Decoding

Created: 2025-12-19
Status: PENDING APPROVAL

## Summary

Integrate the Triton shared-context attention kernel as an alternative backend for `ARTabPFNPredictor`. This enables a more memory-efficient inference mode where context K/V is stored once (`[H, Nctx, D]`) instead of replicated per batch (`[B, H, Nctx, D]`), while buffer K/V remains per-batch. The Triton backend should be faster for large context sizes and significantly more memory efficient.

## Scope

- **In scope**:
  - Add backend selection to `ARTabPFNPredictor` (flex_attention vs triton)
  - Create alternative cache structure for Triton backend
  - Wire up `hybrid_attention` from `triton_kernels.py` into the inference path
  - Add integration tests in `test_triton_gpu.py`

- **Out of scope**:
  - Modifying the Triton kernel itself
  - Teacher forcing path (can be added later)
  - Multi-query (Lq > 1) support (kernel currently requires Lq=1)

## Key Insight

The architectural difference between backends:

| Aspect | Current (flex_attention) | New (Triton) |
|--------|-------------------------|--------------|
| Context K/V | `[B, H, Nctx, D]` per-batch | `[H, Nctx, D]` shared |
| Buffer K/V | Part of same cache | `[B, H, Nbuf, D]` separate |
| Memory | O(B * Nctx) | O(Nctx + B * Nbuf) |
| Attention | flex_attention over full KV | hybrid_attention (Triton ctx + PyTorch buf) |

---

## Phases

### Phase 1: Backend Selection Infrastructure

**Goal**: Add backend parameter and dispatch logic to `ARTabPFNPredictor`

**Work**:
- Add `backend: str = "flex_attention"` parameter to `__init__` and `from_trained_model`
- Valid values: `"flex_attention"`, `"triton"`
- Add runtime check: if `backend="triton"` but Triton unavailable, warn and fallback

**Steps**:
1. Modify `ARTabPFNPredictor.__init__()` to accept `backend` parameter
2. Modify `ARTabPFNPredictor.from_trained_model()` to accept and forward `backend`
3. Add import for `triton_available` from `triton_kernels.py`
4. Add validation logic with fallback warning

**Verification**:
- [ ] Constructor accepts backend parameter
- [ ] Fallback warning is logged if Triton unavailable

---

### Phase 2: Triton-Compatible Cache Structure

**Goal**: Create alternative KV cache structure optimized for Triton backend

**Work**:
- Context K/V: `[H, Nctx, D]` - stored once, shared across batch
- Buffer K/V: `[B, H, max_buf, D]` - per-batch, separate from context
- Track `context_len` and `buffer_len` separately

**Steps**:
1. Add `init_kv_cache_triton()` method that creates:
   - `layer.k_ctx_cache` / `layer.v_ctx_cache`: `[H, max_ctx, D]`
   - `layer.k_buf_cache` / `layer.v_buf_cache`: `[B, H, max_buf, D]`
2. Store `self.context_len` and `self.buffer_len` state
3. Modify `init_kv_cache()` to dispatch based on backend

**Verification**:
- [ ] Triton cache uses correct shapes
- [ ] Memory usage is reduced compared to flex_attention cache

---

### Phase 3: Triton Prefill Path

**Goal**: Implement context prefill that populates shared context cache

**Work**:
- `prefill_context_triton()`: Process context, store K/V as `[H, Nctx, D]`
- Key difference: K/V from first batch element is used (all batch elements have same context)

**Steps**:
1. Add `prefill_context_triton()` method
2. Process context through transformer layers
3. After projection, extract K/V and store as `[H, Nctx, D]` (squeeze batch dim)
4. Modify `prefill_context()` to dispatch based on backend

**Verification**:
- [ ] Context K/V cached with shape `[H, Nctx, D]`
- [ ] Output matches flex_attention path numerically

---

### Phase 4: Triton Decode Path

**Goal**: Implement autoregressive decode using `hybrid_attention`

**Work**:
- `_layer_decode_with_cache_triton()`: Use `hybrid_attention` for row attention
- Accumulate buffer K/V per-batch as decoding progresses

**Steps**:
1. Add `_layer_decode_with_cache_triton()` method
2. Compute Q, K, V projections as normal
3. Append new buffer K/V to `layer.k_buf_cache` / `layer.v_buf_cache`
4. Call `hybrid_attention(q, k_ctx, v_ctx, k_buf, v_buf)` instead of `flex_attention`
5. Modify `_layer_decode_with_cache()` to dispatch based on backend
6. Update `transformer_decode()` to use appropriate decode method

**Verification**:
- [ ] Decode produces correct output
- [ ] Buffer cache grows correctly during decoding

---

### Phase 5: Wire Up Complete Flow

**Goal**: Connect all pieces so `sample_sequence` works with Triton backend

**Steps**:
1. Ensure `autoregressive_decode()` works with both backends (no changes needed - it calls `transformer_decode`)
2. Update `sample_sequence()` dispatch (if needed)
3. Test end-to-end: `sample_sequence` with `backend="triton"`

**Verification**:
- [ ] `sample_sequence(backend="triton")` produces predictions
- [ ] Predictions match `backend="flex_attention"` numerically (within tolerance)

---

### Phase 6: Integration Tests

**Goal**: Add tests to `scripts/test_triton_gpu.py` to validate integration

**Steps**:
1. Add test: `test_predictor_triton_backend()` - basic smoke test
2. Add test: `test_predictor_backend_equivalence()` - compare Triton vs flex_attention outputs
3. Add benchmark: Compare memory usage and speed for both backends
4. Add test with varying context/target sizes

**Verification**:
- [ ] All integration tests pass
- [ ] Memory savings demonstrated
- [ ] Speed comparison documented

---

## Implementation Notes

### Attention Shape Translation

The Triton `hybrid_attention` expects:
```python
q: [B, H, Lq, D]      # Lq=1 for AR decode
k_ctx: [H, Nctx, D]   # Shared
v_ctx: [H, Nctx, D]   # Shared
k_buf: [B, H, Nbuf, D]  # Per-batch
v_buf: [B, H, Nbuf, D]  # Per-batch
```

Current inference code has:
```python
q: [B, H, N, Dh]  # Where N=1 or 2 (target or [buffer, target])
k_full: [B, H, total_len, Dh]  # From cache
```

The main work is restructuring the cache and attention call, not the projections.

### Buffer Position Tracking

In the Triton backend:
- Context positions: `[0, Nctx)` - filled during prefill
- Buffer positions: grow during decoding, one per committed buffer token
- `self.buffer_len` tracks current buffer length

### First Target Special Case

For the first target (no buffer), `hybrid_attention` handles empty buffer gracefully:
```python
if k_buf.shape[2] == 0:
    # Just return context attention
```

---

## Open Questions

1. **Should we support both backends simultaneously for A/B testing?** (e.g., run both and compare)
2. **Should the teacher forcing path (`evaluate_joint_density`) also support Triton?** (More complex due to batched tokens)

---

**Please review. Edit directly if needed, then confirm to proceed.**
