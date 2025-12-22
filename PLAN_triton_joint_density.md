# Plan: Triton Kernel for Joint Density Evaluation

Created: 2025-12-20
Status: PENDING APPROVAL

## Summary

Implement a custom Triton kernel for teacher-forcing joint density evaluation to enable GPU-optimized inference for `evaluate_joint_density()`. Currently, this path falls back to `flex_attention` even when the Triton backend is selected. The new kernel will leverage the shared-context optimization pattern from the existing sampling kernel while handling the more complex teacher-forcing attention mask.

## Scope

- **In scope**:
  - New Triton kernel for multi-query teacher-forcing attention
  - PyTorch reference implementation for testing
  - Integration with `ARTabPFNPredictor.evaluate_joint_density()`
  - Comprehensive test suite

- **Out of scope**:
  - Changes to the sampling path (already working)
  - Changes to the model architecture
  - Performance benchmarking (can be done as follow-up)

## Background: Current Architecture

### Existing Triton Kernel (`triton_kernels.py:40-185`)
- Optimizes shared-context attention where K/V is shared across batch dimension
- Constraint: `Lq == 1` (single query per batch element)
- Returns LSE statistics for merging with buffer attention

### Teacher-Forcing Attention Pattern (`inference.py:483-535`)
The joint density evaluation processes `2*Nt` tokens at once:
- Sequence: `[Buffer_0, ..., Buffer_{Nt-1}, Target_0, ..., Target_{Nt-1}]`
- Buffer_i attends to: context (all) + buffers [0..i] (causal)
- Target_i attends to: context (all) + buffers [0..i-1] (strictly < i)

### Key Insight
The shared-context optimization still applies: context K/V (`[H, Nc, D]`) is shared across all batch elements. We can:
1. Compute context attention with Triton (multi-query version)
2. Compute buffer-to-buffer/target-to-buffer attention with PyTorch
3. Merge using LSE statistics

---

## Phases

### Phase 1: Extend Triton Kernel for Multi-Query Context Attention

**Goal**: Create a Triton kernel that handles `Lq > 1` queries attending to shared context.

**Files to modify/create**:
- `autoregressive_nano_tabpfn/model/triton_kernels.py` - Add new kernel

**Design**:
```python
@triton.jit
def _shared_context_attn_kernel_multiq(
    # Same structure as existing, but with Lq dimension
    Q_ptr, K_ptr, V_ptr, Out_ptr, LSE_ptr,
    B, H, LQ, N_CTX, HEAD_DIM: tl.constexpr,
    # Additional strides for Lq dimension
    stride_q_b, stride_q_h, stride_q_lq, stride_q_d,
    ...
):
    # Grid: (cdiv(B*LQ, BLOCK_SIZE_BQ), H)
    # Each program handles a tile of (batch, query) pairs
```

**Key differences from existing kernel**:
- Grid parallelizes over `B * Lq` instead of just `B`
- Q block pointer includes Lq dimension
- Output and LSE shapes include Lq dimension

**Work**:
- [ ] Add `_shared_context_attn_kernel_multiq` kernel function
- [ ] Add `triton_context_attention_multiq` Python wrapper
- [ ] Add `pytorch_context_attention_multiq` reference implementation
- [ ] Update existing `triton_context_attention` to delegate to multiq version

**Verification**:
- [ ] Unit tests comparing Triton vs PyTorch for various `(B, H, Lq, N_CTX, D)` combinations
- [ ] Test edge cases: non-divisible batch/query/context sizes

---

### Phase 2: Implement Buffer Attention with Teacher-Forcing Mask

**Goal**: Efficient attention between queries and buffer K/V with the teacher-forcing mask pattern.

**Files to modify**:
- `autoregressive_nano_tabpfn/model/triton_kernels.py`

**Design**:
The buffer attention in teacher-forcing has a specific pattern:
- Queries are `[Buffer_0..Nt-1, Target_0..Nt-1]` (2*Nt queries)
- Keys/Values are the same buffers (Nt buffers)
- Buffer_i attends to buffers [0..i] (causal)
- Target_i attends to buffers [0..i-1] (strictly less)

```python
def pytorch_teacher_forcing_buffer_attention(
    q: Tensor,           # [B, H, 2*Nt, D]
    k_buf: Tensor,       # [B, H, Nt, D]
    v_buf: Tensor,       # [B, H, Nt, D]
    num_targets: int,    # Nt
) -> Tuple[Tensor, Tensor]:
    """Buffer attention with teacher-forcing mask, returns LSE for merging."""
```

**Work**:
- [ ] Implement `pytorch_teacher_forcing_buffer_attention` function
- [ ] Create mask generation logic matching `_create_teacher_forcing_mask`
- [ ] Handle the case where Target_0 has no buffers to attend to (only context)

**Verification**:
- [ ] Compare output against `flex_attention` with teacher-forcing BlockMask
- [ ] Test various Nt values including edge cases (Nt=1)

---

### Phase 3: Create Hybrid Teacher-Forcing Attention

**Goal**: Combine context and buffer attention with LSE merging for teacher-forcing.

**Files to modify**:
- `autoregressive_nano_tabpfn/model/triton_kernels.py`

**Design**:
```python
def hybrid_teacher_forcing_attention(
    q: Tensor,           # [B, H, 2*Nt, D] - buffers then targets
    k_ctx: Tensor,       # [H, Nc, D] - shared context
    v_ctx: Tensor,       # [H, Nc, D]
    k_buf: Tensor,       # [B, H, Nt, D] - buffer K/V
    v_buf: Tensor,       # [B, H, Nt, D]
    num_targets: int,    # Nt
    use_triton: bool = True,
) -> Tensor:
    """
    Teacher-forcing attention for joint density evaluation.

    1. Context attention (Triton multi-query)
    2. Buffer attention (PyTorch with TF mask)
    3. LSE merge
    """
```

**Special handling**:
- Target_0 only attends to context (no buffers) - buffer LSE should be -inf
- All queries attend to full context (no masking)

**Work**:
- [ ] Implement `hybrid_teacher_forcing_attention`
- [ ] Handle Target_0 edge case correctly
- [ ] Implement merge with proper LSE handling for -inf cases

**Verification**:
- [ ] Compare against full teacher-forcing attention (flex_attention path)
- [ ] Test batch size 1 and larger batches
- [ ] Test Nt=1, Nt=2, and larger values

---

### Phase 4: Integrate into ARTabPFNPredictor

**Goal**: Wire up the new Triton path in `evaluate_joint_density()`.

**Files to modify**:
- `autoregressive_nano_tabpfn/model/inference.py`

**Design**:
```python
def evaluate_joint_density(self, x_context, y_context, x_target, y_target):
    if self.backend == "triton":
        return self._evaluate_joint_density_triton(...)
    return self._evaluate_joint_density_flex(...)

def _evaluate_joint_density_triton(self, ...):
    # Similar to flex version but:
    # 1. Uses separate context/buffer caches
    # 2. Calls hybrid_teacher_forcing_attention in layers
```

**Work**:
- [ ] Add `_evaluate_joint_density_triton` method
- [ ] Add `_teacher_forcing_decode_triton` method
- [ ] Add `_layer_teacher_forcing_triton` for layer-level computation
- [ ] Remove the fallback warning in `evaluate_joint_density`

**Verification**:
- [ ] Compare Triton vs Flex outputs for various batch/target sizes
- [ ] Integration test with full predictor

---

### Phase 5: Testing and Validation

**Goal**: Comprehensive test coverage ensuring correctness.

**Files to modify/create**:
- `autoregressive_nano_tabpfn/tests/test_triton_kernel.py`

**Work**:
- [ ] Add `TestTritonMultiQueryAttention` class
- [ ] Add `TestTeacherForcingBufferAttention` class
- [ ] Add `TestHybridTeacherForcingAttention` class
- [ ] Add integration tests for `evaluate_joint_density` with Triton backend
- [ ] Test numerical stability edge cases

**Verification**:
- [ ] All tests pass with `pytest`
- [ ] Tests cover various parameter combinations

---

## Implementation Notes

### Kernel Design Considerations

1. **Grid Parallelization**: The multi-query kernel should parallelize over `(B * Lq, H)` to maximize occupancy.

2. **Memory Access Pattern**: Queries are laid out as `[B, H, Lq, D]`. The kernel should process tiles of queries efficiently.

3. **Buffer Attention Optimization**: For moderate Nt values, PyTorch SDPA may be sufficient. A Triton kernel for buffer attention could be a future optimization.

4. **LSE Handling**: When a query has no valid KV positions (e.g., Target_0 with no buffers), the LSE should be -inf and the merge should correctly return only the context contribution.

### Numerical Stability

- LSE computation should use float32 accumulators
- Merge should handle -inf LSE values gracefully
- Online softmax in Triton kernel already handles numerical stability

### Testing Strategy

1. **Unit tests**: Each new function tested in isolation
2. **Integration tests**: Full `evaluate_joint_density` path
3. **Correctness tests**: Compare Triton vs PyTorch/Flex reference
4. **Edge cases**: Nt=1, non-divisible sizes, empty contexts

---

## Open Questions

1. **Buffer attention kernel**: Should we also write a Triton kernel for the buffer attention, or is PyTorch SDPA sufficient for the expected Nt sizes?

2. **Chunk processing**: For very large Nt values, should we add chunking to avoid memory issues?

---

**Please review. Edit directly if needed, then confirm to proceed.**
