# Bug: Triton kernel LSE mismatch

Output is correct (~1e-3), but LSE differs by ~0.8 from PyTorch's `logsumexp()`.

## Triton kernel (stores LSE as `[H, B]`)
```python
# LSE storage at end of kernel
off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
LSE_ptr_base = LSE_ptr + pid_h * B + off_b  # <-- [H, B] layout
tl.store(LSE_ptr_base, lse, mask=off_b < B)
```

## Wrapper (allocates LSE as `[B, H]`)
```python
lse = torch.empty((B, H), device=q.device, dtype=torch.float32)  # <-- [B, H] layout
```

## The bug
Kernel writes `[H, B]`, wrapper expects `[B, H]`. Memory layout mismatch.

## Fix
Either:
1. Change wrapper: `lse = torch.empty((H, B), ...).T.contiguous()`
2. Change kernel: `LSE_ptr_base = LSE_ptr + off_b * H + pid_h`
