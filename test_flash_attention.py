import pytest
import torch
import torch.nn.functional as F
from model import (
    TransformerModelArgs,
    Attention,
    precompute_freqs_cis,
    apply_rotary_emb,
    repeat_kv,
    Transformer
)
from flash_attn_interface import flash_attn_func
from torch.nn.utils import parameters_to_vector

# -----------------------------------------------------------
# 1) Reference attention (standard SDPA) forward/backward tests
# -----------------------------------------------------------
@pytest.mark.parametrize("bs, sl, nh, hd", [
    (1, 16, 1, 16),
    (2, 64, 2, 32),
    (4, 128, 4, 16),
    (8, 256, 8, 16),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_attention_forward_backward(bs, sl, nh, hd, dtype):
    """
    Verify the standard PyTorch scaled_dot_product_attention path for various dtypes.
    """
    dim = nh * hd
    args = TransformerModelArgs(dim=dim, n_heads=nh, seq_len=sl)
    attn = Attention(args).cuda().to(dtype=dtype).eval()

    # inputs
    x = torch.randn(bs, sl, dim, device="cuda", dtype=dtype, requires_grad=True)
    freqs = precompute_freqs_cis(hd, sl).cuda().to(dtype)

    # forward (reference)
    out_ref = attn(x, freqs, use_flash_attn=False)

    # backward
    loss = out_ref.sum()
    loss.backward()

    # ensure gradients are non-zero and finite
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0

# -----------------------------------------------------------
# 2) FlashAttention vs reference forward/backward for bf16
# -----------------------------------------------------------
@pytest.mark.parametrize("bs, sl, nh, hd", [
    (1, 16, 1, 16),
    (2, 64, 2, 32),
    (4, 128, 4, 16),
    (8, 256, 8, 16),
])
def test_flash_attention_forward_backward(bs, sl, nh, hd):
    """
    Compare FlashAttention3 output to reference for bfloat16 only.
    """
    dtype = torch.bfloat16
    dim = nh * hd
    args = TransformerModelArgs(dim=dim, n_heads=nh, seq_len=sl)
    attn = Attention(args).cuda().to(dtype=dtype).eval()

    # inputs
    x_ref = torch.randn(bs, sl, dim, device="cuda", dtype=dtype, requires_grad=True)
    x_flash = x_ref.clone().detach().requires_grad_()
    freqs = precompute_freqs_cis(hd, sl).cuda().to(dtype)

    # forward
    out_ref = attn(x_ref, freqs, use_flash_attn=False)
    out_flash = attn(x_flash, freqs, use_flash_attn=True)

    # compare
    tol = 1e-2
    assert torch.allclose(out_ref, out_flash, atol=tol, rtol=tol), \
        f"Flash vs reference max diff: {(out_ref - out_flash).abs().max()}"

    # backward
    loss_ref = out_ref.sum()
    loss_flash = out_flash.sum()
    loss_ref.backward()
    loss_flash.backward()
    assert torch.allclose(x_ref.grad, x_flash.grad, atol=tol, rtol=tol), \
        f"Gradients max diff: {(x_ref.grad - x_flash.grad).abs().max()}"

# -----------------------------------------------------------
# 3) Causal vs non-causal masks
# -----------------------------------------------------------
def test_causal_vs_noncausal():
    """
    Verify that causal=True and causal=False paths differ,
    and that flash matches reference for non-causal.
    """
    bs, sl, nh, hd = 2, 32, 4, 16
    dtype = torch.bfloat16
    dim = nh * hd
    args = TransformerModelArgs(dim=dim, n_heads=nh, seq_len=sl)
    attn = Attention(args).cuda().to(dtype=dtype).eval()

    # inputs
    x = torch.randn(bs, sl, dim, device="cuda", dtype=dtype)
    freqs = precompute_freqs_cis(hd, sl).cuda().to(dtype)

    # extract Q, K, V
    xq = attn.wq(x).view(bs, sl, nh, hd)
    xk = attn.wk(x).view(bs, sl, nh, hd)
    xv = attn.wv(x).view(bs, sl, nh, hd)
    xq, xk = apply_rotary_emb(xq, xk, freqs)
    keys = repeat_kv(xk, attn.n_rep)
    values = repeat_kv(xv, attn.n_rep)

    # reference non-causal
    q_t, k_t, v_t = map(lambda t: t.transpose(1,2), (xq, keys, values))
    out_ref_nc = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
    out_ref_nc = out_ref_nc.transpose(1,2).reshape(bs, sl, dim)

    # flash non-causal
    out_flash_nc = flash_attn_func(xq, keys, values, causal=False)[0].reshape(bs, sl, dim)

    # compare non-causal
    assert torch.allclose(out_ref_nc, out_flash_nc, atol=1e-2, rtol=1e-2)

    # flash causal vs non-causal difference
    out_flash_c = flash_attn_func(xq, keys, values, causal=True)[0].reshape(bs, sl, dim)
    assert not torch.allclose(out_flash_c, out_flash_nc)

# -----------------------------------------------------------
# 4) Dropout mask consistency
# -----------------------------------------------------------
def test_dropout_mask_consistency():
    """
    Ensure dropout masks are identical for ref and flash when seeding.
    """
    bs, sl, nh, hd = 2, 32, 4, 16
    dtype = torch.bfloat16
    dim = nh * hd
    args = TransformerModelArgs(dim=dim, n_heads=nh, seq_len=sl)
    attn = Attention(args).cuda().to(dtype=dtype).eval()

    x = torch.randn(bs, sl, dim, device="cuda", dtype=dtype)
    freqs = precompute_freqs_cis(hd, sl).cuda().to(dtype)

    p, seed = 0.2, 1234
    torch.manual_seed(seed)
    out_ref = attn(x, freqs, use_flash_attn=False)
    drop_ref = F.dropout(out_ref, p=p, training=True)

    torch.manual_seed(seed)
    out_flash = attn(x, freqs, use_flash_attn=True)
    drop_flash = F.dropout(out_flash, p=p, training=True)

    mask_ref = (drop_ref == 0)
    mask_flash = (drop_flash == 0)
    assert torch.equal(mask_ref, mask_flash)

# -----------------------------------------------------------
# 5) Integration-style one-step training sanity check
# -----------------------------------------------------------
@pytest.mark.parametrize("use_flash", [False, True])
def test_integration_train_step(use_flash):
    """
    Run one train step and ensure no NaNs and parameters update.
    """
    # small transformer config
    vocab_size, sl, dim, nh = 50, 16, 32, 4
    args = TransformerModelArgs(
        dim=dim,
        n_heads=nh,
        seq_len=sl,
        vocab_size=vocab_size,
        n_layers=1,
    )
    model = Transformer(args, use_flash_attn=use_flash).cuda()
    # for flash, use supported dtype
    if use_flash:
        model = model.to(dtype=torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tokens = torch.randint(0, vocab_size, (2, sl), device="cuda")
    labels = tokens.clone()

    # record initial parameters
    init_params = parameters_to_vector(model.parameters()).detach().clone()

    # forward/backward/step
    logits = model(tokens)
    loss = F.cross_entropy(logits.flatten(0,1), labels.flatten(0,1))
    loss.backward()
    optimizer.step()

    # checks
    assert torch.isfinite(loss).all()
    new_params = parameters_to_vector(model.parameters())
    assert not torch.allclose(init_params, new_params)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    pytest.main([__file__])
