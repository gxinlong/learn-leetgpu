import torch
import smoke_gemm
import smoke_conv
import smoke_vector_add

torch.cuda.set_device(0)
torch.manual_seed(123)
for M, N, K in [(3, 5, 7), (17, 31, 9)]:
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.randn(M, N, device='cuda')
    expected = -0.75 * (a.cpu() @ b.cpu()) + 0.25 * c.cpu()
    smoke_gemm.solve(a, b, c, M, N, K, -0.75, 0.25)
    torch.cuda.synchronize()
    error = (c.cpu() - expected).abs().max().item()
    assert torch.allclose(c.cpu(), expected, atol=1e-5, rtol=1e-4), error
    print('GEMM numerical PASS', (M, N, K), 'max_abs_error', error, flush=True)

for IR, IC, KR, KC in [(7, 9, 3, 2), (17, 31, 2, 5)]:
    x = torch.randn(IR, IC, device='cuda')
    k = torch.randn(KR, KC, device='cuda')
    y = torch.empty(IR-KR+1, IC-KC+1, device='cuda')
    # CPU reference is independent of the Triton kernel and CUDA/cuDNN paths.
    expected = torch.nn.functional.conv2d(x.cpu()[None, None], k.cpu()[None, None])[0, 0]
    smoke_conv.solve(x, k, y, IR, IC, KR, KC)
    torch.cuda.synchronize()
    error = (y.cpu() - expected).abs().max().item()
    assert torch.allclose(y.cpu(), expected, atol=1e-5, rtol=1e-4), error
    print('CONV numerical PASS', (IR, IC, KR, KC), 'max_abs_error', error, flush=True)

for dtype in [torch.float32, torch.float16]:
    for N in [1, 1031, 65537]:
        a = torch.randn(N, device='cuda', dtype=dtype)
        b = torch.randn(N, device='cuda', dtype=dtype)
        c = torch.empty_like(a)
        expected = a.cpu() + b.cpu()
        smoke_vector_add.solve(a, b, c, N)
        torch.cuda.synchronize()
        assert torch.equal(c.cpu(), expected)
        print('VECTOR numerical PASS', dtype, N, flush=True)
