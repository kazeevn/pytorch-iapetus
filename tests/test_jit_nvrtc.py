import torch

@torch.jit.script
def fused_op(x, y):
    return x * x + torch.sin(y) * 2.0

for dev in ['cuda:0', 'cuda:2']:
    print(f'Testing JIT kernel on {dev}...')
    x = torch.randn(1000, device=dev)
    y = torch.randn(1000, device=dev)
    for _ in range(5):
        out = fused_op(x, y)
    torch.cuda.synchronize(dev)
    expected = x * x + torch.sin(y) * 2.0
    assert torch.allclose(out, expected), f'Mismatch on {dev}'
    print(f'  JIT compilation and execution succeeded on {dev}!')

print('ALL JIT TESTS PASSED!')
