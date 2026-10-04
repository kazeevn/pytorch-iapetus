"""Check that the pinned cuDNN runs convolutions on every installed GPU."""

import torch
import torch.nn.functional as F


assert torch.backends.cudnn.is_available(), "PyTorch was built without cuDNN"
assert torch.backends.cudnn.version() == 8700, torch.backends.cudnn.version()
assert torch.cuda.is_available(), "CUDA GPU required for the cuDNN test"

torch.manual_seed(0)
cpu_x = torch.randn(2, 8, 32, 32, requires_grad=True)
cpu_weight = torch.randn(16, 8, 3, 3, requires_grad=True)
cpu_y = F.conv2d(cpu_x, cpu_weight, padding=1)
cpu_y.square().mean().backward()

for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    x = cpu_x.detach().to(device).requires_grad_()
    weight = cpu_weight.detach().to(device).requires_grad_()
    with torch.backends.cudnn.flags(enabled=True, benchmark=False):
        with torch.autograd.profiler.profile() as profile:
            y = F.conv2d(x, weight, padding=1)
            y.square().mean().backward()
            torch.cuda.synchronize(device)
    operations = {event.key for event in profile.key_averages()}
    assert "aten::cudnn_convolution" in operations, (
        f"cuDNN convolution was not used on {torch.cuda.get_device_name(index)}: {operations}"
    )
    torch.testing.assert_close(y.cpu(), cpu_y.detach(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad.cpu(), cpu_x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weight.grad.cpu(), cpu_weight.grad, rtol=1e-4, atol=1e-4)
    print(f"cuDNN 8.7 convolution: PASS on {torch.cuda.get_device_name(index)}")
