import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)
    # Instantiate your PyTorch model
    N, C, H, W = 2, 8, 16, 16
    x = torch.randn(N, C, H, W).cuda()
    model = ConvModel(H, W, C, out_channels=32, kernel_size=5, stride=1, padding=1).cuda().eval()

    # Profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # Torch-Inductor compilation
        scripted_model = torch.compile(model, backend="inductor")

        # the first execution is warm up, the next should be steady-state
        for i in range(5):
            out = scripted_model(x)
    prof.export_chrome_trace("inductor_trace.json")

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
