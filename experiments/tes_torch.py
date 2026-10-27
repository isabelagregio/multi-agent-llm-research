import torch

print("PyTorch version:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = x @ y

    print("Matrix multiplication successful!", flush=True)
    print("Result shape:", z.shape, flush=True)