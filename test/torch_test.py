import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_initialized())
print(torch.cuda.memory_summary())


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
