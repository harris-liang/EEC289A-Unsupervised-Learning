import torch

# Apple Silicon
# https://developer.apple.com/metal/pytorch/
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# CUDA
if torch.cuda.is_available():
    cuda_device = torch.device("cuda")
    x = torch.ones(1, device=cuda_device)
    print (x)
else:
    print ("CUDA device not found.")