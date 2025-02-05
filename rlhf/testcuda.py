import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.version.enableCuda)
print(torch.cuda.memory_allocated())  # Memory allocated on GPU
print(torch.cuda.memory_reserved())  # Memory reserved for caching

# TODO Delete