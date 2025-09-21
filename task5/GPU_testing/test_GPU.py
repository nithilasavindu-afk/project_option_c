import torch
import numpy as np
import time

print("=== Deep Learning GPU Setup Test ===")
print(f"Python version: {torch.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {total_memory:.1f} GB")
    
    # Performance test
    print("\n=== GPU Performance Test ===")
    device = torch.device("cuda")
    
    # Create test matrices
    size = 1000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Time GPU computation
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(10):
        c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"GPU computation time (10x {size}x{size} matrix multiply): {gpu_time:.3f} seconds")
    
    # Memory usage test
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print("\n✅ GPU is ready for deep learning!")
    
else:
    print("\n❌ CUDA not available. Check your installation.")
    print("Your models will run on CPU (slower).")