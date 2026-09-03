"""
Reproducibility Utilities

What this teaches:
- Importance of reproducibility in ML
- How random seeds affect results
- Version tracking best practices

Educational:
Without fixed seeds, results vary between runs, making debugging
and comparison impossible. This is critical for:
- Scientific research (peer review)
- Production debugging (reproduce customer issues)
- A/B testing (fair comparisons)
"""

import os
import random
import numpy as np
import torch
from typing import Dict
import platform


def set_global_seed(seed: int = 42):
    """
    Set random seeds for all libraries.
    
    Args:
        seed: Random seed value (default: 42, the answer to everything)
    
    Educational:
    - Python's random module (for data loading, augmentation)
    - NumPy (for array operations)
    - PyTorch (for model training/inference)
    - CUDA (for GPU operations)
    
    Why 42? It's a reference to "Hitchhiker's Guide to the Galaxy"
    and has become the de-facto standard seed in ML community.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🎲 Random seed set to {seed} (reproducible results)")


def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility documentation.
    
    Returns:
        Dict with Python version, OS, hardware, etc.
    
    Educational:
    Results can vary across:
    - Python versions (different random algorithms)
    - OS (different BLAS implementations)
    - Hardware (CPU vs GPU, different GPU models)
    - Library versions (algorithm changes)
    
    Always document your environment!
    """
    import sys
    import cv2
    
    info = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'opencv_version': cv2.__version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info


def print_system_info():
    """Print system information for reproducibility."""
    print("\n" + "="*60)
    print("📋 SYSTEM INFORMATION (for reproducibility)")
    print("="*60)
    
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("="*60 + "\n")


def create_reproducibility_report(output_path: str = "reproducibility_report.txt"):
    """
    Create a reproducibility report file.
    
    This should be committed to git so others can reproduce your results.
    """
    info = get_system_info()
    
    with open(output_path, 'w') as f:
        f.write("REPRODUCIBILITY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Generated for: Autopilot Vision System\n\n")
        
        f.write("SYSTEM INFORMATION:\n")
        f.write("-" * 60 + "\n")
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("USAGE:\n")
        f.write("To reproduce results, ensure:\n")
        f.write("1. Python version matches\n")
        f.write("2. Install requirements: pip install -r requirements/base.txt\n")
        f.write("3. Set random seed: set_global_seed(42)\n")
        f.write("4. Use same model versions\n")
    
    print(f"📝 Reproducibility report saved to {output_path}")
