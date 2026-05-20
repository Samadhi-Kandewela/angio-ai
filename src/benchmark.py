import torch
import torch.nn as nn
import time
import argparse
import numpy as np
from thop import profile, clever_format

from model import get_model as get_unet
from model_lightweight import get_model as get_lightweight_unet

def benchmark_model(model_name, n_classes=1, input_size=(1, 3, 512, 512), device='cpu'):
    print(f"Benchmarking {model_name} on {device}...")
    
    # Load model
    if model_name == 'unet':
        model = get_unet(n_classes=n_classes)
    else:
        model = get_lightweight_unet(model_name, n_classes=n_classes)
        
    if model is None:
        print(f"Model {model_name} not found!")
        return
        
    model.eval()
    model.to(device)
    
    input_tensor = torch.randn(input_size).to(device)
    
    # 1. FLOPs and Params
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"[{model_name}] Params: {params_str}, FLOPs: {flops_str}")
    
    # 2. Inference Speed (FPS)
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
        
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            times.append(end - start)
            
    avg_time = np.mean(times)
    fps = 1 / avg_time
    
    print(f"[{model_name}] Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"[{model_name}] FPS: {fps:.2f}\n")
    
    return {
        'model': model_name,
        'params': params,
        'flops': flops,
        'latency_ms': avg_time * 1000,
        'fps': fps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device to benchmark on (cpu or cuda)')
    args = parser.parse_args()
    
    models_to_test = ['unet', 'dscunet', 'mobileunet', 'mobileunetv3']
    results = []
    
    for m in models_to_test:
        res = benchmark_model(m, device=args.device)
        if res:
            results.append(res)
            
    # Print Summary Table
    print(f"{'Model':<15} | {'Params':<10} | {'FLOPs':<10} | {'Latency (ms)':<15} | {'FPS':<10}")
    print("-" * 75)
    for r in results:
        flops_str, params_str = clever_format([r['flops'], r['params']], "%.3f")
        print(f"{r['model']:<15} | {params_str:<10} | {flops_str:<10} | {r['latency_ms']:<15.2f} | {r['fps']:<10.2f}")
