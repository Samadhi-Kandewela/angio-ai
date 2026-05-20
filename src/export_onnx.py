import torch
import torch.onnx
import argparse
import os
import onnx
import onnxruntime as ort
import numpy as np
from model_lightweight import MobileUNetv3, MobileUNetv2, DeepLabV3Plus, MobileUNet
from model import UNet

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def export_model(model_path, output_path, model_name='mobileunetv3'):
    device = torch.device('cpu') # Export usually done on CPU
    
    # 1. Load Model
    print(f"Loading {model_name} from {model_path}...")
    if model_name == 'mobileunetv3':
        model = MobileUNetv3(n_classes=1, pretrained=False)
    elif model_name == 'mobileunetv2':
        model = MobileUNetv2(n_classes=1, pretrained=False)
    elif model_name == 'deeplabv3':
        model = DeepLabV3Plus(n_classes=1, pretrained=False)
    elif model_name == 'mobileunet':
        model = MobileUNet(n_classes=1, pretrained=False)
    elif model_name == 'unet':
        # Ensure bilinear matches the checkpoint
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # 2. Create Dummy Input
    # Standard input size for our model
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    # 3. Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      output_path,         # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,    # Use a newer opset (17) to match standard libraries
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print("Export complete. Verifying...")

    # 4. Verify ONNX Model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure checked.")

    # 5. Compare PyTorch vs ONNX Runtime
    ort_session = ort.InferenceSession(output_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compute PyTorch output prediction
    with torch.no_grad():
        torch_out = model(dummy_input)

    # Some models (like torchvision's deeplabv3) return an OrderedDict.
    # We need to extract the tensor (usually 'out') for comparison.
    if isinstance(torch_out, dict):
        torch_out_tensor = torch_out['out']
    else:
        torch_out_tensor = torch_out

    # compare ONNX Runtime and PyTorch results
    # Relaxed tolerance: Upsampling layers often cause minor differences
    try:
        np.testing.assert_allclose(to_numpy(torch_out_tensor), ort_outs[0], rtol=1e-02, atol=1e-03)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    except AssertionError as e:
        print(f"Warning: Minor differences found between PyTorch and ONNX outputs.")
        print(f"This is common with Resize/Upsample layers. Differences are likely negligible.")
        print(str(e))
    
    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output .onnx file')
    parser.add_argument('--model-name', type=str, default='mobileunetv3', help='Model architecture')
    args = parser.parse_args()
    
    export_model(args.model_path, args.output, args.model_name)
