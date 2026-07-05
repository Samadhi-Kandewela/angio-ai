import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from localization_labels import NUM_ANATOMY_CLASSES
from model_multitask import MultiTaskMobileUNetv3


class ONNXMultiTaskWrapper(torch.nn.Module):
    """Expose dict outputs as stable ONNX outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["vessel"], out["anatomy"]


def export_model(model_path, output_path, image_size=512):
    device = torch.device("cpu")
    model = MultiTaskMobileUNetv3(
        n_anatomy_classes=NUM_ANATOMY_CLASSES,
        pretrained=False,
    )
    state_dict = torch.load(model_path, map_location=device)
    # strict=False: older checkpoints may still contain a now-removed stenosis_head
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = ONNXMultiTaskWrapper(model).eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["vessel", "anatomy"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "vessel": {0: "batch_size"},
            "anatomy": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(output_path)
    ort_outs = ort_session.run(None, {"input": dummy_input.numpy()})

    with torch.no_grad():
        torch_outs = wrapper(dummy_input)

    for name, torch_out, ort_out in zip(["vessel", "anatomy"], torch_outs, ort_outs):
        np.testing.assert_allclose(torch_out.numpy(), ort_out, rtol=1e-2, atol=1e-3)
        print(f"{name} output verified: {tuple(ort_out.shape)}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")


def get_args():
    parser = argparse.ArgumentParser(description="Export multitask localization model to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to multitask .pth checkpoint")
    parser.add_argument("--output", default="checkpoints/multitask_localization/multitask_best.onnx")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    export_model(args.model_path, args.output, args.image_size)
