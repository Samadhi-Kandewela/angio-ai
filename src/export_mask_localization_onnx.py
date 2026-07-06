import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from localization_labels import MERGED_NUM_ANATOMY_CLASSES
from model_mask_localization import MaskLocalizationNet


class ONNXMaskLocalizationWrapper(torch.nn.Module):
    """Expose the dict output as a stable, named ONNX output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["anatomy"]


def export_model(model_path, output_path, image_size=512):
    device = torch.device("cpu")
    model = MaskLocalizationNet(n_anatomy_classes=MERGED_NUM_ANATOMY_CLASSES, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ONNXMaskLocalizationWrapper(model).eval()
    # Single-channel binary vessel mask input, not the 3-channel RGB frame --
    # this model takes the segmentation model's own output as its input.
    dummy_input = torch.randn(1, 1, image_size, image_size, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["anatomy"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "anatomy": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(output_path)
    ort_outs = ort_session.run(None, {"input": dummy_input.numpy()})

    with torch.no_grad():
        torch_out = wrapper(dummy_input)

    np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-2, atol=1e-3)
    print(f"anatomy output verified: {tuple(ort_outs[0].shape)}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")


def get_args():
    parser = argparse.ArgumentParser(description="Export mask-input localization model (MaskLocalizationNet) to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to mask_localization_v2 .pth checkpoint")
    parser.add_argument("--output", default="checkpoints/mask_localization_v2/best.onnx")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    export_model(args.model_path, args.output, args.image_size)
