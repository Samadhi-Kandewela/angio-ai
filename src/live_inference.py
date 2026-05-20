import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import onnxruntime as ort
from model_lightweight import MobileUNetv2, MobileUNet, DSCUNet, DeepLabV3Plus, MobileUNetv3

def process_video(input_path, output_path, model_path, device='cpu'):
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.")
        return
        
    # Load Model
    is_onnx = model_path.endswith('.onnx')
    print(f"Loading model from {model_path} (Type: {'ONNX' if is_onnx else 'PyTorch'})...")
    
    if is_onnx:
        # ONNX Runtime
        try:
            ort_session = ort.InferenceSession(model_path)
            input_name = ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return
    else:
        # PyTorch
        try:
            if 'deeplab' in model_path.lower():
                model = DeepLabV3Plus(n_classes=1, pretrained=False)
            elif 'mobileunetv3' in model_path.lower():
                model = MobileUNetv3(n_classes=1, pretrained=False)
            elif 'mobileunetv2' in model_path.lower():
                model = MobileUNetv2(n_classes=1, pretrained=False)
            elif 'mobileunet' in model_path.lower():
                model = MobileUNet(n_classes=1, pretrained=False)
            else:
                print("Warning: Could not infer model type from filename. Defaulting to MobileUNetv2.")
                model = MobileUNetv2(n_classes=1, pretrained=False)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            return
    
    # Open Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video capture.")
        return

    # Video properties
    width_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {input_path} ({width_orig}x{height_orig} @ {fps} fps)")
    print("Press Ctrl+C to stop early.")
    
    # We will resize output to 512x512 side-by-side -> 1024x512
    out_w, out_h = 1024, 512
    
    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    frame_count = 0
    total_time = 0
    
    try:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Preprocess
                # Resize to model input (512x512)
                img = cv2.resize(frame, (512, 512))
                
                # Normalize 0-1 and tensor
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, 512, 512)
                
                # Inference
                start = time.time()
                
                if is_onnx:
                    # ONNX Inference
                    # Input must be numpy
                    ort_inputs = {input_name: img_tensor.cpu().numpy()}
                    pred_np = ort_session.run(None, ort_inputs)[0] # (1, 1, 512, 512)
                    # ONNX output is usually logits, need sigmoid? 
                    # Depends on export. PyTorch model output logits.
                    # So apply sigmoid in numpy
                    pred_np = 1 / (1 + np.exp(-pred_np))
                else:
                    # PyTorch Inference
                    pred = model(img_tensor)
                    pred = torch.sigmoid(pred)
                    pred_np = pred.squeeze().cpu().numpy()
                
                end = time.time()
                
                total_time += (end - start)
                
                # Post-process mask
                if is_onnx:
                     mask = pred_np.squeeze()
                else:
                     mask = pred_np
                
                # Create Heatmap (Green for vessel)
                # Mask is 0-1 probability. 
                # We map this to Green Intensity.
                heatmap = np.zeros_like(img, dtype=np.uint8)
                heatmap[:, :, 1] = (mask * 255).astype(np.uint8) # Green Channel
                
                # Threshold for cleaner look?
                # mask_binary = (mask > 0.5).astype(np.uint8)
                # heatmap[:, :, 1] = mask_binary * 255
                
                # Blend: Original (512x512) + Heatmap
                # addWeighted(src1, alpha, src2, beta, gamma)
                overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
                
                # Text: FPS
                inference_fps = 1.0 / (end - start + 1e-6)
                cv2.putText(overlay, f"Inf: {inference_fps:.1f} FPS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Side-by-side
                combined = np.hstack((img, overlay))
                
                out.write(combined)
                
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames", end='\r')
                    
    except KeyboardInterrupt:
        print("\nStopping early...")
                
    cap.release()
    out.release()
    print(f"\nDone! Video saved to {output_path}")
    if total_time > 0:
        print(f"Average Inference FPS: {frame_count / total_time:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input angiogram video or image')
    parser.add_argument('--output', type=str, default='demo_result.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    process_video(args.input, args.output, args.model, device)
