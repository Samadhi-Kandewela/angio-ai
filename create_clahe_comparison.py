import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_comparison():
    video_path = "input_1.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from video.")
        return

    # Resize to standard size used in pipeline
    original = cv2.resize(frame, (512, 512))
    
    # Apply exact CLAHE transform from pipeline
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # We want to show grayscale version for the paper/documentation as it looks more clinical
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # Plot side-by-side using matplotlib for nice labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Before CLAHE (Original)", fontsize=16)
    axes[0].axis("off")
    
    axes[1].imshow(enhanced_rgb)
    axes[1].set_title("After CLAHE (Enhanced)", fontsize=16)
    axes[1].axis("off")
    
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "CLAHE_Before_After.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison image to: {out_path}")

if __name__ == "__main__":
    generate_comparison()
