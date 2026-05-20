import cv2
import glob
import numpy as np
import os

def create_video_from_images(image_pattern, output_file, duration=5, fps=30):
    # Find images
    images = glob.glob(image_pattern)
    if not images:
        print(f"No images found matching {image_pattern}")
        # Create a dummy noise video if no images found
        print("Creating dummy noise video instead...")
        frames = []
        for _ in range(duration * fps):
            frames.append(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    else:
        print(f"Found {len(images)} images: {images}")
        # Load images
        loaded_imgs = [cv2.imread(img) for img in images]
        
        # Resize to same size (512x512)
        loaded_imgs = [cv2.resize(img, (512, 512)) for img in loaded_imgs]
        
        frames = []
        total_frames = duration * fps
        
        # Create a "slideshow" effect
        frames_per_img = total_frames // len(loaded_imgs)
        
        for img in loaded_imgs:
            for _ in range(frames_per_img):
                # Add slight noise to simulate video "movement"
                noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
                noisy_img = cv2.add(img, noise)
                frames.append(noisy_img)
                
    # Write video
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()
    print(f"Created {output_file} ({duration}s, {fps}fps)")

if __name__ == "__main__":
    # Look for the visualization images we created earlier
    create_video_from_images('vis_*.png', 'demo.mp4')
