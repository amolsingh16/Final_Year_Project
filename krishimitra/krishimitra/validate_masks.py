"""
Step 3: Ground Truth Mask Verification Script.
Generates 3-panel visual composite previews (Original Image | Initial Mask | Color Overlay)
in 'segmentation_dataset/verification_samples/' for visual inspection.
"""

import os
import random
import cv2
import numpy as np
from segmentation_config import (
    RAW_IMAGES_DIR, INITIAL_MASKS_DIR, VERIFICATION_SAMPLES_DIR
)

def generate_verification_previews(num_samples=30):
    print("=== STEP 3: GROUND TRUTH MASK VERIFICATION ===")
    os.makedirs(VERIFICATION_SAMPLES_DIR, exist_ok=True)

    if not os.path.exists(RAW_IMAGES_DIR) or not os.path.exists(INITIAL_MASKS_DIR):
        print("[ERROR] Please run 'python generate_masks.py' first.")
        return

    images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if os.path.isfile(os.path.join(RAW_IMAGES_DIR, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])
    if not images:
        print("[ERROR] No raw images found in 'segmentation_dataset/images/'.")
        return

    # Randomly select sample images for verification
    random.seed(42)
    sample_images = random.sample(images, min(num_samples, len(images)))

    for fname in sample_images:
        base_name = os.path.splitext(fname)[0]
        mask_fname = f"{base_name}.png"

        img_path = os.path.join(RAW_IMAGES_DIR, fname)
        mask_path = os.path.join(INITIAL_MASKS_DIR, mask_fname)

        if not os.path.exists(mask_path):
            continue

        img_bgr = cv2.imread(img_path)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask_gray is None:
            continue

        # Color Mask Visualization (Green overlay on leaf)
        mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        overlay = img_bgr.copy()
        overlay[mask_gray > 0] = [0, 255, 0]  # Green overlay
        alpha = 0.35
        blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

        # Draw red boundary contours
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)

        # Combine 3 panels: [ Original | Binary Mask | Overlay ]
        composite = np.hstack([img_bgr, mask_3ch, blended])

        # Add text headers
        header = np.zeros((30, composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"Original: {fname}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(header, "Initial Binary Mask", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(header, "Green Overlay + Red Contour", (522, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        final_preview = np.vstack([header, composite])

        out_path = os.path.join(VERIFICATION_SAMPLES_DIR, f"verify_{base_name}.png")
        cv2.imwrite(out_path, final_preview)

    print(f"[SUCCESS] Saved {len(sample_images)} verification previews to '{VERIFICATION_SAMPLES_DIR}'.")
    print("Please inspect samples to verify leaf boundary accuracy before split & training.")

if __name__ == "__main__":
    generate_verification_previews()
