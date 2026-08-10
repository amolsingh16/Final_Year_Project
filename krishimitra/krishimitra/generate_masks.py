"""
Step 2: Automatic Initial Pseudo-Mask Generation Script.
Samples a balanced 3,000-image subset (300 per class) from the original dataset
and generates initial binary leaf pseudo-masks using OpenCV & GrabCut.

Note: These are initial pseudo-masks and are not claimed to be perfect manually annotated ground-truth masks.
"""

import os
import shutil
import cv2
import numpy as np
from segmentation_config import (
    RAW_IMAGES_DIR, INITIAL_MASKS_DIR, IMAGES_PER_CLASS
)

def create_initial_leaf_mask(image_bgr):
    """
    Applies HSV thresholding, GrabCut, morphological operations, and largest contour
    filtering to generate a binary leaf pseudo-mask (0=background, 255=leaf).
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Range 1: Green leaf pixels
    lower_green = np.array([25, 25, 25])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Range 2: Yellowish/brownish diseased leaf pixels
    lower_yellow = np.array([10, 25, 25])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combined initial threshold mask
    initial_mask = cv2.bitwise_or(mask_green, mask_yellow)

    # Apply GrabCut refinement if threshold mask is non-empty
    if np.sum(initial_mask) > 100:
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[initial_mask > 0] = cv2.GC_PR_FGD
        gc_mask[initial_mask == 0] = cv2.GC_BGD

        # Bounding box around initial threshold
        coords = cv2.findNonZero(initial_mask)
        if coords is not None:
            x, y, w_box, h_box = cv2.boundingRect(coords)
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w_box = min(w - x, w_box + 2 * margin)
            h_box = min(h - y, h_box + 2 * margin)
            
            rect = (x, y, w_box, h_box)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(image_bgr, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
                bin_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
            except Exception:
                bin_mask = initial_mask
        else:
            bin_mask = initial_mask
    else:
        bin_mask = initial_mask

    # Morphological Cleanup: Fill interior leaf holes and remove small specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    # Largest contour retention (main leaf structure)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        # Retain contours with area > 1% of total image area
        min_area = 0.01 * (h * w)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if valid_contours:
            cv2.drawContours(final_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        else:
            c_max = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [c_max], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = cleaned

    return final_mask


def main():
    print("=== STEP 2: AUTOMATIC INITIAL PSEUDO-MASK GENERATION ===")
    os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
    os.makedirs(INITIAL_MASKS_DIR, exist_ok=True)

    dataset_root = "dataset"
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    classes = sorted([c for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c))])
    total_sampled = 0

    for c in classes:
        c_train = os.path.join(train_dir, c)
        c_test = os.path.join(test_dir, c)

        def get_valid_image_paths(folder_path):
            if not os.path.exists(folder_path):
                return []
            valid_paths = []
            for f in sorted(os.listdir(folder_path)):
                p = os.path.join(folder_path, f)
                if os.path.isfile(p):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ('.jpg', '.jpeg', '.png', ''):
                        valid_paths.append(p)
            return valid_paths

        train_files = get_valid_image_paths(c_train)
        test_files = get_valid_image_paths(c_test)
        all_class_files = train_files + test_files

        # Sample 300 images per class
        sampled_files = all_class_files[:IMAGES_PER_CLASS]
        print(f"Class '{c}': Selected {len(sampled_files)} images for segmentation subset.")

        for idx, src_path in enumerate(sampled_files):
            fname = f"{c}_{idx:04d}.jpg"
            mask_fname = f"{c}_{idx:04d}.png"

            dest_img_path = os.path.join(RAW_IMAGES_DIR, fname)
            dest_mask_path = os.path.join(INITIAL_MASKS_DIR, mask_fname)

            # Read image
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                continue

            # Ensure 256x256 resize
            if img_bgr.shape[:2] != (256, 256):
                img_bgr = cv2.resize(img_bgr, (256, 256))

            # Save raw image copy
            cv2.imwrite(dest_img_path, img_bgr)

            # Generate initial pseudo-mask
            mask = create_initial_leaf_mask(img_bgr)
            cv2.imwrite(dest_mask_path, mask)

            total_sampled += 1

    print(f"\n[SUCCESS] Generated {total_sampled} image-mask pairs in 'segmentation_dataset/'.")
    print("NOTE: These automatically generated masks are INITIAL PSEUDO-MASKS for verification & correction.")

if __name__ == "__main__":
    main()
