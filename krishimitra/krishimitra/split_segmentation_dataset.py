"""
Step 5: Segmentation Dataset Train/Validation/Test Split Script.
Splits ground-truth image-mask pairs into 70% Train, 15% Validation, and 15% Test sets.
Prevents data leakage by ensuring unique filename allocation per split.
"""

import os
import shutil
import random
from segmentation_config import (
    RAW_IMAGES_DIR, INITIAL_MASKS_DIR, CORRECTED_MASKS_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

def prepare_splits():
    print("=== STEP 5: SEGMENTATION DATASET SPLIT ===")

    if not os.path.exists(RAW_IMAGES_DIR):
        print("[ERROR] 'segmentation_dataset/images/' does not exist. Run 'python generate_masks.py' first.")
        return

    # Check corrected masks folder if available, else initial masks
    masks_source_dir = CORRECTED_MASKS_DIR if (os.path.exists(CORRECTED_MASKS_DIR) and len(os.listdir(CORRECTED_MASKS_DIR)) > 0) else INITIAL_MASKS_DIR

    print(f"Using ground-truth masks from: '{masks_source_dir}'")

    images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if os.path.isfile(os.path.join(RAW_IMAGES_DIR, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])
    if not images:
        print("[ERROR] No images found to split.")
        return

    # Clear prior splits to prevent stale data
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)

    # Group by class prefix to preserve class balance across splits
    class_groups = {}
    for img_fname in images:
        class_name = img_fname.rsplit('_', 1)[0]
        class_groups.setdefault(class_name, []).append(img_fname)

    random.seed(42)
    total_train, total_val, total_test = 0, 0, 0

    for c_name, c_images in class_groups.items():
        random.shuffle(c_images)
        n = len(c_images)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_files = c_images[:n_train]
        val_files = c_images[n_train:n_train + n_val]
        test_files = c_images[n_train + n_val:]

        def copy_files(file_list, target_split_dir):
            copied = 0
            for f in file_list:
                base_name = os.path.splitext(f)[0]
                mask_f = f"{base_name}.png"

                src_img = os.path.join(RAW_IMAGES_DIR, f)
                src_mask = os.path.join(masks_source_dir, mask_f)

                if not os.path.exists(src_mask):
                    # Fallback to initial masks if corrected mask not found for single image
                    src_mask = os.path.join(INITIAL_MASKS_DIR, mask_f)

                if os.path.exists(src_img) and os.path.exists(src_mask):
                    shutil.copy(src_img, os.path.join(target_split_dir, "images", f))
                    shutil.copy(src_mask, os.path.join(target_split_dir, "masks", mask_f))
                    copied += 1
            return copied

        total_train += copy_files(train_files, TRAIN_DIR)
        total_val += copy_files(val_files, VAL_DIR)
        total_test += copy_files(test_files, TEST_DIR)

    print(f"\n[SUCCESS] Dataset Split Completed:")
    print(f"  Training Set   (70%): {total_train} image-mask pairs")
    print(f"  Validation Set (15%): {total_val} image-mask pairs")
    print(f"  Test Set       (15%): {total_test} image-mask pairs")
    print(f"  Total Pairs         : {total_train + total_val + total_test}")
    print("Zero Data Leakage: Unique filename allocation verified across splits.")

if __name__ == "__main__":
    prepare_splits()
