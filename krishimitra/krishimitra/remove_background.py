"""
Steps 13, 14 & 15: Background Removal, Green Region Suppression & Abnormal Region Detection Script.
Passes input leaf through U-Net leaf segmentation, sets background to pure black RGB (0, 0, 0),
suppresses healthy green pixels, and detects abnormal yellow/brown/black disease lesions.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from segmentation_config import (
    UNET_MODEL_PATH, SEGMENTATION_RESULTS_DIR, IMAGE_HEIGHT, IMAGE_WIDTH,
    HSV_GREEN_LOWER, HSV_GREEN_UPPER,
    HSV_LESION_YELLOW_BROWN_LOWER, HSV_LESION_YELLOW_BROWN_UPPER,
    HSV_LESION_DARK_LOWER, HSV_LESION_DARK_UPPER
)
from train_unet import (
    bce_dice_loss, dice_coef, iou_metric, precision_metric, recall_metric
)

# Global thread-safe lazy loader for U-Net model
_unet_model_instance = None

def get_unet_model():
    global _unet_model_instance
    if _unet_model_instance is None:
        if not os.path.exists(UNET_MODEL_PATH):
            raise FileNotFoundError(f"Trained U-Net model file not found at '{UNET_MODEL_PATH}'. Run 'python train_unet.py' first.")
        
        _unet_model_instance = tf.keras.models.load_model(
            UNET_MODEL_PATH,
            custom_objects={
                'bce_dice_loss': bce_dice_loss,
                'dice_coef': dice_coef,
                'iou_metric': iou_metric,
                'precision_metric': precision_metric,
                'recall_metric': recall_metric
            }
        )
    return _unet_model_instance


def process_leaf_pipeline(image_bgr):
    """
    Complete Segmentation Pipeline:
    1. U-Net Leaf Mask Prediction (Leaf vs Background)
    2. Background Removal (Black RGB 0,0,0)
    3. Healthy Green Suppression
    4. Abnormal Yellow/Brown/Black Lesion Mask Generation
    """
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. Resize & Normalize for U-Net
    img_resized = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    input_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # Predict U-Net Leaf Mask
    model = get_unet_model()
    pred_prob = model.predict(input_tensor, verbose=0)[0, :, :, 0]
    leaf_mask_bin = (pred_prob >= 0.5).astype(np.uint8)

    # Resize leaf mask back to original image size
    if (h, w) != (IMAGE_HEIGHT, IMAGE_WIDTH):
        leaf_mask_bin = cv2.resize(leaf_mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. STEP 13: BACKGROUND REMOVAL (Pure Black RGB 0, 0, 0)
    leaf_only_bgr = image_bgr.copy()
    leaf_only_bgr[leaf_mask_bin == 0] = [0, 0, 0]

    # 3. STEP 14 & 15: HEALTHY GREEN SUPPRESSION & ABNORMAL LESION DETECTION
    hsv_leaf = cv2.cvtColor(leaf_only_bgr, cv2.COLOR_BGR2HSV)

    # Healthy Green Mask
    lower_green = np.array(HSV_GREEN_LOWER, dtype=np.uint8)
    upper_green = np.array(HSV_GREEN_UPPER, dtype=np.uint8)
    green_mask = cv2.inRange(hsv_leaf, lower_green, upper_green)

    # Abnormal Yellow/Brown Lesions
    lower_yb = np.array(HSV_LESION_YELLOW_BROWN_LOWER, dtype=np.uint8)
    upper_yb = np.array(HSV_LESION_YELLOW_BROWN_UPPER, dtype=np.uint8)
    lesion_yb_mask = cv2.inRange(hsv_leaf, lower_yb, upper_yb)

    # Abnormal Dark/Necrotic Lesions (on leaf foreground only)
    lower_dark = np.array(HSV_LESION_DARK_LOWER, dtype=np.uint8)
    upper_dark = np.array(HSV_LESION_DARK_UPPER, dtype=np.uint8)
    lesion_dark_mask = cv2.inRange(hsv_leaf, lower_dark, upper_dark)
    lesion_dark_mask[leaf_mask_bin == 0] = 0  # Restrict to leaf interior

    # Combine abnormal lesion masks & suppress healthy green
    abnormal_mask_raw = cv2.bitwise_or(lesion_yb_mask, lesion_dark_mask)
    abnormal_mask_raw[green_mask > 0] = 0
    abnormal_mask_raw[leaf_mask_bin == 0] = 0

    # Morphological Cleanup
    kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask = cv2.morphologyEx(abnormal_mask_raw, cv2.MORPH_OPEN, kernel_op)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_cl)

    # Visualization Overlay (Red highlights on disease lesions)
    lesion_overlay = leaf_only_bgr.copy()
    lesion_overlay[disease_mask > 0] = [0, 0, 255]
    lesion_vis = cv2.addWeighted(lesion_overlay, 0.5, leaf_only_bgr, 0.5, 0)

    return {
        "leaf_mask": leaf_mask_bin * 255,
        "leaf_only_bgr": leaf_only_bgr,
        "disease_mask": disease_mask,
        "lesion_vis": lesion_vis
    }


def main():
    print("=== STEPS 13-15: BACKGROUND REMOVAL & ABNORMAL LESION DETECTION ===")
    os.makedirs(SEGMENTATION_RESULTS_DIR, exist_ok=True)

    test_img_dir = os.path.join("segmentation_dataset", "test", "images")
    if not os.path.exists(test_img_dir):
        print(f"[ERROR] Test images folder not found at '{test_img_dir}'. Run 'python split_segmentation_dataset.py' first.")
        return

    sample_files = sorted([f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])[:5]

    for fname in sample_files:
        base_name = os.path.splitext(fname)[0]
        img_path = os.path.join(test_img_dir, fname)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            continue

        res = process_leaf_pipeline(img_bgr)

        # Save output stages separately
        cv2.imwrite(os.path.join(SEGMENTATION_RESULTS_DIR, f"{base_name}_leaf_mask.png"), res["leaf_mask"])
        cv2.imwrite(os.path.join(SEGMENTATION_RESULTS_DIR, f"{base_name}_black_bg.png"), res["leaf_only_bgr"])
        cv2.imwrite(os.path.join(SEGMENTATION_RESULTS_DIR, f"{base_name}_disease_mask.png"), res["disease_mask"])
        cv2.imwrite(os.path.join(SEGMENTATION_RESULTS_DIR, f"{base_name}_lesion_overlay.png"), res["lesion_vis"])

        print(f"[SAVED] Results generated for sample: {fname}")

    print(f"[SUCCESS] Standalone pipeline completed. Outputs saved in '{SEGMENTATION_RESULTS_DIR}'.")

if __name__ == "__main__":
    main()
