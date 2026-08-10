"""
Steps 11 & 12: Segmentation Evaluation & Visualization Script.
Evaluates 'unet_leaf_segmentation.h5' strictly on the independent TEST split.
Calculates exact pixel-level confusion matrix counts (TP, TN, FP, FN) and prints
un-fabricated IoU, Dice, Precision, and Recall metrics.
Saves 4-panel visual comparison overlays to 'segmentation_results/'.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from segmentation_config import (
    TEST_DIR, UNET_MODEL_PATH, SEGMENTATION_RESULTS_DIR,
    IMAGE_HEIGHT, IMAGE_WIDTH
)
from train_unet import (
    bce_dice_loss, dice_coef, iou_metric, precision_metric, recall_metric
)

def evaluate_test_set():
    print("=== STEPS 11 & 12: SEGMENTATION EVALUATION & VISUALIZATION ===")

    if not os.path.exists(UNET_MODEL_PATH):
        print(f"[ERROR] Trained U-Net model not found at '{UNET_MODEL_PATH}'. Run 'python train_unet.py' first.")
        return

    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] Test set folder not found at '{TEST_DIR}'. Run 'python split_segmentation_dataset.py' first.")
        return

    os.makedirs(SEGMENTATION_RESULTS_DIR, exist_ok=True)

    # Load Trained U-Net Model
    model = tf.keras.models.load_model(
        UNET_MODEL_PATH,
        custom_objects={
            'bce_dice_loss': bce_dice_loss,
            'dice_coef': dice_coef,
            'iou_metric': iou_metric,
            'precision_metric': precision_metric,
            'recall_metric': recall_metric
        }
    )

    test_img_dir = os.path.join(TEST_DIR, "images")
    test_mask_dir = os.path.join(TEST_DIR, "masks")

    test_files = sorted([f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])
    if not test_files:
        print("[ERROR] No test images found.")
        return

    print(f"Evaluating U-Net model on {len(test_files)} independent test samples...")

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    sample_viz_count = 0
    max_samples_to_save = 20

    for idx, fname in enumerate(test_files):
        base_name = os.path.splitext(fname)[0]
        mask_fname = f"{base_name}.png"

        img_p = os.path.join(test_img_dir, fname)
        mask_p = os.path.join(test_mask_dir, mask_fname)

        img_bgr = cv2.imread(img_p)
        mask_gt_gray = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask_gt_gray is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            img_rgb = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
        if mask_gt_gray.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            mask_gt_gray = cv2.resize(mask_gt_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        # Preprocess input tensor
        input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

        # Predict probability mask
        pred_prob = model.predict(input_tensor, verbose=0)[0, :, :, 0]

        # Binarize GT & Pred masks at 0.5 threshold
        gt_bin = (mask_gt_gray > 127).astype(np.uint8)
        pred_bin = (pred_prob >= 0.5).astype(np.uint8)

        # Pixel Confusion Matrix Counts
        tp = np.sum((gt_bin == 1) & (pred_bin == 1))
        tn = np.sum((gt_bin == 0) & (pred_bin == 0))
        fp = np.sum((gt_bin == 0) & (pred_bin == 1))
        fn = np.sum((gt_bin == 1) & (pred_bin == 0))

        total_tp += int(tp)
        total_tn += int(tn)
        total_fp += int(fp)
        total_fn += int(fn)

        # Save Visual Evaluations for first N samples
        if sample_viz_count < max_samples_to_save:
            gt_3ch = cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_3ch = cv2.cvtColor((pred_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Color Overlay (Green = Prediction, Red Contour = GT boundary)
            overlay = img_bgr.copy()
            overlay[pred_bin > 0] = [0, 255, 0]
            blended = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
            gt_contours, _ = cv2.findContours(gt_bin * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, gt_contours, -1, (0, 0, 255), 2)

            # 4-Panel Visualization: [ Original | GT Mask | Pred Mask | Overlay ]
            four_panel = np.hstack([img_bgr, gt_3ch, pred_3ch, blended])

            # Top Header Bar
            header = np.zeros((30, four_panel.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, f"Sample: {base_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(header, "Ground Truth Mask", (266, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(header, "U-Net Predicted Mask", (522, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(header, "Green Pred Overlay", (778, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            final_viz = np.vstack([header, four_panel])
            cv2.imwrite(os.path.join(SEGMENTATION_RESULTS_DIR, f"{base_name}_eval.png"), final_viz)
            sample_viz_count += 1

    # Un-fabricated Metrics Computation
    eps = 1e-6
    precision = float(total_tp) / (total_tp + total_fp + eps)
    recall = float(total_tp) / (total_tp + total_fn + eps)
    iou = float(total_tp) / (total_tp + total_fp + total_fn + eps)
    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)

    print("\n=========================================================")
    print("         U-NET INDEPENDENT TEST EVALUATION METRICS       ")
    print("=========================================================")
    print(f"  Test Samples Evaluated : {len(test_files)}")
    print(f"  Total True Positive  (TP) pixels : {total_tp:,}")
    print(f"  Total True Negative  (TN) pixels : {total_tn:,}")
    print(f"  Total False Positive (FP) pixels : {total_fp:,}")
    print(f"  Total False Negative (FN) pixels : {total_fn:,}")
    print("---------------------------------------------------------")
    print(f"  Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  IoU       : {iou:.4f} ({iou * 100:.2f}%)")
    print(f"  Dice Coef : {dice:.4f} ({dice * 100:.2f}%)")
    print("=========================================================")
    print(f"\n[SUCCESS] Visual evaluation previews saved to '{SEGMENTATION_RESULTS_DIR}'.")

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn
    }

if __name__ == "__main__":
    evaluate_test_set()
