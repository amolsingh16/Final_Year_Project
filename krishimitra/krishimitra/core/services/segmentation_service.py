"""
Django Integration Service: Supervised U-Net Leaf Segmentation & Lesion Detection.
Handles thread-safe U-Net inference, background removal, healthy green suppression,
and disease lesion mask generation for the Django web request pipeline.
"""

import os
import uuid
import threading
import numpy as np
import cv2
from PIL import Image
from django.conf import settings

from segmentation_config import (
    UNET_MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH,
    HSV_GREEN_LOWER, HSV_GREEN_UPPER,
    HSV_LESION_YELLOW_BROWN_LOWER, HSV_LESION_YELLOW_BROWN_UPPER,
    HSV_LESION_DARK_LOWER, HSV_LESION_DARK_UPPER
)
from train_unet import (
    bce_dice_loss, dice_coef, iou_metric, precision_metric, recall_metric
)

_unet_lock = threading.Lock()
_django_unet_model = None

def get_django_unet_model():
    global _django_unet_model
    if _django_unet_model is None:
        with _unet_lock:
            if _django_unet_model is None:
                if not os.path.exists(UNET_MODEL_PATH):
                    return None
                import tensorflow as tf
                _django_unet_model = tf.keras.models.load_model(
                    UNET_MODEL_PATH,
                    custom_objects={
                        'bce_dice_loss': bce_dice_loss,
                        'dice_coef': dice_coef,
                        'iou_metric': iou_metric,
                        'precision_metric': precision_metric,
                        'recall_metric': recall_metric
                    }
                )
    return _django_unet_model


def process_leaf_segmentation(uploaded_file):
    """
    Django Service wrapper for U-Net Leaf Segmentation & Lesion Extraction.
    Returns relative media URLs for:
    - leaf_mask_url
    - black_bg_url
    - disease_mask_url
    Returns None values gracefully if model missing or on processing error.
    """
    try:
        model = get_django_unet_model()
        if model is None:
            return {
                "leaf_mask_url": None,
                "black_bg_url": None,
                "disease_mask_url": None
            }

        # Read image from stream
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        uploaded_file.seek(0)

        if img_bgr is None:
            return {"leaf_mask_url": None, "black_bg_url": None, "disease_mask_url": None}

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Resize & Normalize for U-Net
        img_resized = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
        input_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

        # Predict Leaf Mask
        pred_prob = model.predict(input_tensor, verbose=0)[0, :, :, 0]
        leaf_mask_bin = (pred_prob >= 0.5).astype(np.uint8)

        if (h, w) != (IMAGE_HEIGHT, IMAGE_WIDTH):
            leaf_mask_bin = cv2.resize(leaf_mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Background Removal (Black RGB 0, 0, 0)
        black_bg_bgr = img_bgr.copy()
        black_bg_bgr[leaf_mask_bin == 0] = [0, 0, 0]

        # 3. Healthy Green Suppression & Lesion Detection
        hsv_leaf = cv2.cvtColor(black_bg_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_leaf, np.array(HSV_GREEN_LOWER), np.array(HSV_GREEN_UPPER))

        lesion_yb = cv2.inRange(hsv_leaf, np.array(HSV_LESION_YELLOW_BROWN_LOWER), np.array(HSV_LESION_YELLOW_BROWN_UPPER))
        lesion_dark = cv2.inRange(hsv_leaf, np.array(HSV_LESION_DARK_LOWER), np.array(HSV_LESION_DARK_UPPER))
        lesion_dark[leaf_mask_bin == 0] = 0

        abnormal_raw = cv2.bitwise_or(lesion_yb, lesion_dark)
        abnormal_raw[green_mask > 0] = 0
        abnormal_raw[leaf_mask_bin == 0] = 0

        kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disease_mask = cv2.morphologyEx(abnormal_raw, cv2.MORPH_OPEN, kernel_op)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_cl)

        # Save output images to media/uploads/segmented/
        unique_id = uuid.uuid4().hex[:10]
        seg_dir = os.path.join(settings.MEDIA_ROOT, "uploads", "segmented")
        os.makedirs(seg_dir, exist_ok=True)

        leaf_mask_fn = f"unet_mask_{unique_id}.png"
        black_bg_fn = f"black_bg_{unique_id}.jpg"
        disease_mask_fn = f"disease_mask_{unique_id}.png"

        cv2.imwrite(os.path.join(seg_dir, leaf_mask_fn), leaf_mask_bin * 255)
        cv2.imwrite(os.path.join(seg_dir, black_bg_fn), black_bg_bgr)
        cv2.imwrite(os.path.join(seg_dir, disease_mask_fn), disease_mask)

        return {
            "leaf_mask_url": f"uploads/segmented/{leaf_mask_fn}",
            "black_bg_url": f"uploads/segmented/{black_bg_fn}",
            "disease_mask_url": f"uploads/segmented/{disease_mask_fn}"
        }

    except Exception:
        return {
            "leaf_mask_url": None,
            "black_bg_url": None,
            "disease_mask_url": None
        }
