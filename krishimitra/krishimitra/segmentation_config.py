"""
Configuration parameters for Supervised U-Net Leaf Segmentation Module.
Centralized thresholds and parameters to prevent hardcoded duplication across files.
"""

import os
from pathlib import Path
from django.conf import settings

# Base directory setup
BASE_DIR = Path(__file__).resolve().parent

# Dataset Paths
SEGMENTATION_DATASET_DIR = os.path.join(BASE_DIR, "segmentation_dataset")
RAW_IMAGES_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "images")
INITIAL_MASKS_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "masks")
CORRECTED_MASKS_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "corrected_masks")
VERIFICATION_SAMPLES_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "verification_samples")

TRAIN_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "train")
VAL_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "validation")
TEST_DIR = os.path.join(SEGMENTATION_DATASET_DIR, "test")

SEGMENTATION_RESULTS_DIR = os.path.join(BASE_DIR, "segmentation_results")

# Model File Path
UNET_MODEL_PATH = os.path.join(BASE_DIR, "unet_leaf_segmentation.h5")

# Sub-sampling Parameters
IMAGES_PER_CLASS = 300  # 300 images per class * 10 classes = 3,000 balanced images

# U-Net Model Hyperparameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20

# Data Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# HSV Healthy Green Suppression Thresholds
HSV_GREEN_LOWER = [30, 30, 30]
HSV_GREEN_UPPER = [85, 255, 255]

# HSV Abnormal Lesion Thresholds (Yellow, Brown, Dark/Necrotic)
HSV_LESION_YELLOW_BROWN_LOWER = [10, 30, 30]
HSV_LESION_YELLOW_BROWN_UPPER = [35, 255, 255]

HSV_LESION_DARK_LOWER = [0, 0, 0]
HSV_LESION_DARK_UPPER = [180, 255, 65]
