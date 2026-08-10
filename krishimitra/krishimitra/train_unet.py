"""
Steps 6-10: U-Net Model Implementation, Custom Loss, Metrics, Data Generators & Training.
Trains U-Net for binary leaf vs background segmentation on 256x256 images.
Saves best model weights as 'unet_leaf_segmentation.h5' and training curves as 'training_history.png'.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

from segmentation_config import (
    TRAIN_DIR, VAL_DIR, UNET_MODEL_PATH,
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS
)

# ---------------------------------------------------------
# Custom Loss Functions & Metrics
# ---------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_mean = K.mean(bce)
    d_loss = dice_loss(y_true, y_pred)
    return bce_mean + d_loss

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1.0 - y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + fp + smooth)

def recall_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1.0 - y_pred_f))
    return (tp + smooth) / (tp + fn + smooth)

# ---------------------------------------------------------
# Data Generator Sequence with Training Augmentation Only
# ---------------------------------------------------------

class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=16, augment=False, shuffle=True):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        
        self.file_names = sorted([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_names))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_files = [self.file_names[i] for i in batch_indices]

        X = np.zeros((len(batch_files), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
        Y = np.zeros((len(batch_files), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

        for i, fname in enumerate(batch_files):
            base_name = os.path.splitext(fname)[0]
            mask_fname = f"{base_name}.png"

            img_p = os.path.join(self.img_dir, fname)
            mask_p = os.path.join(self.mask_dir, mask_fname)

            img_bgr = cv2.imread(img_p)
            mask_gray = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            if img_bgr is None or mask_gray is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if img_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                img_rgb = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
            if mask_gray.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                mask_gray = cv2.resize(mask_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

            # Preprocessing: Scale image to [0, 1] float32, threshold mask to binary [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0
            mask_binary = (mask_gray > 127).astype(np.float32)[:, :, np.newaxis]

            # Augmentation (Training split ONLY)
            if self.augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    img_normalized = np.fliplr(img_normalized)
                    mask_binary = np.fliplr(mask_binary)
                # Random vertical flip
                if np.random.rand() > 0.5:
                    img_normalized = np.flipud(img_normalized)
                    mask_binary = np.flipud(mask_binary)
                # Slight brightness variation (+/- 10%)
                if np.random.rand() > 0.5:
                    factor = np.random.uniform(0.9, 1.1)
                    img_normalized = np.clip(img_normalized * factor, 0.0, 1.0)

            X[i] = img_normalized
            Y[i] = mask_binary

        return X, Y

# ---------------------------------------------------------
# U-Net Architecture Construction
# ---------------------------------------------------------

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder Block 1
    c1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32, (3, 3), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Encoder Block 2
    c2 = layers.Conv2D(64, (3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(64, (3, 3), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Encoder Block 3
    c3 = layers.Conv2D(128, (3, 3), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(128, (3, 3), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Encoder Block 4
    c4 = layers.Conv2D(256, (3, 3), padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(256, (3, 3), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    b = layers.Conv2D(512, (3, 3), padding='same')(p4)
    b = layers.BatchNormalization()(b)
    b = layers.Activation('relu')(b)
    b = layers.Conv2D(512, (3, 3), padding='same')(b)
    b = layers.BatchNormalization()(b)
    b = layers.Activation('relu')(b)

    # Decoder Block 4
    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b)
    u4 = layers.concatenate([u4, c4])
    c5 = layers.Conv2D(256, (3, 3), padding='same')(u4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(256, (3, 3), padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    # Decoder Block 3
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = layers.Conv2D(128, (3, 3), padding='same')(u3)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(128, (3, 3), padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    # Decoder Block 2
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = layers.Conv2D(64, (3, 3), padding='same')(u2)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Conv2D(64, (3, 3), padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)

    # Decoder Block 1
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = layers.Conv2D(32, (3, 3), padding='same')(u1)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Conv2D(32, (3, 3), padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)

    # Sigmoid Output Layer for Binary Segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c8)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="U-Net_Leaf_Segmentation")
    return model

# ---------------------------------------------------------
# Training Main Execution
# ---------------------------------------------------------

def main():
    print("=== STEPS 6-10: U-NET LEAF SEGMENTATION TRAINING ===")

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("[ERROR] Train/Val split folders not found. Run 'python split_segmentation_dataset.py' first.")
        return

    # Data Generators
    train_gen = SegmentationDataGenerator(TRAIN_DIR, batch_size=BATCH_SIZE, augment=True, shuffle=True)
    val_gen = SegmentationDataGenerator(VAL_DIR, batch_size=BATCH_SIZE, augment=False, shuffle=False)

    print(f"Train Batches: {len(train_gen)}, Val Batches: {len(val_gen)}")

    # Model Construction & Compilation
    model = build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_metric, precision_metric, recall_metric]
    )

    print("\nU-Net Model Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(UNET_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Training Execution
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save Training History Curves Plot
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train BCE+Dice Loss', color='navy', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val BCE+Dice Loss', color='crimson', linewidth=2, linestyle='--')
    plt.title('U-Net Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Dice Metric Curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], label='Train Dice Coef', color='forestgreen', linewidth=2)
    plt.plot(history.history['val_dice_coef'], label='Val Dice Coef', color='darkorange', linewidth=2, linestyle='--')
    plt.title('U-Net Dice Coefficient Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_plot_path = "training_history.png"
    plt.savefig(hist_plot_path, dpi=200)
    plt.close()

    print(f"\n[SUCCESS] Best U-Net Model saved to: '{UNET_MODEL_PATH}'")
    print(f"[SUCCESS] Training history plot saved to: '{hist_plot_path}'")
    print("NOTE: 'plant_disease_model.h5' was left 100% untouched.")

if __name__ == "__main__":
    main()
