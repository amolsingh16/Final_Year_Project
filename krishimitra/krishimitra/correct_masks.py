"""
Step 4: Ground Truth Mask Correction Tool.
Provides an interactive OpenCV GUI tool allowing users to paint foreground leaf pixels (white)
or erase background noise (black) and save final corrected ground-truth masks.
"""

import os
import cv2
import numpy as np
from segmentation_config import (
    RAW_IMAGES_DIR, INITIAL_MASKS_DIR, CORRECTED_MASKS_DIR
)

drawing = False
mode = True  # True = Add foreground (white), False = Erase background (black)
brush_size = 10
img_display = None
mask_edit = None

def draw_mask(event, x, y, flags, param):
    global drawing, mode, brush_size, img_display, mask_edit

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode = True
        cv2.circle(mask_edit, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = False
        cv2.circle(mask_edit, (x, y), brush_size, 0, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            val = 255 if mode else 0
            cv2.circle(mask_edit, (x, y), brush_size, val, -1)
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False

def run_correction_tool():
    global img_display, mask_edit, brush_size

    print("=== STEP 4: MASK CORRECTION TOOL ===")
    os.makedirs(CORRECTED_MASKS_DIR, exist_ok=True)

    images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if os.path.isfile(os.path.join(RAW_IMAGES_DIR, f)) and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '')])
    if not images:
        print("[ERROR] No images found. Run 'python generate_masks.py' first.")
        return

    print("Controls:")
    print("  Left Click + Drag  : Paint Leaf Foreground (White)")
    print("  Right Click + Drag : Erase Background (Black)")
    print("  '+' / '-' Keys     : Increase / Decrease Brush Size")
    print("  's' Key            : Save Corrected Mask & Next")
    print("  'n' Key            : Skip to Next Image")
    print("  'q' Key            : Quit Tool")

    cv2.namedWindow("Mask Correction Tool")
    cv2.setMouseCallback("Mask Correction Tool", draw_mask)

    for fname in images:
        base_name = os.path.splitext(fname)[0]
        mask_fname = f"{base_name}.png"

        img_path = os.path.join(RAW_IMAGES_DIR, fname)
        init_mask_path = os.path.join(INITIAL_MASKS_DIR, mask_fname)
        corr_mask_path = os.path.join(CORRECTED_MASKS_DIR, mask_fname)

        if os.path.exists(corr_mask_path):
            mask_edit = cv2.imread(corr_mask_path, cv2.IMREAD_GRAYSCALE)
        elif os.path.exists(init_mask_path):
            mask_edit = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask_edit = np.zeros((256, 256), dtype=np.uint8)

        img_bgr = cv2.imread(img_path)

        while True:
            # Create color overlay
            overlay = img_bgr.copy()
            overlay[mask_edit > 0] = [0, 255, 0]
            blended = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)

            # Combined view [ Original | Blended Mask ]
            display = np.hstack([img_bgr, blended])

            # Status bar
            status_text = f"Image: {fname} | Brush: {brush_size}px | (s: Save | n: Next | q: Quit)"
            cv2.putText(display, status_text, (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Mask Correction Tool", display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('+') or key == ord('='):
                brush_size = min(50, brush_size + 2)
            elif key == ord('-') or key == ord('_'):
                brush_size = max(2, brush_size - 2)
            elif key == ord('s'):
                cv2.imwrite(corr_mask_path, mask_edit)
                # Also update mask in INITIAL_MASKS_DIR as final ground truth
                cv2.imwrite(init_mask_path, mask_edit)
                print(f"[SAVED] Corrected mask saved: {mask_fname}")
                break
            elif key == ord('n'):
                print(f"[SKIPPED] Skipped {fname}")
                break
            elif key == ord('q'):
                print("[EXIT] Exiting Correction Tool.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("[COMPLETED] Mask correction session ended.")

if __name__ == "__main__":
    run_correction_tool()
