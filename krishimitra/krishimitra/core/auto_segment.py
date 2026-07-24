import cv2
import os
import numpy as np

input_folder = "dataset/train"
output_folder = "segmented_images"

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for file in files:

        if file.endswith(".jpg") or file.endswith(".png"):

            path = os.path.join(root, file)

            img = cv2.imread(path)

            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # blur for noise removal
            blur = cv2.GaussianBlur(gray, (5,5),0)

            # thresholding
            _, thresh = cv2.threshold(blur, 120,255,cv2.THRESH_BINARY_INV)

            # find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:

                # largest contour = leaf
                c = max(contours, key=cv2.contourArea)

                mask = np.zeros(gray.shape, np.uint8)

                cv2.drawContours(mask,[c],-1,255,-1)

                segmented = cv2.bitwise_and(img,img,mask=mask)

                save_path = os.path.join(output_folder,file)

                cv2.imwrite(save_path,segmented)

print("Segmentation Completed")