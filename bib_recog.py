import os
import cv2
import pytesseract
import numpy as np
from ocr import ocr
from PIL import Image

def scale_roi(x, y, w, h, image_shape):
    expansion_factor = 1.15
    new_x = int(x - (w * (expansion_factor - 1) / 2))
    new_y = int(y - (h * (expansion_factor - 1) / 2))
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)

    # Ensure the new coordinates are within the image boundaries
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    new_w = min(new_w, image_shape[1] - new_x)
    new_h = min(new_h, image_shape[0] - new_y)

    return new_x, new_y, new_w, new_h   

# Load the cascade
cascade = cv2.CascadeClassifier('cascade1/cascade.xml')

# Folder paths
input_folder = 'test_images/test_3/dagan sa kadaugan'

# Iterate through each file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        # Read the input image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        display_image = image.copy()
        if image is None:
            continue

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect objects in the image
        objects = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=7, minSize=(55, 50))
        result_image = image.copy()

        # Draw rectangles around the detected objects
        print(f"Number of objects detected in {filename}: {len(objects)}")
        for (x, y, w, h) in objects:
            if w < 100 or h < 100:
                continue
            new_x, new_y, new_w, new_h = scale_roi(x, y, w, h, gray.shape)
            roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]  # Extract the region of interest
            if new_y + new_h < gray.shape[0] - 100:  # Check if object is at the bottom (for watermarks)
                if new_x > 100 and new_x < gray.shape[1] - 100:
                    roi_image = result_image[new_y:new_y+new_h, new_x:new_x+new_w]
                    if roi_image.shape[0] > 0 and roi_image.shape[1] > 0:
                        text = ocr(roi_image)
                        if text is not None and text.isalnum():
                            cv2.rectangle(display_image, (new_x, new_y), (new_x+new_w, new_y+new_h), (255, 0, 0), 2)
                            cv2.putText(display_image, text, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (123, 255, 123), 8)
                            # print(text.split())

        # Display the result
        max_width = 1366
        max_height = 768
        if image.shape[0] > max_width or image.shape[1] > max_height:
            display_image = Image.fromarray(display_image, "RGB")
            display_image.thumbnail((max_width, max_height))
            display_image = np.array(display_image)
            # display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Object Detection', display_image)
            cv2.waitKey(0)
        # image = cv2.resize(image, (960, 540))

# Close all OpenCV windows
cv2.destroyAllWindows()