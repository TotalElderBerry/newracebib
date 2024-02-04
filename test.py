import cv2
import pytesseract
import numpy as np
import imutils
from PIL import Image
from pre_process import pre_image

def scale_roi(x,y,w,h):
    expansion_factor = 1.1
    new_x = int(x - (w * (expansion_factor - 1) / 2))
    new_y = int(y - (h * (expansion_factor - 1) / 2))
    new_w = int(w * expansion_factor)
    new_h = int(h * expansion_factor)

    # Ensure the new coordinates are within the image boundaries
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    new_w = min(new_w, image.shape[1] - new_x)
    new_h = min(new_h, image.shape[0] - new_y)

    return new_x,new_y,new_w,new_h

# Load the cascade
cascade = cv2.CascadeClassifier('cascade/cascade.xml')

# Read the input image
image = cv2.imread('t5.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect objects in the image
objects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(45, 45))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  # Set the path to your Tesseract executable
custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Adjust OCR settings as needed
# Adjust OCR settings as needed
# Draw rectangles around the detected objects
print(f"Number of objects: {len(objects)}")
for (x, y, w, h) in objects:
    new_x,new_y,new_w,new_h = scale_roi(x,y,w,h)
    temp_x = x
    temp_y = y
    temp_w = w
    temp_h = h
    roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]  # Extract the region of interest
    
    if temp_y+temp_h < gray.shape[0]-200: #check if object is at the bottom (for watermarks)
        roi = pre_image(roi)
        cnts = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) > 0 else cnts[1]
        
        for contour in cnts:
            for x in range(roi.shape[1]):
                if len(roi) == 0:
                    break
                if roi[0, x] < 200:
                    cv2.floodFill(roi, None, seedPoint=(x, 0), newVal=0, loDiff=3, upDiff=3)  # Fill the background with white color

                # Fill dark bottom pixels:
                if roi[-1, x] < 200:
                    cv2.floodFill(roi, None, seedPoint=(x, roi.shape[0]-1), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

                for y in range(roi.shape[0]):
                    # Fill dark left side pixels:
                    if roi[y, 0] < 200:
                        cv2.floodFill(roi, None, seedPoint=(0, y), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

                    # Fill dark right side pixels:
                    if roi[y, 0] < 200:
                        cv2.floodFill(roi, None, seedPoint=(gray.shape[1]-1, y), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

            # get rectangle bounding contour
            [contour_x, contour_y, contour_w, contour_h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if contour_w < 50 and contour_h < 50:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(roi, (contour_x, contour_y), (contour_x + contour_w, contour_y + contour_h), (255, 0, 255), 2)

    text = pytesseract.image_to_string(roi,lang="eng",config=custom_config)  # Perform OCR on the ROI
    print(f"Detected text: {text}")
    cv2.rectangle(image, (temp_x, temp_y), (temp_x+temp_w, temp_y+temp_h), (255, 0, 0), 2)
    cv2.putText(image, text, (temp_x, temp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow('Object Detection', roi)

# Display the result
# image = cv2.resize(image, (960, 540)) 
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

