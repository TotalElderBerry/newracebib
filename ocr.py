import cv2
import numpy as np
from PIL import Image
from pre_process import pre_image, deskew, crop, scale_roi
import pytesseract
import os

kernel = np.ones((3,3), np.uint8)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  # Set the path to your Tesseract executable
custom_config = r'-psm 11 -c tessedit_char_whitelist=0123456789'  # Adjust OCR settings as needed

def is_dark(image_path, threshold=100):
    # Calculate the mean pixel intensity
    mean_intensity = cv2.mean(image_path)[0]

    # Check if the image is dark based on the threshold
    return mean_intensity < threshold

def is_small(image):
    return image.shape[0] <= 100 and image.shape[2] <= 100


def is_inside_contour(cnts, prev, current):
    first_x, first_y, first_w, first_h = cv2.boundingRect(cnts[prev])
    second_x, second_y, second_w, second_h = cv2.boundingRect(cnts[current])
    
    return second_x > first_x and second_y > first_y and \
           (second_x + second_w) < (first_x + first_w) and \
           (second_y + second_h) < (first_y + first_h)


def ocr(roi):
    final_text = ""
    # print(f"image width: {roi.shape[0]} image height: {roi.shape[1]}")

    is_image_small = is_small(roi)
    image = cv2.resize(roi,(300,250))
    isdark = is_dark(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma = 1.1
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255
        for i in np.arange(0,256)]).astype("uint8")
    gray = cv2.LUT(gray,table)

    skewed_image = deskew(gray)
    if skewed_image is None:
        return
    scale = cv2.resize(skewed_image,None, fx= 1.5, fy= 1.5, interpolation=cv2.INTER_AREA)

    scale = crop(scale)

    image = scale.copy()
    # scale = cv2.cvtColor(scale, cv2.COLOR_BGR2GRAY)
    # if isdark:
    #     _, scale = cv2.threshold(scale, 127, 255, cv2.THRESH_BINARY)
    # else:
    scale = cv2.medianBlur(scale, 5)
    scale = cv2.dilate(scale, kernel, iterations=1)
    # scale = cv2.erode(scale, kernel, iterations=2)
    if is_image_small:
        scale = cv2.Canny(scale, 100,200)
    else:
        scale = cv2.Canny(scale, 90,170)
    scale = cv2.morphologyEx(scale, cv2.MORPH_CLOSE, kernel)
    scale = cv2.GaussianBlur(scale,(3,3),0)
    scale = cv2.threshold(scale, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        
    cnts = cv2.findContours(scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) > 0 else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    i = 0
    prev=curr = -1
    for contour in cnts:
        x,y,w,h = cv2.boundingRect(contour)
        tempx,tempy,tempw,temph = cv2.boundingRect(cnts[i])
        i = i+1
        
        if w < 40 and h < 40:
            continue
        if w > h:
            continue
        if y > 200:
            continue

        if x > 0 and y > 0 and y+h != scale.shape[0] and w > 10 and h > 10:
            if prev == -1 and curr == -1:
                curr = i-1
            else:
                prev = curr
                curr = i-1
           
            is_inside_contour_value = is_inside_contour(cnts,prev,curr)
            if is_inside_contour_value:
                continue
            cv2.rectangle(scale,(x,y),(x+w,y+h),(123,0,255),2)
            x,y,w,h = scale_roi(x,y,w,h,scale)
            
            #
            roi = image[y:y+h, x:x+w]
            #PREPROCESS ROUND 2
            roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            roi = cv2.erode(roi, kernel, iterations=1)
            roi = cv2.medianBlur(roi,3)

            # roi = cv2.Canny(roi, 128,255)
            # roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
            new_width, new_height = 200, 200  # Adjust these dimensions as needed

            # Create a blank white image as the new background
            background = np.ones((new_height, new_width), dtype=np.uint8) * 0

            # Calculate the position to place the ROI in the center of the new background
            x_offset = (new_width - roi.shape[1]) // 2
            y_offset = (new_height - roi.shape[0]) // 2

            # Copy the binary ROI onto the new background
            if roi.shape[0] <= new_width and roi.shape[1] <= new_height:
                background[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi
                padding = 2
                cv2.rectangle(background, (x_offset - padding, y_offset - padding),(x_offset + roi.shape[1] + padding, y_offset + roi.shape[0] + padding),(123, 255, 0), 2)
                background = cv2.resize(background, None, fx=1.1,fy=1.1,interpolation=cv2.INTER_CUBIC)
            # background = deskew(background)

            
                background = cv2.erode(background,kernel,iterations=1)
                # closing
                # background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
                # background = cv2.bitwise_not(background)
                # background = cv2.GaussianBlur(background,(3,3),0)
                # background = image[y:y+h, x:x+w]
                text = pytesseract.image_to_string(background,lang="eng",config=custom_config)  # Perform OCR on the ROI
                text = text.strip()
                if not text.isalnum():
                    for x in range(roi.shape[1]):
                        if len(background) == 0:
                            break
                        if background[0, x] < 201:
                            cv2.floodFill(background, None, seedPoint=(x, 0), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color
                        # Fill dark bottom pixels:
                        if background[-1, x] < 201:
                            cv2.floodFill(background, None, seedPoint=(x, background.shape[0]-1), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color
                        for y in range(roi.shape[0]):
                            # Fill dark left side pixels:
                            if roi[y, 0] < 201 and y < new_height:
                                cv2.floodFill(background, None, seedPoint=(0, y), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color
            text = pytesseract.image_to_string(background,lang="eng",config=custom_config)  # Perform OCR on the ROI
            text = text.strip()
            # print(f"Detected text: {text}")
            # Offset for the text position
            text_offset_x, text_offset_y = 10, 10  # Adjust these values as needed

            # Add the offset to the x and y positions
            text_x = x_offset + text_offset_x
            text_y = y_offset + text_offset_y

            # Draw the text on the background
            # cv2.putText(background, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123, 255, 123), 2)
            # cv2.putText(background, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow('Bib', background)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            final_text = final_text + text 

    # cv2.imshow('Bib', scale)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_text

