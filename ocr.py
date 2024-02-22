import cv2
import numpy as np
from PIL import Image
from pre_process import pre_image, deskew, crop, scale_roi
import pytesseract
import os

kernel = np.ones((3,3), np.uint8)
temp_kernel = np.ones((9,9), np.uint8)

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


def convert_bg(roi,color):
    new_width, new_height = roi.shape[1]*2, roi.shape[0]*2  # Adjust these dimensions as needed
    # Create a blank white image as the new background
    background = np.ones((new_height, new_width), dtype=np.uint8) * color

    # Calculate the position to place the ROI in the center of the new background
    x_offset = (new_width - roi.shape[1]) // 2
    y_offset = (new_height - roi.shape[0]) // 2

    # Copy the binary ROI onto the new background
    background[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi
    background = cv2.resize(background, None, fx=1.2,fy=1.2,interpolation=cv2.INTER_CUBIC)

    background = cv2.erode(background,kernel,iterations=1)
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, temp_kernel)
    return background

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
    
    scale = cv2.medianBlur(scale, 5)
    scale = cv2.dilate(scale, kernel, iterations=1)
    # scale = cv2.erode(scale, kernel, iterations=2)
    if is_image_small:
        scale = cv2.Canny(scale, 100,200)
    else:
        scale = cv2.Canny(scale, 90,130)
    scale = cv2.GaussianBlur(scale,(5,5),0)
    scale = cv2.threshold(scale, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # scale = cv2.morphologyEx(scale, cv2.MORPH_CLOSE, kernel)
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
            # roi = cv2.medianBlur(roi, 5)
            # roi = deskew(roi)
            roi = cv2.GaussianBlur(roi,(5,5),0)
            roi = cv2.resize(roi, None, fx=1.15,fy=1.15,interpolation=cv2.INTER_CUBIC)
            # roi = cv2.erode(roi, kernel, iterations=1)
            roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, temp_kernel)
            roi = cv2.dilate(roi, kernel, iterations=3)

            # roi = cv2.resize(roi, None, fx=1.15,fy=1.15,interpolation=cv2.INTER_CUBIC)
            # roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # roi = cv2.erode(roi, kernel, iterations=1)
            # roi = cv2.medianBlur(roi,3)
            # roi = cv2.GaussianBlur(roi,(3,3),0)
            # roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, temp_kernel)
            roi = cv2.bitwise_not(roi)
            
            # Create a blank white image as the new background
            background = convert_bg(roi,255)
            text = pytesseract.image_to_string(background,lang="eng",config=custom_config)  # Perform OCR on the ROI
            text = text.strip()
            # print(f"first preprocessed: {text}")
            # cv2.imshow('Bib', background)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            if not text.isalnum() or text.isalpha():

                background = convert_bg(roi,0)
                
                text = pytesseract.image_to_string(background,lang="eng",config=custom_config)  # Perform OCR on the ROI
                text = text.strip().replace(".","").replace(",","").replace("/","").replace("‘","")
                if text == 'I':
                    text = '1'
                if text == 'O':
                    text = '0'
                
                # print(text)
            if text.isnumeric():
                final_text = final_text + text 

    cv2.imshow('Bib', scale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final_text

