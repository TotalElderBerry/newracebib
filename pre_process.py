import cv2
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import rotate
from deskew import determine_skew
from matplotlib import pyplot as plt
import os

kernel = np.ones((3,3), np.uint8)


# Convert to gray image
def conv_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Rescale(shrink)
def shrink(image):
    return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


# Rescale(enlarge)
def enlarge(image):
    # Convert from numpy to picture
    # Convert from PIL Image to NumPy array
    image_array = np.array(image)

    # Get the size of the image
    height, width = image_array.shape[:2]

    # Define the resizing factor based on your conditions
    val = 1.5 if width < 100 and height < 100 else 1.75

    # Resize the image using OpenCV
    return cv2.resize(image_array, None, fx=val, fy=val, interpolation=cv2.INTER_CUBIC)

# dilation(thin)
def dilate(image):
    return cv2.dilate(image, kernel, iterations=1)


# erosion(thicken)
def erode(image):
    return cv2.erode(image, kernel, iterations=1)

def scale_roi(x,y,w,h,image):
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

# crop image
def crop(image):
    # Convert from numpy to picture
    image = Image.fromarray(image)
    # Size of the image in pixels (size of original image)
    width, height = image.size
    val = .2
    # Setting the points for cropped image
    # Calculate the top and bottom coordinates for cropping
    top = int(height * val)
    bottom = int(height * val)
    left = int(width * .0)
    right = int(width * .0)

    # Cropped image with the specified top and bottom percentages
    cropped = image.crop((left, top, width - right, height - bottom))
    return np.asarray(cropped)


# erosion and followed by dilation
def opening(image):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# Blur to reduce noise
def remove_noise(image):
    return cv2.medianBlur(image, 3)


# Threshold
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Edging
def edging(image):
    return cv2.Canny(image,120,350)

# Gaussian Blur
def gaussian(image):
    return cv2.GaussianBlur(image,(5,5),0)


def deskew(image):
    angle = determine_skew(image)
    # print(angle)
    if angle is None:
        return None
    if angle > -10:
        rotated = rotate(image, angle, resize=True) * 255
        return rotated.astype(np.uint8)
    return image

def pre_image(image):
    image = enlarge(image)
    image = crop(image)
    gamma = 1
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255
        for i in np.arange(0,256)]).astype("uint8")
    image = cv2.LUT(image,table)
    image = deskew(image)
    # image - cv2.convertScaleAbs(image, alpha=1.95, beta=1)
    image = enlarge(image)
    image = edging(image)
    image = thresholding(image)
    # image = opening(image)
    # image = dilate(image)
    image = gaussian(image)
    image = thresholding(image)
    image = remove_noise(image)
    return image