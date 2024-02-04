import os
from PIL import Image


# reads all the files in the /negative folder and generates neg.txt from them.
# we'll run it manually like this:
# $ python
# Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:21:23) [MSC v.1916 32 bit (Intel)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> from cascadeutils import generate_negative_description_file
# >>> generate_negative_description_file()
# >>> exit()
def generate_negative_description_file():
    # open the output file for writing. will overwrite all existing data in there
    with open('neg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir('n'):
            f.write('n/' + filename + '\n')

def resize_images(input_folder):
    max_width = 1366
    max_height = 768


    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Resize the image if it exceeds the screen resolution
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height))

            # Save the resized image (overwrite the existing image)
            image.save(input_path)
