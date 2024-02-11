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
def generate_negative_description_file(folder):
    # open the output file for writing. will overwrite all existing data in there
    with open(f'{folder}.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(folder):
            f.write(f'{folder}/' + filename + '\n')

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

def rename_files(folder_path):
    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    # Filter only image files (assuming all are jpg)
    image_files = [file for file in file_list if file.lower().endswith('.jpg')]

    # Sort the image files to ensure they are renamed in order
    image_files.sort()

    # Iterate through each image file and rename it dynamically
    for i, filename in enumerate(image_files, start=1):
        new_filename = os.path.join(folder_path, f'a{i}.jpg')

        # Construct the new filename
        new_filename = os.path.join(folder_path, f'a{i}.jpg')

        # Rename the file
        os.rename(os.path.join(folder_path, filename), new_filename)

        print(f'Renamed {filename} to {new_filename}')

    print('Renaming complete.')