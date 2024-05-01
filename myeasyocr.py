import easyocr
reader = easyocr.Reader(['en'])

def easy():
    text = ""
    print("in easyocr")
    try:
        # Perform text recognition on the cropped ROI
        result = reader.readtext(r"C:\Users\Gemma\Downloads\custom haar cascade\test_images\test_3\dagan sa kadaugan\336046754_1612344879278918_6900719118654405053_n.jpg")
        print(result)
        # Print the detected text
        for detection in result:
            print(detection[1])  # Access the detected text
    except Exception as e:
        print("Error occurred:", e)

easy()