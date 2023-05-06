from PIL import Image
from pytesseract import pytesseract

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# image_path = open(r"E:\Document_Classification\Images", encoding='utf-8')


def extract(img):
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(img)
    return text[:-1]


img = Image.open(r"E:\Document_Classification\Images\Pan Card.png")
print(extract(img))
