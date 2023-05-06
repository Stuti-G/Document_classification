from PIL import Image, ImageDraw, ImageFont
from document_classification import document_classify
import os


image = Image.open("./Data Base/PAN Card/PAN-Card.tiff")
image = image.convert("RGB")

save_folder = r"G:\Document_Classification"
save_folder2 = r"G:\Document_Classification\Images"


c = 1
for page in image:
    img_name = f"img-{c}.png"
    page.save(os.path.join(save_folder, img_name), "PNG")
    print(f"Page {c} Classification:")
    img_name_class = document_classify(img_name)
    print(img_name_class)
    page.save(os.path.join(save_folder2, img_name_class), "PNG")
    c += 1
