from pdf2image import convert_from_path
import os
from document_classification import document_classify
poppler_path = r"G:\poppler-23.01.0\Library\bin"
pdf_path = r"G:\Document_Classification\Document.pdf"


pages = convert_from_path(pdf_path=pdf_path, poppler_path=poppler_path)

save_folder = r"G:\Document_Classification"
save_folder2 = r"G:\Document_Classification\Images"


c = 1
for page in pages:
    img_name = f"img-{c}.png"
    page.save(os.path.join(save_folder, img_name), "PNG")
    print(f"Page {c} Classification:")
    img_name_class = document_classify(img_name)
    print(img_name_class)
    img_name2 = img_name_class + ".png"
    page.save(os.path.join(save_folder2, img_name2), "PNG")
    c += 1
