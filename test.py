import img2pdf

with open('./tiff2pdf.pdf', 'wb') as f:
    f.write(img2pdf.convert('./test.png'))
