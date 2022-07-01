from PIL import Image
import numpy as np

# im = Image.open('tree.jpg')
im = Image.open('my.png').convert("RGB")

pix = im.load()

print(pix)