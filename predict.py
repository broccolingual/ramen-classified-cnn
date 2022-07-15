import glob
import os
import uuid

from keras.models import load_model
import numpy as np
from PIL import Image

from preprocess import img2nparray

image_size = (64, 64)


def load_image(path):
    c_image = img2nparray(path, image_size)
    c_image = np.asarray(c_image)
    c_image = c_image / 255.0
    return c_image


model = load_model("./cnn.h5")
files = glob.glob("./judge/*")
all = len(files)
jiro = 0
for f in files:
    img = load_image(f)
    prd = model.predict(np.array([img]))[0]
    idx = prd.argmax()
    per = int(prd[idx]*100)
    if idx == 0:
        print(f"二郎系ラーメン: {per}%")
        jiro += 1
    elif idx == 1:
        print(f"その他のラーメン: {per}%")
    os.rename(f, os.path.join(os.path.dirname(
        f), f"{idx}_{per}%_{uuid.uuid4()}.jpg"))

print(f"All: {all}")
print(f"Jiro: {round(jiro / all * 100)} %")
