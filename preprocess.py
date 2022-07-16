import glob
from itertools import chain
import os
import random

import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

I_DATA = []
A_DATA = []

NUM_OF_TESTDATA = 30


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def img2nparray(path: str, image_size: tuple):
    image = Image.open(path)
    image = image.convert("RGB")
    c_image = crop_max_square(image)
    c_image = c_image.resize(image_size)
    return c_image


def processedImages(path: str, classes: tuple, image_size: tuple):
    for i, label in enumerate(classes):
        raw_img_dir = os.path.join(path, label)
        ext_list = ["jpg", "png"]
        raw_img_paths = list(chain.from_iterable(
            [f for f in [glob.glob(raw_img_dir + "/*." + ext) for ext in ext_list]]))
        for raw_img_path in raw_img_paths:
            c_image = img2nparray(raw_img_path, image_size)
            for angle in range(-20, 20, 10):
                img_r = c_image.rotate(angle)
                I_DATA.append(np.asarray(img_r))
                A_DATA.append(i)
                img_trains = img_r.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT)
                I_DATA.append(np.asarray(img_trains))
                A_DATA.append(i)


if __name__ == '__main__':
    classes = ["jiro", "other"]
    processedImages("./", classes, (64, 64))
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(I_DATA), np.array(A_DATA), test_size=0.2)
    print(f"{len(x_train)=}, {len(x_test)=}, {len(y_train)=}, {len(y_test)=}")
    np.save("./ramen.npy", (x_train, x_test, y_train, y_test))
