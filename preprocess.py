from itertools import chain
import os
import random
import glob

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

I_TRAIN = []
I_TEST = []
A_TRAIN = []
A_TEST = []

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
        random.shuffle(raw_img_paths)
        for j, raw_img_path in enumerate(raw_img_paths):
            c_image = img2nparray(raw_img_path, image_size)
            if j < NUM_OF_TESTDATA:
                I_TEST.append(np.asarray(c_image))
                A_TEST.append(i)
            else:
                for angle in range(-20, 20, 5):
                    img_r = c_image.rotate(angle)
                    I_TRAIN.append(np.asarray(img_r))
                    A_TRAIN.append(i)
                    img_trains = img_r.transpose(
                        Image.Transpose.FLIP_LEFT_RIGHT)
                    I_TRAIN.append(np.asarray(img_trains))
                    A_TRAIN.append(i)


if __name__ == '__main__':
    classes = ["jiro", "other"]
    processedImages("./", classes, (64, 64))
    np.save("./ramen.npy", (np.array(I_TRAIN), np.array(I_TEST),
            np.array(A_TRAIN), np.array(A_TEST)))
