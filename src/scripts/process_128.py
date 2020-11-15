import os
import math
import re
from glob import glob

import numpy as np
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm


def make_folder_structure():

    BASE_PATH = "/Users/larsankile/GitLocal/face-recognizer-9000"

    labels = set()
    labelmap = {}
    with open(os.path.join(BASE_PATH, "metafiles", "identity_CelebA.txt"), "r") as f:
        for line in f.readlines():
            name, label = line.split()
            labels.add(label)
            labelmap[name] = label
    
    """
    for label in labels:
        os.mkdir(os.path.join(BASE_PATH, "data", "celeba", label))
    """
    
    paths = glob(os.path.join(BASE_PATH, "data/img_align_celeba/*.jpg"))

    for path in tqdm(paths):

        name = path.split("/")[-1]
        label = labelmap[name]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        crop = iaa.CropToFixedSize(height=min([h, w]), width=min([h,w]), position='center')
        img = crop(image=img)
        img = cv2.resize(img, (128, 128))

        cv2.imwrite(os.path.join(BASE_PATH, "data", "celeba", label, name), img)
    

def move_images_to_train_and_val():
    BASE_PATH = "/Users/larsankile/GitLocal/face-recognizer-9000/data/celeba"

    for folder in tqdm(os.listdir(BASE_PATH)):
        if folder in ["train", "val"]:
            continue

        files = glob(BASE_PATH + f"/{folder}/*.jpg")
        np.random.shuffle(files)

        sep_idx = int(math.ceil(len(files) * 0.9))
        train, val = files[:sep_idx], files[sep_idx:]

        new_folder = int(folder) - 1

        for f in train:
            name = f.split("/")[-1]
            os.renames(f, os.path.join(BASE_PATH, "train", folder, name))

        for f in val:
            name = f.split("/")[-1]
            os.renames(f, os.path.join(BASE_PATH, "val", folder, name))


def shift_every_folder():
    BASE_PATH = "/Users/larsankile/GitLocal/face-recognizer-9000/data/celeba"
    paths = glob(BASE_PATH + f"/*/*/*.jpg")

    for path in tqdm(paths):
        parts = path.split("/")
        base = parts[:-2]
        folder = int(parts[-2])
        name = parts[-1]
        os.renames(path, "/" + os.path.join(*base, str(folder - 1), name))


shift_every_folder()