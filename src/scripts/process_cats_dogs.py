import os
from collections import Counter, defaultdict
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class CatsDogs:
    IMG_SIZE = 64
    BASE_PATH = "/Users/larsankile/GitLocal/face-recognizer-9000"
    training_data = []
    training_labels = []

    counts = defaultdict(int)

    def __init__(self):
        self.labels = {}
        c = Counter()
        files_for_label = defaultdict(list)
        
        self.paths = []
        for path in glob(os.path.join(self.BASE_PATH, "data/catsanddogs", "*", "*.jpg")):
    
            self.paths.append(path)
            
        np.random.shuffle(self.paths)
                
    def make_training_data(self):
        it = tqdm(self.paths)
        num_none = 0
        for i, path in enumerate(it):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                num_none += 1
                it.set_postfix({"corrupt": num_none})
                continue

            h, w = img.shape

            img = img[int(max(h-w, 0) / 2):h-int(max(h-w, 0) / 2), int(max(w-h, 0) / 2):w-int(max(w-h, 0) / 2)]
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

            label, f = path.split(os.path.sep)[-2:]

            self.training_data.append(img)
            self.training_labels.append(int(label))

            self.counts[label] += 1


        savefilename = os.path.join(self.BASE_PATH, f"catsdogs_processed_{self.IMG_SIZE}px_{len(self.training_data)}_horizontal.torch")

        print("Saving torch object to file")
        torch.save(
            {
                "x": torch.Tensor(list(self.training_data)).view(-1, self.IMG_SIZE, self.IMG_SIZE) / 255.0,
                "y": torch.Tensor(self.training_labels).to(torch.int64),
                "num_classes": 2,
            },
            savefilename,
        )

        print("Done!")


catsdogs = CatsDogs()
catsdogs.make_training_data()
