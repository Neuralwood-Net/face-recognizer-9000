import os
from collections import Counter, defaultdict
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class CelebAlign:
    IMG_SIZE = 64
    BASE_PATH = "/Users/larsankile/GitLocal/face-recognizer-9000"
    training_data = []
    training_labels = []
    TOTAL_EXAMPLES = 100_000

    counts = defaultdict(int)

    def __init__(self):
        self.labels = {}
        c = Counter()
        files_for_label = defaultdict(list)

        with open(os.path.join(self.BASE_PATH, "metafiles", "identity_CelebA.txt"), "r") as f:
            for line in f.readlines():
                name, label = line.split()
                self.labels[name] = int(label)
        
        for path in glob(os.path.join(self.BASE_PATH, "data/img_align_celeba", "*.jpg")):
            f = path.split(os.path.sep)[-1]
            label = self.labels[f]
            files_for_label[label].append(path)
            c.update([label])
    
        self.paths = []
        self.num_labels = 0
        for label, count in c.most_common():
            self.paths.extend(files_for_label[label])
            self.num_labels += 1

            if len(self.paths) > self.TOTAL_EXAMPLES:
                break
            
        np.random.shuffle(self.paths)
            
        print("Number of labels:", self.num_labels)
    
    def make_training_data(self):
        for i, path in enumerate(tqdm(self.paths)):
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            img = img[int((h-w) / 2):h-int((h-w) / 2), :]
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

            f = path.split(os.path.sep)[-1]
            label = self.labels[f]
            self.training_data.append(img)
            self.training_labels.append(label)

            self.counts[label] += 1

            unique = set(self.training_labels)
            class_mapping = {elem: idx for idx, elem in enumerate(unique)}

        savefilename = os.path.join(self.BASE_PATH, f"celebalign_processed_{self.IMG_SIZE}px_{self.TOTAL_EXAMPLES}_horizontal.torch")

        torch.save(
            {
                "x": torch.Tensor(list(self.training_data)).view(-1, self.IMG_SIZE, self.IMG_SIZE) / 255.0,
                "y": torch.Tensor([class_mapping[elem] for elem in self.training_labels]).to(torch.int64),
                "num_classes": self.num_labels,
            },
            savefilename,
        )


celeb = CelebAlign()
celeb.make_training_data()
