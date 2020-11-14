import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .utils import get_unique_predictions


class Recognizer:
    def __init__(self):
        data = torch.load(
            "demo/recognizer/SqueezeNet-1605361529.9021263_cropped.data",
            map_location=torch.device("cpu"),
        )

        squeezenet = models.squeezenet1_1()
        in_chl = squeezenet.classifier[1].in_channels
        squeezenet.classifier[1] = nn.Conv2d(in_chl, 4, 1, 1)
        squeezenet.load_state_dict(data["parameters"])
        squeezenet.eval()

        self.net = squeezenet
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def recognize(self, ims, labels, allow_duplicates=True):
        images = []
        for im in ims:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = im / std - mean
            inp = inp.transpose((2, 0, 1))
            images.append(inp)

        t = torch.Tensor(images)

        if allow_duplicates:
            probs, lab_idx = torch.max(F.softmax(self.net(t), dim=1), dim=1)
            pred_labels = [labels[label_idx] for label_idx in lab_idx]

        else:
            p = F.softmax(self.net(t), dim=1)
            probs, pred_labels = get_unique_predictions(p.detach().numpy(), labels)
        # return p.item(), lab_idx.item()
        # return p, lab_idx
        return probs, pred_labels
